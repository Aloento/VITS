import math
from typing import Optional, List

import torch
from torch import nn, Tensor
from torch.nn import Conv1d, ConvTranspose1d, Conv2d
from torch.nn import functional as F
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm

import attentions
import commons
import modules
import monotonic_align
from commons import init_weights, get_padding


class StochasticDurationPredictor(nn.Module):
  def __init__(self, in_channels, filter_channels, kernel_size, p_dropout, n_flows=4, gin_channels=0):
    """
    随机持续时间模型
    输入是一段语音信号的mel spectrogram，输出是对每个时刻的语音帧的持续时间的概率分布

    ::
        ## Inference

        x -> TextCondEncoder() -> Flow() -> dr_hat
        noise ----------------------^

        ## Training
                                                                              |---------------------|
        x -> TextCondEncoder() -> + -> PosteriorEncoder() -> split() -> z_u, z_v -> (d - z_u) -> concat() -> Flow() -> noise
        d -> DurCondEncoder()  -> ^                                                    |
        |------------------------------------------------------------------------------|

    :param in_channels: 输入数据的通道数
    :param filter_channels: 隐藏通道数
    :param kernel_size: 卷积层的内核大小
    :param p_dropout: 在训练中应用的Dropout概率
    :param n_flows: flow 块数
    :param gin_channels: 条件张量通道数
    """

    super().__init__()

    self.in_channels = in_channels
    self.filter_channels = filter_channels
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.n_flows = n_flows
    self.gin_channels = gin_channels

    self.log_flow = modules.Log()

    # posterior encoder
    self.flows = nn.ModuleList()
    self.flows.append(modules.ElementwiseAffine(2))

    # n_flows指定了要堆叠多少个流，每个流都由ConvFlow和Flip两个模块构成
    for i in range(n_flows):
      self.flows.append(modules.ConvFlow(2, filter_channels, kernel_size, n_layers=3))
      # 执行变量的交换。它通过将数据张量的第2个维度翻转，通常是时间步。
      # 这种操作可以改变时序的顺序，从而扩大了模型的变换空间，从而提高了建模语音时序结构的能力
      self.flows.append(modules.Flip())

    # condition encoder duration
    self.post_pre = nn.Conv1d(1, filter_channels, 1)
    self.post_proj = nn.Conv1d(filter_channels, filter_channels, 1)
    self.post_convs = modules.DDSConv(filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout)

    # flow layers
    self.post_flows = nn.ModuleList()
    self.post_flows.append(modules.ElementwiseAffine(2))

    for i in range(4):
      self.post_flows.append(modules.ConvFlow(2, filter_channels, kernel_size, n_layers=3))
      self.post_flows.append(modules.Flip())

    # condition encoder text
    self.pre = nn.Conv1d(in_channels, filter_channels, 1)
    self.proj = nn.Conv1d(filter_channels, filter_channels, 1)
    self.convs = modules.DDSConv(filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout)

    if gin_channels != 0:
      self.cond = nn.Conv1d(gin_channels, filter_channels, 1)

  def forward(self, x, x_mask, w=None, g=None, reverse=False, noise_scale=1.0):
    """
    :param x: :math:`[B, C, T]`
    :param x_mask: :math:`[B, 1, T]`
    :param w: :math:`[B, 1, T]`
    :param g: :math:`[B, C]`
    """

    # condition encoder text
    x = torch.detach(x)
    x = self.pre(x)

    if g is not None:
      g = torch.detach(g)
      x = x + self.cond(g)

    x = self.convs(x, x_mask)
    x = self.proj(x) * x_mask

    if not reverse:
      flows = self.flows
      assert w is not None

      # condition encoder duration
      logdet_tot_q = 0
      h_w = self.post_pre(w)
      h_w = self.post_convs(h_w, x_mask)
      h_w = self.post_proj(h_w) * x_mask
      e_q = torch.randn(w.size(0), 2, w.size(2)).to(device=x.device, dtype=x.dtype) * x_mask
      z_q = e_q

      # posterior encoder
      for flow in self.post_flows:
        z_q, logdet_q = flow(z_q, x_mask, g=(x + h_w))
        logdet_tot_q += logdet_q

      z_u, z1 = torch.split(z_q, [1, 1], 1)
      u = torch.sigmoid(z_u) * x_mask
      z0 = (w - u) * x_mask

      # posterior encoder - neg log likelihood
      logdet_tot_q += torch.sum((F.logsigmoid(z_u) + F.logsigmoid(-z_u)) * x_mask, [1, 2])
      logq = torch.sum(-0.5 * (math.log(2 * math.pi) + (e_q ** 2)) * x_mask, [1, 2]) - logdet_tot_q

      logdet_tot = 0
      z0, logdet = self.log_flow(z0, x_mask)
      logdet_tot += logdet
      z = torch.cat([z0, z1], 1)

      # flow layers
      for flow in flows:
        z, logdet = flow(z, x_mask, g=x, reverse=reverse)
        logdet_tot = logdet_tot + logdet

      # flow layers - neg log likelihood
      nll = torch.sum(0.5 * (math.log(2 * math.pi) + (z ** 2)) * x_mask, [1, 2]) - logdet_tot
      return nll + logq  # [b]

    else:
      flows = list(reversed(self.flows))
      flows = flows[:-2] + [flows[-1]]  # remove a useless vflow
      z = torch.randn(x.size(0), 2, x.size(2)).to(device=x.device, dtype=x.dtype) * noise_scale

      for flow in flows:
        z = flow(z, x_mask, g=x, reverse=reverse)

      z0, z1 = torch.split(z, [1, 1], 1)
      logw = z0

      return logw


class DurationPredictor(nn.Module):
  def __init__(
      self,
      in_channels: int,
      filter_channels: int,
      kernel_size: int,
      p_dropout: float,
      gin_channels=0
  ):
    """
    Glow-TTS 预测时长的模型

    ::

        [2 x (conv1d_kxk -> relu -> layer_norm -> dropout)] -> conv1d_1x1 -> durs

    :param in_channels: 输入张量的通道数
    :param filter_channels: 网络的隐藏通道数
    :param kernel_size: 卷积层的卷积核大小
    :param p_dropout: 在每个卷积层后使用的 dropout 比例
    """
    super().__init__()

    self.in_channels = in_channels
    self.filter_channels = filter_channels
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.gin_channels = gin_channels

    self.drop = nn.Dropout(p_dropout)
    self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size // 2)
    self.norm_1 = modules.LayerNorm(filter_channels)
    self.conv_2 = nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size // 2)
    self.norm_2 = modules.LayerNorm(filter_channels)
    # output layer
    self.proj = nn.Conv1d(filter_channels, 1, 1)

    if gin_channels != 0:
      self.cond = nn.Conv1d(gin_channels, in_channels, 1)

  def forward(self, x, x_mask, g=None):
    """
    :param x: :math:`[B, C, T]`
    :param x_mask: :math:`[B, 1, T]`
    :param g: :math:`[B, C, 1]`
    """

    x = torch.detach(x)

    if g is not None:
      g = torch.detach(g)
      x = x + self.cond(g)

    x = self.conv_1(x * x_mask)
    x = torch.relu(x)
    x = self.norm_1(x)
    x = self.drop(x)
    x = self.conv_2(x * x_mask)
    x = torch.relu(x)
    x = self.norm_2(x)
    x = self.drop(x)
    x = self.proj(x * x_mask)

    return x * x_mask


class TextEncoder(nn.Module):
  def __init__(
      self,
      n_vocab: int,
      out_channels: int,
      hidden_channels: int,
      filter_channels: int,
      n_heads: int,
      n_layers: int,
      kernel_size: int,
      p_dropout: float
  ):
    """
    将文本序列中的每个单词转换为向量

    :param n_vocab: 嵌入层中字符的数量
    :param out_channels: 输出通道数
    :param hidden_channels: 隐藏通道数
    :param filter_channels: 卷积层的滤波器通道数
    :param n_heads: 注意力头数
    :param n_layers: Transformer层数
    :param kernel_size: Transformer网络中FFN层的卷积核大小
    :param p_dropout: Dropout 概率
    """

    super().__init__()
    self.n_vocab = n_vocab
    self.out_channels = out_channels
    self.hidden_channels = hidden_channels
    self.filter_channels = filter_channels
    self.n_heads = n_heads
    self.n_layers = n_layers
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout

    # 我们定义了三个神经网络层
    # nn.Embedding 是一个用于将整数编码转换为密集向量表示的层
    # 将输入的离散标记编码成一个隐藏向量表示
    self.emb = nn.Embedding(n_vocab, hidden_channels)
    # 对权重进行初始化，使用正态分布随机初始化权重，均值为0，方差为 hidden_channels ** (幂运算) -0.5。
    nn.init.normal_(self.emb.weight, 0.0, hidden_channels ** -0.5)

    # 多头自注意力机制编码器层
    self.encoder = attentions.RelativePositionTransformer(
      hidden_channels,
      filter_channels,
      n_heads,
      n_layers,
      kernel_size,
      p_dropout
    )

    # 一维卷积层，hidden_channels 是输入通道数，1 是卷积核的大小
    self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

  def forward(self, x: Tensor, x_lengths: Tensor):
    """
    :param x: :math:`[B, T]`
    :param x_lengths: :math:`[B]`
    """
    # 首先对输入 x 进行词嵌入 (self.emb(x))，将单词转换为向量。
    # 然后将其乘以 $\sqrt{hidden_channels}$ 归一化词嵌入向量
    x = self.emb(x) * math.sqrt(self.hidden_channels)  # [b, t, h]

    # 对 x 进行转置，以适应下面 Encoder 层的输入形状
    # 将维度顺序从 (batch_size, sequence_length, hidden_channels)
    #       转为 (batch_size, hidden_channels, sequence_length)。
    x = torch.transpose(x, 1, -1)  # [b, h, t]

    # 根据输入 x_lengths 构造掩码矩阵 x_mask，
    # 用于将填充的位置的注意力权重设置为0，避免填充对注意力计算产生影响
    # 形状为 (batch_size, 1, sequence_length)
    x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)  # [b, 1, t]

    # 对序列进行多头自注意力编码，进行特征提取（残差连接）
    x = self.encoder(x * x_mask, x_mask)

    # 将编码后的序列 x 通过 self.proj 进行投影，特征变换
    # 将 x 的最后一个维度从 hidden_channels 转换为 out_channels * 2
    # 然后将其乘以 x_mask，将填充位置的输出设置为0
    stats = self.proj(x) * x_mask

    # 将 stats 沿着最后一个维度分割为两个部分 高斯分布的均值 m 和 高斯分布的标准差 logs
    # 两个部分是为了用于后续的高斯分布参数计算，用于计算损失函数的均值和标准差
    m, logs = torch.split(stats, self.out_channels, dim=1)
    return x, m, logs, x_mask


class ResidualCouplingBlock(nn.Module):
  """
  Residual Coupling Block 是 Flow-based Models 中的一个组件，用于将输入变换为输出，并保持输出的概率密度函数连续且可导。
  Residual Coupling Block 的主要思想是，将输入向量分成两部分，
  一部分固定不变，另一部分作为变量进行变换，
  最后将变换后的结果与固定的那部分进行合并，从而得到输出。
  在这个过程中，变换函数需要是可逆的，即可以通过输出逆推回输入。
  同时，为了保持概率密度函数连续且可导，变换函数需要满足一定的条件，
  例如，需要是光滑的、可微的、可逆的等。

  Residual Coupling Block 通常包含多个 Residual Coupling Layer，每个 Layer 可以看作是一次变换。
  在 Residual Coupling Layer 中，输入向量被分成两部分，一部分作为固定部分，另一部分进行变换。
  其中，变换函数通常采用一个神经网络实现，通过学习适当的参数来实现输入向量的变换。
  在变换完成后，输出向量与固定部分进行合并，得到最终的输出。
  """

  def __init__(
      self,
      channels: int,
      hidden_channels: int,
      kernel_size: int,
      dilation_rate: int,
      n_layers: int,
      n_flows: int = 4,
      gin_channels: int = 0
  ):
    """
    :param channels: 输入和输出张量通道数
    :param hidden_channels: 隐藏网络通道数
    :param kernel_size: WaveNet 层的内核大小
    :param dilation_rate: WaveNet 层的膨胀率
    :param n_layers: WaveNet 层数
    :param n_flows: Residual Coupling 块的数量
    :param gin_channels: 调节张量的通道数
    """

    super().__init__()

    self.hidden_channels = hidden_channels
    self.kernel_size = kernel_size
    self.dilation_rate = dilation_rate
    self.n_layers = n_layers
    self.n_flows = n_flows
    self.gin_channels = gin_channels

    self.flows = nn.ModuleList()
    for i in range(n_flows):
      self.flows.append(
        modules.ResidualCouplingLayer(
          channels,
          hidden_channels,
          kernel_size,
          dilation_rate,
          n_layers,
          gin_channels=gin_channels,
          mean_only=True
        )
      )
      self.flows.append(modules.Flip())

  def forward(self, x: Tensor, x_mask: Tensor, g: Optional[Tensor] = None, reverse=False):
    """
    设置 reverse=True 用于推理。

    :param x: :math:`[B, C, T]`
    :param x_mask: :math:`[B, 1, T]`
    :param g: :math:`[B, C, 1]`
    """

    if not reverse:
      for flow in self.flows:
        x, _ = flow(x, x_mask, g=g, reverse=reverse)
    else:
      for flow in reversed(self.flows):
        x = flow(x, x_mask, g=g, reverse=reverse)

    return x


class PosteriorEncoder(nn.Module):
  def __init__(
      self,
      in_channels: int,
      out_channels: int,
      hidden_channels: int,
      kernel_size: int,
      dilation_rate: int,
      n_layers: int,
      gin_channels=0
  ):
    """
    后验编码器

    - x -> conv1x1() -> WaveNet() (non-causal) -> conv1x1() -> split() -> [m, s] -> sample(m, s) -> z

    :param in_channels: 输入张量通道数
    :param out_channels: 输出张量通道数
    :param hidden_channels: 隐藏通道数
    :param kernel_size: WaveNet卷积层的内核大小
    :param dilation_rate: WaveNet层的膨胀速率
    :param n_layers: WaveNet层数
    :param gin_channels: 调节张量通道数
    """

    super().__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.hidden_channels = hidden_channels
    self.kernel_size = kernel_size
    self.dilation_rate = dilation_rate
    self.n_layers = n_layers
    self.gin_channels = gin_channels

    self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
    self.enc = modules.WN(
      hidden_channels,
      kernel_size,
      dilation_rate,
      n_layers,
      gin_channels=gin_channels
    )
    self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

  def forward(self, x: Tensor, x_lengths: Tensor, g: Optional[Tensor] = None):
    """
    :param x: :math:`[B, C, T]`
    :param x_lengths: :math:`[B, 1]`
    :param g: :math:`[B, C, 1]`
    """

    x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
    x = self.pre(x) * x_mask
    x = self.enc(x, x_mask, g=g)
    stats = self.proj(x) * x_mask
    m, logs = torch.split(stats, self.out_channels, dim=1)
    z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
    return z, m, logs, x_mask


class HifiganGenerator(torch.nn.Module):
  def __init__(
      self,
      initial_channel: int,
      resblock_type: str,
      resblock_kernel_sizes: List[int],
      resblock_dilation_sizes: List[List[int]],
      upsample_rates: List[int],
      upsample_initial_channel: int,
      upsample_kernel_sizes: List[int],
      gin_channels=0
  ):
    """
    带有多感知场融合（Multi-Receptive Field Fusion, MRF）的HiFiGAN发生器

    Network:
        x -> lrelu -> upsampling_layer -> resblock1_k1x1 -> z1 -> + -> z_sum / #resblocks -> lrelu -> conv_post_7x1 -> tanh -> o
                                             ..          -> zI ---|
                                          resblockN_kNx1 -> zN ---'

    :param initial_channel: 张量的通道数
    :param resblock_type: ResBlock的类型。'1'或'2'
    :param resblock_kernel_sizes: 每个ResBlock的内核大小的列表
    :param resblock_dilation_sizes: 每个ResBlock中每个层的膨胀值的列表
    :param upsample_rates: 每个上采样层的上采样因子（步幅）
    :param upsample_initial_channel: 第一层上采样的通道数。每个连续的上采样层将其除以2
    :param upsample_kernel_sizes: 每个转置卷积的内核大小的列表
    """

    super(HifiganGenerator, self).__init__()

    self.num_kernels = len(resblock_kernel_sizes)
    self.num_upsamples = len(upsample_rates)
    # initial upsampling layers
    self.conv_pre = Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)
    resblock = modules.ResBlock1 if resblock_type == '1' else modules.ResBlock2

    # upsampling layers
    self.ups = nn.ModuleList()
    for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
      self.ups.append(
        weight_norm(
          ConvTranspose1d(
            upsample_initial_channel // (2 ** i),
            upsample_initial_channel // (2 ** (i + 1)),
            k, u,
            # Solve imbalanced gpu memory at multi-gpu distributed training.
            padding=(u // 2 + u % 2),
            output_padding=u % 2
          )
        )
      )

    # MRF blocks
    self.resblocks = nn.ModuleList()
    for i in range(len(self.ups)):
      ch = upsample_initial_channel // (2 ** (i + 1))
      for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
        self.resblocks.append(resblock(ch, k, d))

    # post convolution layer
    # Py 的 for 不是一个作用域（真的是令人叹为观止）
    self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)
    self.ups.apply(init_weights)

    if gin_channels != 0:
      self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

  def forward(self, x: Tensor, g=None):
    """
    :param x: 特征输入 [B, C, T]
    :param g: 全局调节输入
    :return: output waveform. [B, 1, T]
    """

    x = self.conv_pre(x)
    if g is not None:
      x = x + self.cond(g)

    for i in range(self.num_upsamples):
      x = F.leaky_relu(x, modules.LRELU_SLOPE)
      x = self.ups[i](x)
      z_sum = None

      for j in range(self.num_kernels):
        if z_sum is None:
          z_sum = self.resblocks[i * self.num_kernels + j](x)
        else:
          z_sum += self.resblocks[i * self.num_kernels + j](x)

      x = z_sum / self.num_kernels  # type: ignore

    x = F.leaky_relu(x)
    x = self.conv_post(x)
    x = torch.tanh(x)

    return x

  def remove_weight_norm(self):
    print('Removing weight norm...')
    for l in self.ups:
      remove_weight_norm(l)
    for l in self.resblocks:
      l.remove_weight_norm()


class DiscriminatorP(torch.nn.Module):
  def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
    super(DiscriminatorP, self).__init__()
    self.period = period
    self.use_spectral_norm = use_spectral_norm
    norm_f = weight_norm if use_spectral_norm == False else spectral_norm
    self.convs = nn.ModuleList([
      norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
      norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
      norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
      norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
      norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(get_padding(kernel_size, 1), 0))),
    ])
    self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

  def forward(self, x):
    fmap = []

    # 1d to 2d
    b, c, t = x.shape
    if t % self.period != 0:  # pad first
      n_pad = self.period - (t % self.period)
      x = F.pad(x, (0, n_pad), "reflect")
      t = t + n_pad
    x = x.view(b, c, t // self.period, self.period)

    for l in self.convs:
      x = l(x)
      x = F.leaky_relu(x, modules.LRELU_SLOPE)
      fmap.append(x)
    x = self.conv_post(x)
    fmap.append(x)
    x = torch.flatten(x, 1, -1)

    return x, fmap


class DiscriminatorS(torch.nn.Module):
  def __init__(self, use_spectral_norm=False):
    super(DiscriminatorS, self).__init__()
    norm_f = weight_norm if use_spectral_norm == False else spectral_norm
    self.convs = nn.ModuleList([
      norm_f(Conv1d(1, 16, 15, 1, padding=7)),
      norm_f(Conv1d(16, 64, 41, 4, groups=4, padding=20)),
      norm_f(Conv1d(64, 256, 41, 4, groups=16, padding=20)),
      norm_f(Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
      norm_f(Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
      norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
    ])
    self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

  def forward(self, x):
    fmap = []

    for l in self.convs:
      x = l(x)
      x = F.leaky_relu(x, modules.LRELU_SLOPE)
      fmap.append(x)
    x = self.conv_post(x)
    fmap.append(x)
    x = torch.flatten(x, 1, -1)

    return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
  def __init__(self, use_spectral_norm=False):
    """
    鉴别器包装一个尺度鉴别器和一堆周期鉴别器

    ::
        waveform -> ScaleDiscriminator() -> scores_sd, feats_sd --> append() -> scores, feats
               |--> MultiPeriodDiscriminator() -> scores_mpd, feats_mpd ^

    :param use_spectral_norm: 如果为True，则切换到谱范数而不是权重范数
    """

    super(MultiPeriodDiscriminator, self).__init__()
    periods = [2, 3, 5, 7, 11]

    self.discriminators = nn.ModuleList()
    self.discriminators.append(DiscriminatorS(use_spectral_norm=use_spectral_norm))
    self.discriminators.extend([DiscriminatorP(i, use_spectral_norm=use_spectral_norm) for i in periods])

  def forward(self, y, y_hat):
    """
    :param y: 实际的音频波形
    :param y_hat: 预测的音频波形
    :return:
      List[Tensor]: 判别器分数。
      List[List[Tensor]]: 每个判别器的每个层的特征的列表。
    """

    # 实际和预测波形通过每个判别器后的分数
    scores = []
    hat_scores = []
    # 每个判别器的每个层的特征
    feats = []
    hat_feats = []

    for net in self.discriminators:
      y_d_r, fmap_r = net(y)
      scores.append(y_d_r)
      feats.append(fmap_r)

      y_d_g, fmap_g = net(y_hat)
      hat_scores.append(y_d_g)
      hat_feats.append(fmap_g)

    return scores, hat_scores, feats, hat_feats


class SynthesizerTrn(nn.Module):
  """
  Synthesizer for Training
  """

  def __init__(
      self,
      num_chars: int,
      spec_channels: int,
      segment_size: int,
      inter_channels: int,
      hidden_channels: int,
      filter_channels: int,
      n_heads: int,
      n_layers: int,
      kernel_size: int,
      p_dropout: float,
      resblock_type,
      resblock_kernel_sizes,
      resblock_dilation_sizes,
      upsample_rates,
      upsample_initial_channel,
      upsample_kernel_sizes,
      n_speakers=0,
      gin_channels=0,
      use_sdp=True,
      **kwargs
  ):
    """
    :param num_chars: 词汇表的大小
    :param spec_channels: 声谱图的通道数
    :param segment_size: 分段的大小
    :param inter_channels: 编码器和解码器之间的通道数
    :param hidden_channels: 模型中的隐藏通道数
    :param filter_channels: 每个卷积层中的过滤器数
    :param n_heads: 多头自注意力中的头数
    :param n_layers: 编码器和解码器中的卷积层数量
    :param kernel_size: 卷积核的大小
    :param p_dropout: dropout概率
    :param resblock_type: 是否使用残差块
    :param resblock_kernel_sizes: 残差块中的卷积核大小
    :param resblock_dilation_sizes: 残差块中的空洞卷积核大小
    :param upsample_rates: 上采样率
    :param upsample_initial_channel: 上采样前的通道数
    :param upsample_kernel_sizes: 上采样层中的卷积核大小
    :param n_speakers: 说话人的数量（对于单说话人语音合成为0）
    :param gin_channels: 全局条件嵌入的大小
    :param use_sdp: 是否使用随机持续时间预测器
    :param kwargs: 任意数量的关键字参数
    """

    super().__init__()
    self.n_vocab = num_chars
    self.spec_channels = spec_channels
    self.inter_channels = inter_channels
    self.hidden_channels = hidden_channels
    self.filter_channels = filter_channels
    self.n_heads = n_heads
    self.n_layers = n_layers
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.resblock = resblock_type
    self.resblock_kernel_sizes = resblock_kernel_sizes
    self.resblock_dilation_sizes = resblock_dilation_sizes
    self.upsample_rates = upsample_rates
    self.upsample_initial_channel = upsample_initial_channel
    self.upsample_kernel_sizes = upsample_kernel_sizes
    self.segment_size = segment_size
    self.n_speakers = n_speakers
    self.gin_channels = gin_channels
    self.use_sdp = use_sdp

    self.text_encoder = TextEncoder(
      num_chars,
      inter_channels,
      hidden_channels,
      filter_channels,
      n_heads,
      n_layers,
      kernel_size,
      p_dropout
    )

    self.waveform_decoder = HifiganGenerator(
      inter_channels,
      resblock_type,
      resblock_kernel_sizes,
      resblock_dilation_sizes,
      upsample_rates,
      upsample_initial_channel,
      upsample_kernel_sizes,
      gin_channels=gin_channels
    )

    self.posterior_encoder = PosteriorEncoder(
      spec_channels,
      inter_channels,
      hidden_channels,
      5, 1, 16,
      gin_channels=gin_channels
    )

    self.flow = ResidualCouplingBlock(
      inter_channels,
      hidden_channels,
      5, 1, 4,
      gin_channels=gin_channels
    )

    if use_sdp:
      self.duration_predictor = StochasticDurationPredictor(
        hidden_channels, 192, 3, 0.5, 4, gin_channels=gin_channels
      )
    else:
      self.duration_predictor = DurationPredictor(  # type: ignore
        hidden_channels, 256, 3, 0.5, gin_channels=gin_channels
      )

    if n_speakers > 1:
      self.emb_g = nn.Embedding(n_speakers, gin_channels)

  def forward(self, x, x_lengths, y, y_lengths, sid=None):
    x, m_p, logs_p, x_mask = self.text_encoder(x, x_lengths)

    if self.n_speakers > 0:
      g = self.emb_g(sid).unsqueeze(-1)  # [b, h, 1]
    else:
      g = None

    z, m_q, logs_q, y_mask = self.posterior_encoder(y, y_lengths, g=g)
    z_p = self.flow(z, y_mask, g=g)

    with torch.no_grad():
      # negative cross-entropy
      s_p_sq_r = torch.exp(-2 * logs_p)  # [b, d, t]
      neg_cent1 = torch.sum(-0.5 * math.log(2 * math.pi) - logs_p, [1], keepdim=True)  # [b, 1, t_s]
      neg_cent2 = torch.matmul(-0.5 * (z_p ** 2).transpose(1, 2), s_p_sq_r)  # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
      neg_cent3 = torch.matmul(z_p.transpose(1, 2), (m_p * s_p_sq_r))  # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
      neg_cent4 = torch.sum(-0.5 * (m_p ** 2) * s_p_sq_r, [1], keepdim=True)  # [b, 1, t_s]
      neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4

      attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
      attn = monotonic_align.maximum_path(neg_cent, attn_mask.squeeze(1)).unsqueeze(1).detach()

    w = attn.sum(2)
    if self.use_sdp:
      l_length = self.duration_predictor(x, x_mask, w, g=g)
      l_length = l_length / torch.sum(x_mask)
    else:
      logw_ = torch.log(w + 1e-6) * x_mask
      logw = self.duration_predictor(x, x_mask, g=g)
      l_length = torch.sum((logw - logw_) ** 2, [1, 2]) / torch.sum(x_mask)  # for averaging

    # expand prior
    m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
    logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)

    z_slice, ids_slice = commons.rand_slice_segments(z, y_lengths, self.segment_size)
    o = self.waveform_decoder(z_slice, g=g)
    return o, l_length, attn, ids_slice, x_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q)

  def infer(self, x, x_lengths, sid=None, noise_scale=1, length_scale=1, noise_scale_w=1., max_len=None):
    x, m_p, logs_p, x_mask = self.text_encoder(x, x_lengths)

    if self.n_speakers > 0:
      g = self.emb_g(sid).unsqueeze(-1)  # [b, h, 1]
    else:
      g = None

    if self.use_sdp:
      logw = self.duration_predictor(x, x_mask, g=g, reverse=True, noise_scale=noise_scale_w)
    else:
      logw = self.duration_predictor(x, x_mask, g=g)

    w = torch.exp(logw) * x_mask * length_scale
    w_ceil = torch.ceil(w)
    y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
    y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, None), 1).to(x_mask.dtype)
    attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
    attn = commons.generate_path(w_ceil, attn_mask)

    m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)  # [b, t', t], [b, t, d] -> [b, d, t']
    logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)  # [b, t', t], [b, t, d] -> [b, d, t']

    z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
    z = self.flow(z_p, y_mask, g=g, reverse=True)
    o = self.waveform_decoder((z * y_mask)[:, :, :max_len], g=g)

    return o, attn, y_mask, (z, z_p, m_p, logs_p)

  def voice_conversion(self, y, y_lengths, sid_src, sid_tgt):
    assert self.n_speakers > 0, "n_speakers have to be larger than 0."

    g_src = self.emb_g(sid_src).unsqueeze(-1)
    g_tgt = self.emb_g(sid_tgt).unsqueeze(-1)
    z, m_q, logs_q, y_mask = self.posterior_encoder(y, y_lengths, g=g_src)
    z_p = self.flow(z, y_mask, g=g_src)
    z_hat = self.flow(z_p, y_mask, g=g_tgt, reverse=True)
    o_hat = self.waveform_decoder(z_hat * y_mask, g=g_tgt)

    return o_hat, y_mask, (z, z_p, z_hat)
