import math
from typing import Optional

import torch
from torch import nn, Tensor
from torch.nn import Conv1d
from torch.nn import functional as F
from torch.nn.utils import weight_norm, remove_weight_norm

import commons
from commons import init_weights, get_padding
from transforms import piecewise_rational_quadratic_transform

LRELU_SLOPE = 0.1


class LayerNorm(nn.Module):
  def __init__(self, channels: int, eps: float = 1e-5):
    """
    标准的 Layer Normalization 模块，用于对输入进行标准化
    可以有效地缓解神经网络在训练过程中的梯度消失或梯度爆炸问题

    - input: (B, C, T)
    - output: (B, C, T)

    :param channels: 输入的通道数（第二维度）
    :param eps: 用于避免除以 0 的小量
    """

    super().__init__()
    self.channels = channels
    self.eps = eps

    # 模块的 gamma 和 beta 参数通过反向传播来更新
    self.gamma = nn.Parameter(torch.ones(channels))
    self.beta = nn.Parameter(torch.zeros(channels))

  def forward(self, x):
    # 将输入的特征维度转换到最后一个位置
    x = x.transpose(1, -1)
    # 对输入进行标准化，并使用 gamma 和 beta 参数进行缩放和平移
    x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
    # 将特征维度转换回原来的位置
    return x.transpose(1, -1)


class DDSConv(nn.Module):
  """
  Dilated and Depth-Separable Convolution
  深度可分离卷积和空洞卷积
  用于在音频信号处理中进行特征提取和降噪
  """

  def __init__(self, channels, kernel_size, n_layers, p_dropout=0.):
    super().__init__()
    self.channels = channels
    self.kernel_size = kernel_size
    self.n_layers = n_layers
    self.p_dropout = p_dropout

    self.drop = nn.Dropout(p_dropout)
    self.convs_sep = nn.ModuleList()
    self.convs_1x1 = nn.ModuleList()
    self.norms_1 = nn.ModuleList()
    self.norms_2 = nn.ModuleList()

    for i in range(n_layers):
      dilation = kernel_size ** i
      padding = (kernel_size * dilation - dilation) // 2
      self.convs_sep.append(nn.Conv1d(channels, channels, kernel_size,
                                      groups=channels, dilation=dilation, padding=padding
                                      ))
      self.convs_1x1.append(nn.Conv1d(channels, channels, 1))
      self.norms_1.append(LayerNorm(channels))
      self.norms_2.append(LayerNorm(channels))

  def forward(self, x, x_mask, g=None):
    """
    模型是一个由多个 depthwise separable convolution 组成的堆叠，每个卷积层都有一个1x1卷积和一些规范化层

    :param g: 条件输入
    """

    if g is not None:
      x = x + g

    for i in range(self.n_layers):
      # 将掩码应用到输入上
      # 一维深度可分离卷积
      y = self.convs_sep[i](x * x_mask)
      # 归一化
      y = self.norms_1[i](y)
      # 激活函数
      y = F.gelu(y)
      # 1x1的卷积
      y = self.convs_1x1[i](y)
      # 归一化
      y = self.norms_2[i](y)
      # 激活函数
      y = F.gelu(y)
      # 防止过拟合
      y = self.drop(y)
      x = x + y

    return x * x_mask


class WaveNet(torch.nn.Module):
  def __init__(
      self,
      hidden_channels,
      kernel_size,
      dilation_rate,
      n_layers,
      gin_channels=0,
      p_dropout=0
  ):
    """
    Wavenet层，采用权重归一化，且没有输入条件。

         |-----------------------------------------------------------------------------|
         |                                    |-> tanh    -|                           |
    res -|- conv1d(dilation) -> dropout -> + -|            * -> conv1d1x1 -> split -|- + -> res
    g -------------------------------------|  |-> sigmoid -|                        |
    o --------------------------------------------------------------------------- + --------- o

    :param hidden_channels: 隐藏层通道数
    :param kernel_size: 第一卷积层的滤波器卷积核大小
    :param dilation_rate: 逐层增加的扩张率。如果为2，则下4层的扩张率分别为1、2、4、8。
    :param n_layers: wavenet层数
    :param gin_channels: 条件输入通道数
    :param p_dropout: dropout比率
    """

    assert kernel_size % 2 == 1
    super(WaveNet, self).__init__()

    self.hidden_channels = hidden_channels
    self.kernel_size = kernel_size,
    self.dilation_rate = dilation_rate
    self.n_layers = n_layers
    self.gin_channels = gin_channels
    self.p_dropout = p_dropout

    self.in_layers = torch.nn.ModuleList()
    self.res_skip_layers = torch.nn.ModuleList()
    self.drop = nn.Dropout(p_dropout)

    # 初始化条件层
    if gin_channels != 0:
      cond_layer = torch.nn.Conv1d(gin_channels, 2 * hidden_channels * n_layers, 1)
      self.cond_layer = torch.nn.utils.weight_norm(cond_layer, name='weight')

    # 中间层
    for i in range(n_layers):
      dilation = dilation_rate ** i
      padding = int((kernel_size * dilation - dilation) / 2)
      in_layer = torch.nn.Conv1d(
        hidden_channels, 2 * hidden_channels, kernel_size, dilation=dilation, padding=padding
      )
      in_layer = torch.nn.utils.weight_norm(in_layer, name='weight')
      self.in_layers.append(in_layer)

      # last one is not necessary
      if i < n_layers - 1:
        res_skip_channels = 2 * hidden_channels
      else:
        res_skip_channels = hidden_channels

      res_skip_layer = torch.nn.Conv1d(hidden_channels, res_skip_channels, 1)
      res_skip_layer = torch.nn.utils.weight_norm(res_skip_layer, name='weight')
      self.res_skip_layers.append(res_skip_layer)

  def forward(self, x, x_mask, g=None, **kwargs):
    output = torch.zeros_like(x)
    n_channels_tensor = torch.IntTensor([self.hidden_channels])

    if g is not None:
      g = self.cond_layer(g)

    for i in range(self.n_layers):
      x_in = self.in_layers[i](x)
      if g is not None:
        cond_offset = i * 2 * self.hidden_channels
        g_l = g[:, cond_offset:cond_offset + 2 * self.hidden_channels, :]
      else:
        g_l = torch.zeros_like(x_in)

      acts = commons.fused_add_tanh_sigmoid_multiply(
        x_in,
        g_l,
        n_channels_tensor)
      acts = self.drop(acts)

      res_skip_acts = self.res_skip_layers[i](acts)
      if i < self.n_layers - 1:
        res_acts = res_skip_acts[:, :self.hidden_channels, :]
        x = (x + res_acts) * x_mask
        output = output + res_skip_acts[:, self.hidden_channels:, :]
      else:
        output = output + res_skip_acts

    return output * x_mask

  def remove_weight_norm(self):
    if self.gin_channels != 0:
      torch.nn.utils.remove_weight_norm(self.cond_layer)
    for l in self.in_layers:
      torch.nn.utils.remove_weight_norm(l)
    for l in self.res_skip_layers:
      torch.nn.utils.remove_weight_norm(l)


class ResBlock1(torch.nn.Module):
  def __init__(self, channels: int, kernel_size=3, dilation=(1, 3, 5)):
    """
    Residual Block 类型 1。它在每个卷积块中有 3 个卷积层

    Network::

        x -> lrelu -> conv1_1 -> conv1_2 -> conv1_3 -> z -> lrelu -> conv2_1 -> conv2_2 -> conv2_3 -> o -> + -> o
        |--------------------------------------------------------------------------------------------------|

    :param channels: 卷积层的隐藏通道数
    :param kernel_size: 每层卷积滤波器的大小
    :param dilation: 每个卷积块中每个卷积层的扩张值的列表
    """

    super(ResBlock1, self).__init__()

    self.convs1 = nn.ModuleList([
      weight_norm(
        Conv1d(
          channels,
          channels,
          kernel_size,
          1,
          dilation=dilation[0],
          padding=get_padding(kernel_size, dilation[0])
        )
      ),
      weight_norm(
        Conv1d(
          channels,
          channels,
          kernel_size,
          1,
          dilation=dilation[1],
          padding=get_padding(kernel_size, dilation[1])
        )
      ),
      weight_norm(
        Conv1d(
          channels,
          channels,
          kernel_size,
          1,
          dilation=dilation[2],
          padding=get_padding(kernel_size, dilation[2])
        )
      )
    ])
    self.convs1.apply(init_weights)

    self.convs2 = nn.ModuleList([
      weight_norm(
        Conv1d(
          channels,
          channels,
          kernel_size,
          1,
          dilation=1,
          padding=get_padding(kernel_size, 1)
        )
      ),
      weight_norm(
        Conv1d(
          channels,
          channels,
          kernel_size,
          1,
          dilation=1,
          padding=get_padding(kernel_size, 1)
        )
      ),
      weight_norm(
        Conv1d(
          channels,
          channels,
          kernel_size, 1, dilation=1,
          padding=get_padding(kernel_size, 1)
        )
      )
    ])
    self.convs2.apply(init_weights)

  def forward(self, x: Tensor, x_mask=None):
    """
    x: [B, C, T]
    """

    for c1, c2 in zip(self.convs1, self.convs2):
      xt = F.leaky_relu(x, LRELU_SLOPE)

      if x_mask is not None:
        xt = xt * x_mask

      xt = c1(xt)
      xt = F.leaky_relu(xt, LRELU_SLOPE)

      if x_mask is not None:
        xt = xt * x_mask

      xt = c2(xt)
      x = xt + x

    if x_mask is not None:
      x = x * x_mask

    return x

  def remove_weight_norm(self):
    for l in self.convs1:
      remove_weight_norm(l)
    for l in self.convs2:
      remove_weight_norm(l)


class ResBlock2(torch.nn.Module):
  def __init__(self, channels: int, kernel_size=3, dilation=(1, 3)):
    """
    Residual Block Type 1是一个带有3个卷积层的残差块。

    Network::

        x -> lrelu -> conv1-> -> z -> lrelu -> conv2-> o -> + -> o
        |---------------------------------------------------|

    :param channels: 卷积层中隐藏通道的数量
    :param kernel_size: 每层卷积过滤器的大小
    :param dilation: 残差块中每个卷积层的膨胀值列表
    """

    super(ResBlock2, self).__init__()
    self.convs = nn.ModuleList([
      weight_norm(
        Conv1d(
          channels,
          channels,
          kernel_size,
          1,
          dilation=dilation[0],
          padding=get_padding(kernel_size, dilation[0])
        )
      ),
      weight_norm(
        Conv1d(
          channels,
          channels,
          kernel_size,
          1,
          dilation=dilation[1],
          padding=get_padding(kernel_size, dilation[1])
        )
      )
    ])
    self.convs.apply(init_weights)

  def forward(self, x, x_mask=None):
    for c in self.convs:
      xt = F.leaky_relu(x, LRELU_SLOPE)

      if x_mask is not None:
        xt = xt * x_mask

      xt = c(xt)
      x = xt + x

    if x_mask is not None:
      x = x * x_mask

    return x

  def remove_weight_norm(self):
    for l in self.convs:
      remove_weight_norm(l)


class Log(nn.Module):
  def forward(self, x, x_mask, reverse=False, **kwargs):
    if not reverse:
      y = torch.log(torch.clamp_min(x, 1e-5)) * x_mask
      logdet = torch.sum(-y, [1, 2])
      return y, logdet
    else:
      x = torch.exp(x) * x_mask
      return x


class Flip(nn.Module):
  def forward(self, x, *args, reverse=False, **kwargs):
    x = torch.flip(x, [1])

    if not reverse:
      logdet = torch.zeros(x.size(0)).to(dtype=x.dtype, device=x.device)
      return x, logdet
    else:
      return x


class ElementwiseAffine(nn.Module):
  def __init__(self, channels: int):
    """
    基于元素的仿射变换，类似于不使用整体统计的批归一化的替代方法。
    :param channels: 输入张量的通道数
    """

    super().__init__()
    self.channels = channels
    self.m = nn.Parameter(torch.zeros(channels, 1))
    self.logs = nn.Parameter(torch.zeros(channels, 1))

  def forward(self, x, x_mask, reverse=False, **kwargs):
    if not reverse:
      y = self.m + torch.exp(self.logs) * x
      y = y * x_mask
      logdet = torch.sum(self.logs * x_mask, [1, 2])
      return y, logdet
    else:
      x = (x - self.m) * torch.exp(-self.logs) * x_mask
      return x


class ResidualCouplingLayer(nn.Module):
  def __init__(self,
               channels: int,
               hidden_channels: int,
               kernel_size: int,
               dilation_rate: int,
               n_layers: int,
               p_dropout: int = 0,
               gin_channels: int = 0,
               mean_only=False
               ):
    assert channels % 2 == 0, "channels should be divisible by 2"
    super().__init__()

    self.channels = channels
    self.hidden_channels = hidden_channels
    self.kernel_size = kernel_size
    self.dilation_rate = dilation_rate
    self.n_layers = n_layers
    self.half_channels = channels // 2
    self.mean_only = mean_only

    # input layer
    self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
    # coupling layers
    self.enc = WaveNet(
      hidden_channels,
      kernel_size,
      dilation_rate,
      n_layers,
      p_dropout=p_dropout,
      gin_channels=gin_channels
    )

    # 输出层
    # 初始化最后一层为0，使得仿射耦合层在一开始什么都不做，这有助于训练的稳定性
    self.post = nn.Conv1d(hidden_channels, self.half_channels * (2 - mean_only), 1)
    self.post.weight.data.zero_()
    self.post.bias.data.zero_()  # type: ignore

  def forward(self, x: Tensor, x_mask: Tensor, g: Optional[Tensor] = None, reverse=False):
    """
    设置 reverse=True 用于推理。

    :param x: :math:`[B, C, T]`
    :param x_mask: :math:`[B, 1, T]`
    :param g: :math:`[B, C, 1]`
    """

    x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
    h = self.pre(x0) * x_mask
    h = self.enc(h, x_mask, g=g)
    stats = self.post(h) * x_mask

    if not self.mean_only:
      m, logs = torch.split(stats, [self.half_channels] * 2, 1)
    else:
      m = stats
      logs = torch.zeros_like(m)

    if not reverse:
      x1 = m + x1 * torch.exp(logs) * x_mask
      x = torch.cat([x0, x1], 1)
      logdet = torch.sum(logs, [1, 2])
      return x, logdet
    else:
      x1 = (x1 - m) * torch.exp(-logs) * x_mask
      x = torch.cat([x0, x1], 1)
      return x


class ConvFlow(nn.Module):
  def __init__(
      self,
      in_channels: int,
      filter_channels: int,
      kernel_size: int,
      n_layers: int,
      num_bins=10,
      tail_bound=5.0
  ):
    """
    Dilated depth separable convolutional based spline flow.
    使用扩张深度可分离卷积实现

    :param in_channels: 输入张量的通道数
    :param filter_channels: 网络中的通道数
    :param kernel_size: 卷积核大小
    :param n_layers: 卷积层数
    :param num_bins: 样条的数量
    :param tail_bound: PRQT的尾部界限
    """

    super().__init__()
    self.in_channels = in_channels
    self.filter_channels = filter_channels
    self.kernel_size = kernel_size
    self.n_layers = n_layers
    self.num_bins = num_bins
    self.tail_bound = tail_bound
    self.half_channels = in_channels // 2

    self.pre = nn.Conv1d(self.half_channels, filter_channels, 1)
    self.convs = DDSConv(filter_channels, kernel_size, n_layers, p_dropout=0.)
    self.proj = nn.Conv1d(filter_channels, self.half_channels * (num_bins * 3 - 1), 1)
    self.proj.weight.data.zero_()
    self.proj.bias.data.zero_()  # type: ignore

  def forward(self, x, x_mask, g=None, reverse=False):
    x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
    h = self.pre(x0)
    h = self.convs(h, x_mask, g=g)
    h = self.proj(h) * x_mask

    b, c, t = x0.shape
    h = h.reshape(b, c, -1, t).permute(0, 1, 3, 2)  # [b, cx?, t] -> [b, c, t, ?]

    unnormalized_widths = h[..., :self.num_bins] / math.sqrt(self.filter_channels)
    unnormalized_heights = h[..., self.num_bins:2 * self.num_bins] / math.sqrt(self.filter_channels)
    unnormalized_derivatives = h[..., 2 * self.num_bins:]

    x1, logabsdet = piecewise_rational_quadratic_transform(
      x1,
      unnormalized_widths,
      unnormalized_heights,
      unnormalized_derivatives,
      inverse=reverse,
      tails='linear',
      tail_bound=self.tail_bound
    )

    x = torch.cat([x0, x1], 1) * x_mask
    logdet = torch.sum(logabsdet * x_mask, [1, 2])
    if not reverse:
      return x, logdet
    else:
      return x
