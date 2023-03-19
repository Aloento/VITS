import math

import torch
from torch import nn
from torch.nn import functional as F

import modules


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
