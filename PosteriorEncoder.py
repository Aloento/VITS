from typing import Optional

import torch
from torch import nn, Tensor

import WaveNet
import commons


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
    self.enc = WaveNet.WaveNet(
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
