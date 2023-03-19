import torch
from torch import nn

import LayerNorm


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
    self.norm_1 = LayerNorm.LayerNorm(filter_channels)
    self.conv_2 = nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size // 2)
    self.norm_2 = LayerNorm.LayerNorm(filter_channels)
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
