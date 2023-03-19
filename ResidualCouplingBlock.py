from typing import Optional

from torch import nn, Tensor

import modules


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
    self.channels = channels
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
