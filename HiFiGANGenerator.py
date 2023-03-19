from typing import List

import torch
from torch import nn, Tensor
from torch.nn import Conv1d, ConvTranspose1d, functional as F
from torch.nn.utils import weight_norm, remove_weight_norm

from commons import init_weights, get_padding

LRELU_SLOPE = 0.1


class HiFiGANGenerator(torch.nn.Module):
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

    super(HiFiGANGenerator, self).__init__()

    self.num_kernels = len(resblock_kernel_sizes)
    self.num_upsamples = len(upsample_rates)
    # initial upsampling layers
    self.conv_pre = Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)
    resblock = ResBlock1 if resblock_type == '1' else ResBlock2

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
    # post convolution layer
    self.conv_posts = nn.ModuleList()

    for i in range(len(self.ups)):
      ch = upsample_initial_channel // (2 ** (i + 1))
      for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
        self.resblocks.append(resblock(ch, k, d))

      if i >= len(self.ups) - 3:
        self.conv_posts.append(Conv1d(ch, 1, 7, 1, padding=3, bias=False))

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
      x = F.leaky_relu(x, LRELU_SLOPE)
      x = self.ups[i](x)
      z_sum = None

      for j in range(self.num_kernels):
        if z_sum is None:
          z_sum = self.resblocks[i * self.num_kernels + j](x)
        else:
          z_sum += self.resblocks[i * self.num_kernels + j](x)

      x = z_sum / self.num_kernels  # type: ignore

    x = F.leaky_relu(x)
    x = self.conv_posts[-1](x)
    x = torch.tanh(x)

    return x

  def hier_forward(self, x, g=None):
    outs = []
    x = self.conv_pre(x)

    if g is not None:
      x = x + self.cond(g)

    for i in range(self.num_upsamples):
      x = F.leaky_relu(x, LRELU_SLOPE)
      x = self.ups[i](x)
      xs = None

      for j in range(self.num_kernels):
        if xs is None:
          xs = self.resblocks[i * self.num_kernels + j](x)
        else:
          xs += self.resblocks[i * self.num_kernels + j](x)

      x = xs / self.num_kernels

      if i >= self.num_upsamples - 3:
        _x = F.leaky_relu(x)
        _x = self.conv_posts[i - self.num_upsamples + 3](_x)
        _x = torch.tanh(_x)
        outs.append(_x)

    return outs

  def remove_weight_norm(self):
    print('Removing weight norm...')
    for l in self.ups:
      remove_weight_norm(l)
    for l in self.resblocks:
      l.remove_weight_norm()


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
