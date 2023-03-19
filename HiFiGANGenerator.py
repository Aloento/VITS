from typing import List

import torch
from torch import nn, Tensor
from torch.nn import Conv1d, ConvTranspose1d, functional as F
from torch.nn.utils import weight_norm, remove_weight_norm

import modules
from commons import init_weights


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
    x = self.conv_posts[-1](x)
    x = torch.tanh(x)

    return x

  def hier_forward(self, x, g=None):
    outs = []
    x = self.conv_pre(x)

    if g is not None:
      x = x + self.cond(g)

    for i in range(self.num_upsamples):
      x = F.leaky_relu(x, modules.LRELU_SLOPE)
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
