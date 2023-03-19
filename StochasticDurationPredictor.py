import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import ResidualCouplingBlock
from LayerNorm import LayerNorm


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

    self.log_flow = Log()

    # posterior encoder
    self.flows = nn.ModuleList()
    self.flows.append(ElementwiseAffine(2))

    # n_flows指定了要堆叠多少个流，每个流都由ConvFlow和Flip两个模块构成
    for i in range(n_flows):
      self.flows.append(ConvFlow(2, filter_channels, kernel_size, n_layers=3))
      # 执行变量的交换。它通过将数据张量的第2个维度翻转，通常是时间步。
      # 这种操作可以改变时序的顺序，从而扩大了模型的变换空间，从而提高了建模语音时序结构的能力
      self.flows.append(ResidualCouplingBlock.Flip())

    # condition encoder duration
    self.post_pre = nn.Conv1d(1, filter_channels, 1)
    self.post_proj = nn.Conv1d(filter_channels, filter_channels, 1)
    self.post_convs = DDSConv(filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout)

    # flow layers
    self.post_flows = nn.ModuleList()
    self.post_flows.append(ElementwiseAffine(2))

    for i in range(4):
      self.post_flows.append(ConvFlow(2, filter_channels, kernel_size, n_layers=3))
      self.post_flows.append(ResidualCouplingBlock.Flip())

    # condition encoder text
    self.pre = nn.Conv1d(in_channels, filter_channels, 1)
    self.proj = nn.Conv1d(filter_channels, filter_channels, 1)
    self.convs = DDSConv(filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout)

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


class Log(nn.Module):
  def forward(self, x, x_mask, reverse=False, **kwargs):
    if not reverse:
      y = torch.log(torch.clamp_min(x, 1e-5)) * x_mask
      logdet = torch.sum(-y, [1, 2])
      return y, logdet
    else:
      x = torch.exp(x) * x_mask
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
      self.convs_sep.append(
        nn.Conv1d(
          channels,
          channels,
          kernel_size,
          groups=channels,
          dilation=dilation,
          padding=padding
        )
      )
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


DEFAULT_MIN_BIN_WIDTH = 1e-3
DEFAULT_MIN_BIN_HEIGHT = 1e-3
DEFAULT_MIN_DERIVATIVE = 1e-3


def piecewise_rational_quadratic_transform(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    inverse=False,
    tails=None,
    tail_bound=1.,
    min_bin_width=DEFAULT_MIN_BIN_WIDTH,
    min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
    min_derivative=DEFAULT_MIN_DERIVATIVE
):
  if tails is None:
    spline_fn = rational_quadratic_spline
    spline_kwargs = {}
  else:
    spline_fn = unconstrained_rational_quadratic_spline
    spline_kwargs = {
      'tails': tails,
      'tail_bound': tail_bound
    }

  outputs, logabsdet = spline_fn(
    inputs=inputs,
    unnormalized_widths=unnormalized_widths,
    unnormalized_heights=unnormalized_heights,
    unnormalized_derivatives=unnormalized_derivatives,
    inverse=inverse,
    min_bin_width=min_bin_width,
    min_bin_height=min_bin_height,
    min_derivative=min_derivative,
    **spline_kwargs
  )
  return outputs, logabsdet


def searchsorted(bin_locations, inputs, eps=1e-6):
  bin_locations[..., -1] += eps
  return torch.sum(
    inputs[..., None] >= bin_locations,
    dim=-1
  ) - 1


def unconstrained_rational_quadratic_spline(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    inverse=False,
    tails='linear',
    tail_bound=1.,
    min_bin_width=DEFAULT_MIN_BIN_WIDTH,
    min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
    min_derivative=DEFAULT_MIN_DERIVATIVE
):
  inside_interval_mask = (inputs >= -tail_bound) & (inputs <= tail_bound)
  outside_interval_mask = ~inside_interval_mask

  outputs = torch.zeros_like(inputs)
  logabsdet = torch.zeros_like(inputs)

  if tails == 'linear':
    unnormalized_derivatives = F.pad(unnormalized_derivatives, pad=(1, 1))
    constant = np.log(np.exp(1 - min_derivative) - 1)
    unnormalized_derivatives[..., 0] = constant
    unnormalized_derivatives[..., -1] = constant

    outputs[outside_interval_mask] = inputs[outside_interval_mask]
    logabsdet[outside_interval_mask] = 0
  else:
    raise RuntimeError('{} tails are not implemented.'.format(tails))

  outputs[inside_interval_mask], logabsdet[inside_interval_mask] = rational_quadratic_spline(
    inputs=inputs[inside_interval_mask],
    unnormalized_widths=unnormalized_widths[inside_interval_mask, :],
    unnormalized_heights=unnormalized_heights[inside_interval_mask, :],
    unnormalized_derivatives=unnormalized_derivatives[inside_interval_mask, :],
    inverse=inverse,
    left=-tail_bound, right=tail_bound, bottom=-tail_bound, top=tail_bound,
    min_bin_width=min_bin_width,
    min_bin_height=min_bin_height,
    min_derivative=min_derivative
  )

  return outputs, logabsdet


def rational_quadratic_spline(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    inverse=False,
    left=0., right=1., bottom=0., top=1.,
    min_bin_width=DEFAULT_MIN_BIN_WIDTH,
    min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
    min_derivative=DEFAULT_MIN_DERIVATIVE
):
  if torch.min(inputs) < left or torch.max(inputs) > right:
    raise ValueError('Input to a transform is not within its domain')

  num_bins = unnormalized_widths.shape[-1]

  if min_bin_width * num_bins > 1.0:
    raise ValueError('Minimal bin width too large for the number of bins')
  if min_bin_height * num_bins > 1.0:
    raise ValueError('Minimal bin height too large for the number of bins')

  widths = F.softmax(unnormalized_widths, dim=-1)
  widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
  cumwidths = torch.cumsum(widths, dim=-1)
  cumwidths = F.pad(cumwidths, pad=(1, 0), mode='constant', value=0.0)
  cumwidths = (right - left) * cumwidths + left
  cumwidths[..., 0] = left
  cumwidths[..., -1] = right
  widths = cumwidths[..., 1:] - cumwidths[..., :-1]

  derivatives = min_derivative + F.softplus(unnormalized_derivatives)

  heights = F.softmax(unnormalized_heights, dim=-1)
  heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
  cumheights = torch.cumsum(heights, dim=-1)
  cumheights = F.pad(cumheights, pad=(1, 0), mode='constant', value=0.0)
  cumheights = (top - bottom) * cumheights + bottom
  cumheights[..., 0] = bottom
  cumheights[..., -1] = top
  heights = cumheights[..., 1:] - cumheights[..., :-1]

  if inverse:
    bin_idx = searchsorted(cumheights, inputs)[..., None]
  else:
    bin_idx = searchsorted(cumwidths, inputs)[..., None]

  input_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0]
  input_bin_widths = widths.gather(-1, bin_idx)[..., 0]

  input_cumheights = cumheights.gather(-1, bin_idx)[..., 0]
  delta = heights / widths
  input_delta = delta.gather(-1, bin_idx)[..., 0]

  input_derivatives = derivatives.gather(-1, bin_idx)[..., 0]
  input_derivatives_plus_one = derivatives[..., 1:].gather(-1, bin_idx)[..., 0]

  input_heights = heights.gather(-1, bin_idx)[..., 0]

  if inverse:
    a = (((inputs - input_cumheights) * (input_derivatives
                                         + input_derivatives_plus_one
                                         - 2 * input_delta)
          + input_heights * (input_delta - input_derivatives)))
    b = (input_heights * input_derivatives
         - (inputs - input_cumheights) * (input_derivatives
                                          + input_derivatives_plus_one
                                          - 2 * input_delta))
    c = - input_delta * (inputs - input_cumheights)

    discriminant = b.pow(2) - 4 * a * c
    assert (discriminant >= 0).all()

    root = (2 * c) / (-b - torch.sqrt(discriminant))
    outputs = root * input_bin_widths + input_cumwidths

    theta_one_minus_theta = root * (1 - root)
    denominator = input_delta + ((input_derivatives + input_derivatives_plus_one - 2 * input_delta)
                                 * theta_one_minus_theta)
    derivative_numerator = input_delta.pow(2) * (input_derivatives_plus_one * root.pow(2)
                                                 + 2 * input_delta * theta_one_minus_theta
                                                 + input_derivatives * (1 - root).pow(2))
    logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)

    return outputs, -logabsdet
  else:
    theta = (inputs - input_cumwidths) / input_bin_widths
    theta_one_minus_theta = theta * (1 - theta)

    numerator = input_heights * (input_delta * theta.pow(2)
                                 + input_derivatives * theta_one_minus_theta)
    denominator = input_delta + ((input_derivatives + input_derivatives_plus_one - 2 * input_delta)
                                 * theta_one_minus_theta)
    outputs = input_cumheights + numerator / denominator

    derivative_numerator = input_delta.pow(2) * (input_derivatives_plus_one * theta.pow(2)
                                                 + 2 * input_delta * theta_one_minus_theta
                                                 + input_derivatives * (1 - theta).pow(2))
    logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)

    return outputs, logabsdet
