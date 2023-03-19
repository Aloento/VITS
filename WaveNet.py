import torch
from torch import nn

import commons


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
