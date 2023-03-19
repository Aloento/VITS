from typing import List, Optional

import torch
from torch import Tensor
from torch.nn import functional as F


def init_weights(m, mean=0.0, std=0.01):
  classname = m.__class__.__name__
  if classname.find("Conv") != -1:
    m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
  return int((kernel_size * dilation - dilation) / 2)


def intersperse_with_language_id(text, lang, item):
  n = len(text)
  _text = [item] * (2 * n + 1)
  _lang = [None] * (2 * n + 1)
  _text[1::2] = text
  _lang[1::2] = lang
  _lang[::2] = lang + [lang[-1]]

  return _text, _lang


def slice_segments(x, ids_str, segment_size=4):
  ret = torch.zeros_like(x[:, :, :segment_size])

  for i in range(x.size(0)):
    idx_str = ids_str[i]
    idx_end = idx_str + segment_size
    ret[i] = x[i, :, idx_str:idx_end]

  return ret


def rand_slice_segments_for_cat(x, x_lengths=None, segment_size=4):
  b, d, t = x.size()
  if x_lengths is None:
    x_lengths = t

  ids_str_max = x_lengths - segment_size + 1
  ids_str = torch.rand([b // 2]).to(device=x.device)

  ids_str = (
      torch.cat([ids_str, ids_str], dim=0) * ids_str_max
  ).to(dtype=torch.long)

  ids_str = torch.max(torch.zeros(ids_str.size()).to(ids_str.device), ids_str).to(dtype=torch.long)
  ret = slice_segments(x, ids_str, segment_size)

  return ret, ids_str


def subsequent_mask(length):
  mask = torch.tril(torch.ones(length, length)).unsqueeze(0).unsqueeze(0)
  return mask


@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
  n_channels_int = n_channels[0]
  in_act = input_a + input_b
  t_act = torch.tanh(in_act[:, :n_channels_int, :])
  s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
  acts = t_act * s_act
  return acts


def convert_pad_shape(pad_shape: List[List[int]]):
  l = pad_shape[::-1]
  pad_shape = [item for sublist in l for item in sublist]
  return pad_shape


def sequence_mask(length: Tensor, max_length: Optional[int] = None):
  """
  生成一个二维的序列掩码，用于在序列张量中过滤填充。
  这个掩码是用来屏蔽输入中填充的位置的。在自然语言处理中，经常需要对变长的文本序列进行处理，
  因为不同的句子长度可能不同，需要对其进行填充，使得所有句子都具有相同的长度。
  但是，由于填充的位置是没有实际意义的，所以在进行模型计算时需要将填充位置的信息屏蔽掉。这就是这个掩码的作用。

  :param length: 序列长度
  :param max_length: 最大序列长度
  :return: :math:`[B, T_max]`
  """

  if max_length is None:
    max_length = length.max()  # type: ignore

  # 生成一个从 0 到 max_length-1 的序列 x，然后将其重复 batch_size 次，形状变为 (batch_size, max_length)。
  # 接下来，将 length 扩展为形状为 (batch_size, 1) 的张量，并将其与 x 逐元素比较，
  # 生成一个布尔型张量，表示所有小于 length 中对应元素的位置为 True，否则为 False。
  x = torch.arange(max_length, dtype=length.dtype, device=length.device)  # type: ignore
  return x.unsqueeze(0) < length.unsqueeze(1)


def generate_path(duration, mask):
  """
  duration: [b, 1, t_x]
  mask: [b, 1, t_y, t_x]
  """
  device = duration.device

  b, _, t_y, t_x = mask.shape
  cum_duration = torch.cumsum(duration, -1)

  cum_duration_flat = cum_duration.view(b * t_x)
  path = sequence_mask(cum_duration_flat, t_y).to(mask.dtype)
  path = path.view(b, t_x, t_y)
  path = path - F.pad(path, convert_pad_shape([[0, 0], [1, 0], [0, 0]]))[:, :-1]
  path = path.unsqueeze(1).transpose(2, 3) * mask

  return path


def clip_grad_value_(parameters, clip_value, norm_type=2):
  if isinstance(parameters, torch.Tensor):
    parameters = [parameters]

  parameters = list(filter(lambda p: p.grad is not None, parameters))
  norm_type = float(norm_type)

  if clip_value is not None:
    clip_value = float(clip_value)

  total_norm = 0

  for p in parameters:
    param_norm = p.grad.data.norm(norm_type)
    total_norm += param_norm.item() ** norm_type

    if clip_value is not None:
      p.grad.data.clamp_(min=-clip_value, max=clip_value)

  total_norm = total_norm ** (1. / norm_type)
  return total_norm
