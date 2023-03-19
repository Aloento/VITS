from typing import List

import numpy as np
import torch
from scipy.signal.windows import kaiser
from torch import nn
from torch.nn import Conv1d, functional as F
from torch.nn.utils import weight_norm, spectral_norm

from commons import get_padding


class CoMBDBlock(torch.nn.Module):
  def __init__(
      self,
      h_u: List[int],
      d_k: List[int],
      d_s: List[int],
      d_d: List[int],
      d_g: List[int],
      d_p: List[int],
      op_f: int,
      op_k: int,
      op_g: int,
      use_spectral_norm=False,
  ):

    super(CoMBDBlock, self).__init__()
    norm_f = weight_norm if use_spectral_norm is False else spectral_norm

    self.convs = nn.ModuleList()
    filters = [[1, h_u[0]]]

    for i in range(len(h_u) - 1):
      filters.append([h_u[i], h_u[i + 1]])

    for _f, _k, _s, _d, _g, _p in zip(filters, d_k, d_s, d_d, d_g, d_p):
      self.convs.append(
        norm_f(  # type: ignore
          Conv1d(
            in_channels=_f[0],
            out_channels=_f[1],
            kernel_size=_k,
            stride=_s,
            dilation=_d,
            groups=_g,
            padding=_p
          )
        )
      )

    self.projection_conv = norm_f(  # type: ignore
      Conv1d(
        in_channels=filters[-1][1],
        out_channels=op_f,
        kernel_size=op_k,
        groups=op_g
      )
    )

  def forward(self, x, b_y, b_y_hat):
    fmap_r = []
    fmap_g = []

    for block in self.convs:
      x = block(x)
      x = F.leaky_relu(x, 0.2)
      f_r, f_g = x.split([b_y, b_y_hat], dim=0)
      fmap_r.append(f_r.tile([2, 1, 1]) if b_y < b_y_hat else f_r)
      fmap_g.append(f_g)

    x = self.projection_conv(x)
    x_r, x_g = x.split([b_y, b_y_hat], dim=0)

    return x_r.tile([2, 1, 1]) if b_y < b_y_hat else x_r, x_g, fmap_r, fmap_g


class CoMBD(torch.nn.Module):
  def __init__(self):
    super(CoMBD, self).__init__()

    self.pqmf_list = nn.ModuleList([
      PQMF(4, 192, 0.13, 10.0),  # lv2
      PQMF(2, 256, 0.25, 10.0)  # lv1
    ])

    combd_h_u = [[16, 64, 256, 1024, 1024, 1024] for _ in range(3)]
    combd_d_k = [[7, 11, 11, 11, 11, 5], [11, 21, 21, 21, 21, 5],
                 [15, 41, 41, 41, 41, 5]]

    combd_d_s = [[1, 1, 4, 4, 4, 1] for _ in range(3)]
    combd_d_d = [[1, 1, 1, 1, 1, 1] for _ in range(3)]
    combd_d_g = [[1, 4, 16, 64, 256, 1] for _ in range(3)]

    combd_d_p = [[3, 5, 5, 5, 5, 2], [5, 10, 10, 10, 10, 2],
                 [7, 20, 20, 20, 20, 2]]

    combd_op_f = [1, 1, 1]
    combd_op_k = [3, 3, 3]
    combd_op_g = [1, 1, 1]

    self.blocks = nn.ModuleList()
    for _h_u, _d_k, _d_s, _d_d, _d_g, _d_p, _op_f, _op_k, _op_g in zip(
        combd_h_u,
        combd_d_k,
        combd_d_s,
        combd_d_d,
        combd_d_g,
        combd_d_p,
        combd_op_f,
        combd_op_k,
        combd_op_g,
    ):
      self.blocks.append(
        CoMBDBlock(
          _h_u,
          _d_k,
          _d_s,
          _d_d,
          _d_g,
          _d_p,
          _op_f,
          _op_k,
          _op_g,
        ))

  @staticmethod
  def _block_forward(ys, ys_hat, blocks):
    outs_real = []
    outs_fake = []
    f_maps_real = []
    f_maps_fake = []

    # y:B, y_hat: 2B if i!=-1 else B,B
    for y, y_hat, block in zip(ys, ys_hat, blocks):
      b_y = y.shape[0]
      b_y_hat = y_hat.shape[0]
      cat_y = torch.cat([y, y_hat], dim=0)
      out_real, out_fake, f_map_r, f_map_g = block(cat_y, b_y, b_y_hat)
      outs_real.append(out_real)
      outs_fake.append(out_fake)
      f_maps_real.append(f_map_r)
      f_maps_fake.append(f_map_g)

    return outs_real, outs_fake, f_maps_real, f_maps_fake

  def _pqmf_forward(self, ys, ys_hat):
    # preprocess for multi_scale forward
    multi_scale_inputs_hat = []
    for pqmf_ in self.pqmf_list:
      multi_scale_inputs_hat.append(pqmf_.analysis(ys_hat[-1])[:, :1, :])

    # real
    # for hierarchical forward
    # outs_real_, f_maps_real_ = self._block_forward(
    #    ys, self.blocks)

    # for multi_scale forward
    # outs_real, f_maps_real = self._block_forward(
    #        ys[:-1], self.blocks[:-1], outs_real, f_maps_real)
    # outs_real.extend(outs_real[:-1])
    # f_maps_real.extend(f_maps_real[:-1])

    # outs_real = [torch.cat([o,o], dim=0) if i!=len(outs_real_)-1 else o for i,o in enumerate(outs_real_)]
    # f_maps_real = [[torch.cat([fmap,fmap], dim=0) if i!=len(f_maps_real_)-1 else fmap for fmap in fmaps ] \
    #        for i,fmaps in enumerate(f_maps_real_)]

    inputs_fake = [
      torch.cat([y, multi_scale_inputs_hat[i]], dim=0)
      if i != len(ys_hat) - 1 else y for i, y in enumerate(ys_hat)
    ]

    outs_real, outs_fake, f_maps_real, f_maps_fake = self._block_forward(ys, inputs_fake, self.blocks)

    # predicted
    # for hierarchical forward
    # outs_fake, f_maps_fake = self._block_forward(
    #    inputs_fake, self.blocks)

    # outs_real_, f_maps_real_ = self._block_forward(
    #    ys, self.blocks)
    # for multi_scale forward
    # outs_fake, f_maps_fake = self._block_forward(
    #    multi_scale_inputs_hat, self.blocks[:-1], outs_fake, f_maps_fake)

    return outs_real, outs_fake, f_maps_real, f_maps_fake

  def forward(self, ys, ys_hat):
    outs_real, outs_fake, f_maps_real, f_maps_fake = self._pqmf_forward(ys, ys_hat)
    return outs_real, outs_fake, f_maps_real, f_maps_fake


class MDC(torch.nn.Module):
  def __init__(
      self,
      in_channels,
      out_channels,
      strides,
      kernel_size,
      dilations,
      use_spectral_norm=False
  ):
    super(MDC, self).__init__()
    norm_f = weight_norm if not use_spectral_norm else spectral_norm
    self.d_convs = nn.ModuleList()

    for _k, _d in zip(kernel_size, dilations):
      self.d_convs.append(
        norm_f(
          Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=_k,
            dilation=_d,
            padding=get_padding(_k, _d)
          )
        )
      )

    self.post_conv = norm_f(
      Conv1d(
        in_channels=out_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=strides,
        padding=get_padding(_k, _d)
      )
    )

    self.softmax = torch.nn.Softmax(dim=-1)

  def forward(self, x):
    _out = None

    for _l in self.d_convs:
      _x = torch.unsqueeze(_l(x), -1)
      _x = F.leaky_relu(_x, 0.2)
      _out = torch.cat([_out, _x], dim=-1) if _out is not None else _x

    x = torch.sum(_out, dim=-1)
    x = self.post_conv(x)
    x = F.leaky_relu(x, 0.2)  # @@

    return x


class SBDBlock(torch.nn.Module):
  def __init__(
      self,
      segment_dim,
      strides,
      filters,
      kernel_size,
      dilations,
      use_spectral_norm=False
  ):
    super(SBDBlock, self).__init__()
    norm_f = weight_norm if not use_spectral_norm else spectral_norm
    self.convs = nn.ModuleList()
    filters_in_out = [(segment_dim, filters[0])]

    for i in range(len(filters) - 1):
      filters_in_out.append([filters[i], filters[i + 1]])

    for _s, _f, _k, _d in zip(strides, filters_in_out, kernel_size, dilations):
      self.convs.append(
        MDC(
          in_channels=_f[0],
          out_channels=_f[1],
          strides=_s,
          kernel_size=_k,
          dilations=_d,
          use_spectral_norm=use_spectral_norm)
      )

    self.post_conv = norm_f(
      Conv1d(
        in_channels=_f[1],
        out_channels=1,
        kernel_size=3,
        stride=1,
        padding=3 // 2
      )
    )  # @@

  def forward(self, x):
    fmap_r = []
    fmap_g = []

    for _l in self.convs:
      x = _l(x)
      f_r, f_g = torch.chunk(x, 2, dim=0)
      fmap_r.append(f_r)
      fmap_g.append(f_g)

    x = self.post_conv(x)  # @@
    x_r, x_g = torch.chunk(x, 2, dim=0)

    return x_r, x_g, fmap_r, fmap_g


class MDCDConfig:
  def __init__(self):
    self.pqmf_params = [16, 256, 0.03, 10.0]
    self.f_pqmf_params = [64, 256, 0.1, 9.0]

    self.filters = [[64, 128, 256, 256, 256], [64, 128, 256, 256, 256],
                    [64, 128, 256, 256, 256], [32, 64, 128, 128, 128]]

    self.kernel_sizes = [[[7, 7, 7], [7, 7, 7], [7, 7, 7], [7, 7, 7], [7, 7, 7]],
                         [[5, 5, 5], [5, 5, 5], [5, 5, 5], [5, 5, 5], [5, 5, 5]],
                         [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
                         [[5, 5, 5], [5, 5, 5], [5, 5, 5], [5, 5, 5], [5, 5, 5]]]

    self.dilations = [[[5, 7, 11], [5, 7, 11], [5, 7, 11], [5, 7, 11], [5, 7, 11]],
                      [[3, 5, 7], [3, 5, 7], [3, 5, 7], [3, 5, 7], [3, 5, 7]],
                      [[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]],
                      [[1, 2, 3], [1, 2, 3], [1, 2, 3], [2, 3, 5], [2, 3, 5]]]

    self.strides = [[1, 1, 3, 3, 1], [1, 1, 3, 3, 1], [1, 1, 3, 3, 1], [1, 1, 3, 3, 1]]

    self.band_ranges = [[0, 6], [0, 11], [0, 16], [0, 64]]
    self.transpose = [False, False, False, True]
    self.segment_size = 8192


class SBD(torch.nn.Module):
  def __init__(self, use_spectral_norm=False):
    super(SBD, self).__init__()
    self.config = MDCDConfig()
    self.pqmf = PQMF(*self.config.pqmf_params)

    if True in self.config.transpose:
      self.f_pqmf = PQMF(*self.config.f_pqmf_params)
    else:
      self.f_pqmf = None

    self.discriminators = torch.nn.ModuleList()

    for _f, _k, _d, _s, _br, _tr in zip(
        self.config.filters,
        self.config.kernel_sizes,
        self.config.dilations,
        self.config.strides,
        self.config.band_ranges,
        self.config.transpose
    ):
      if _tr:
        segment_dim = self.config.segment_size // _br[1] - _br[0]
      else:
        segment_dim = _br[1] - _br[0]

      self.discriminators.append(
        SBDBlock(
          segment_dim=segment_dim,
          filters=_f,
          kernel_size=_k,
          dilations=_d,
          strides=_s,
          use_spectral_norm=use_spectral_norm
        )
      )

  def forward(self, y, y_hat):
    y_d_rs = []
    y_d_gs = []
    fmap_rs = []
    fmap_gs = []

    y_in = self.pqmf.analysis(y)
    y_hat_in = self.pqmf.analysis(y_hat)
    y_in_f = self.f_pqmf.analysis(y)
    y_hat_in_f = self.f_pqmf.analysis(y_hat)

    for d, br, tr in zip(self.discriminators, self.config.band_ranges, self.config.transpose):
      if not tr:
        _y_in = y_in[:, br[0]:br[1], :]
        _y_hat_in = y_hat_in[:, br[0]:br[1], :]
      else:
        _y_in = y_in_f[:, br[0]:br[1], :]
        _y_hat_in = y_hat_in_f[:, br[0]:br[1], :]
        _y_in = torch.transpose(_y_in, 1, 2)
        _y_hat_in = torch.transpose(_y_hat_in, 1, 2)

      # y_d_r, fmap_r = d(_y_in)
      # y_d_g, fmap_g = d(_y_hat_in)

      cat_y = torch.cat([_y_in, _y_hat_in], dim=0)
      y_d_r, y_d_g, fmap_r, fmap_g = d(cat_y)
      y_d_rs.append(y_d_r)
      fmap_rs.append(fmap_r)
      y_d_gs.append(y_d_g)
      fmap_gs.append(fmap_g)

    return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class AvocodoDiscriminator(nn.Module):
  def __init__(self, use_spectral_norm=False):
    super(AvocodoDiscriminator, self).__init__()
    self.combd = CoMBD()
    self.sbd = SBD(use_spectral_norm)

  def forward(self, y, ys_hat):
    ys = [
      self.combd.pqmf_list[0].analysis(y)[:, :1],  # lv2
      self.combd.pqmf_list[1].analysis(y)[:, :1],  # lv1
      y
    ]

    y_c_rs, y_c_gs, fmap_c_rs, fmap_c_gs = self.combd(ys, ys_hat)
    y_s_rs, y_s_gs, fmap_s_rs, fmap_s_gs = self.sbd(y, ys_hat[-1])
    y_c_rs.extend(y_s_rs)
    y_c_gs.extend(y_s_gs)
    fmap_c_rs.extend(fmap_s_rs)
    fmap_c_gs.extend(fmap_s_gs)

    return y_c_rs, y_c_gs, fmap_c_rs, fmap_c_gs


def design_prototype_filter(taps=62, cutoff_ratio=0.142, beta=9.0):
  """
  Design prototype filter for PQMF.
  This method is based on `A Kaiser window approach for the design of prototype
  filters of cosine modulated filterbanks`_.
  Args:
      taps (int): The number of filter taps.
      cutoff_ratio (float): Cut-off frequency ratio.
      beta (float): Beta coefficient for kaiser window.
  Returns:
      ndarray: Impluse response of prototype filter (taps + 1,).
  .. _`A Kaiser window approach for the design of prototype filters of cosine modulated filterbanks`:
      https://ieeexplore.ieee.org/abstract/document/681427
  """

  # check the arguments are valid
  assert taps % 2 == 0, "The number of taps mush be even number."
  assert 0.0 < cutoff_ratio < 1.0, "Cutoff ratio must be > 0.0 and < 1.0."

  # make initial filter
  omega_c = np.pi * cutoff_ratio
  with np.errstate(invalid="ignore"):
    h_i = np.sin(omega_c * (np.arange(taps + 1) - 0.5 * taps)) / (
        np.pi * (np.arange(taps + 1) - 0.5 * taps)
    )
  h_i[taps // 2] = np.cos(0) * cutoff_ratio  # fix nan due to indeterminate form

  # apply kaiser window
  w = kaiser(taps + 1, beta)
  h = h_i * w

  return h


class PQMF(torch.nn.Module):
  """
  PQMF module.
  This module is based on `Near-perfect-reconstruction pseudo-QMF banks`_.
  .. _`Near-perfect-reconstruction pseudo-QMF banks`:
      https://ieeexplore.ieee.org/document/258122
  """

  def __init__(self, subbands=4, taps=62, cutoff_ratio=0.142, beta=9.0):
    """
    Initilize PQMF module.
    The cutoff_ratio and beta parameters are optimized for #subbands = 4.
    See dicussion in https://github.com/kan-bayashi/ParallelWaveGAN/issues/195.
    Args:
        subbands (int): The number of subbands.
        taps (int): The number of filter taps.
        cutoff_ratio (float): Cut-off frequency ratio.
        beta (float): Beta coefficient for kaiser window.
    """
    super(PQMF, self).__init__()

    # build analysis & synthesis filter coefficients
    h_proto = design_prototype_filter(taps, cutoff_ratio, beta)
    h_analysis = np.zeros((subbands, len(h_proto)))
    h_synthesis = np.zeros((subbands, len(h_proto)))
    for k in range(subbands):
      h_analysis[k] = (
          2
          * h_proto
          * np.cos(
            (2 * k + 1)
            * (np.pi / (2 * subbands))
            * (np.arange(taps + 1) - (taps / 2))
            + (-1) ** k * np.pi / 4
          )
      )
      h_synthesis[k] = (
          2
          * h_proto
          * np.cos(
            (2 * k + 1)
            * (np.pi / (2 * subbands))
            * (np.arange(taps + 1) - (taps / 2))
            - (-1) ** k * np.pi / 4
          )
      )

    # convert to tensor
    analysis_filter = torch.Tensor(h_analysis).float().unsqueeze(1)
    synthesis_filter = torch.Tensor(h_synthesis).float().unsqueeze(0)

    # register coefficients as beffer
    self.register_buffer("analysis_filter", analysis_filter)
    self.register_buffer("synthesis_filter", synthesis_filter)

    # filter for downsampling & upsampling
    updown_filter = torch.zeros((subbands, subbands, subbands)).float()

    for k in range(subbands):
      updown_filter[k, k, 0] = 1.0

    self.register_buffer("updown_filter", updown_filter)
    self.subbands = subbands

    # keep padding info
    self.pad_fn = torch.nn.ConstantPad1d(taps // 2, 0.0)

  def analysis(self, x):
    """
    Analysis with PQMF.
    Args:
        x (Tensor): Input tensor (B, 1, T).
    Returns:
        Tensor: Output tensor (B, subbands, T // subbands).
    """
    x = F.conv1d(self.pad_fn(x), self.analysis_filter)
    return F.conv1d(x, self.updown_filter, stride=self.subbands)

  def synthesis(self, x):
    """
    Synthesis with PQMF.
    Args:
        x (Tensor): Input tensor (B, subbands, T // subbands).
    Returns:
        Tensor: Output tensor (B, 1, T).
    """
    # NOTE(kan-bayashi): Power will be dreased so here multipy by # subbands.
    #   Not sure this is the correct way, it is better to check again.
    # TODO(kan-bayashi): Understand the reconstruction procedure
    x = F.conv_transpose1d(
      x, self.updown_filter * self.subbands, stride=self.subbands
    )
    return F.conv1d(self.pad_fn(x), self.synthesis_filter)
