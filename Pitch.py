import math

import torch


class Pitch(torch.nn.Module):
  def __init__(
      self,
      sr=22050,
      w_step=256,
      W=2048,
      tau_max=2048,
      midi_start=5,
      midi_end=85,
      octave_range=12
  ):
    super(Pitch, self).__init__()
    self.sr = sr
    self.w_step = w_step
    self.W = W
    self.tau_max = tau_max

    self.unfold = torch.nn.Unfold(
      (1, self.W),
      1,
      0,
      stride=(1, self.w_step)
    )

    midis = list(range(midi_start, midi_end))
    self.len_midis = len(midis)
    c_ms = torch.tensor([self.midi_to_lag(m, octave_range) for m in midis])

    self.register_buffer('c_ms', c_ms)
    self.register_buffer('c_ms_ceil', torch.ceil(self.c_ms).long())
    self.register_buffer('c_ms_floor', torch.floor(self.c_ms).long())

  def midi_to_lag(self, m: int, octave_range: float = 12):
    """
    converts midi-to-lag, eq. (4)

    Args:
        m: midi
        sr: sample_rate
        octave_range:

    Returns:
        lag: time lag(tau, c(m)) calculated from midi, eq. (4)
    """
    f = 440 * math.pow(2, (m - 69) / octave_range)
    lag = self.sr / f
    return lag

  def yingram_from_cmndf(self, cmndfs: torch.Tensor) -> torch.Tensor:
    """
    yingram calculator from cMNDFs(cumulative Mean Normalized Difference Functions)

    Args:
        cmndfs: torch.Tensor
            calculated cumulative mean normalized difference function
            for details, see models/yin.py or eq. (1) and (2)
        ms: list of midi(int)
        sr: sampling rate

    Returns:
        y:
            calculated batch yingram
    """
    # c_ms = np.asarray([Pitch.midi_to_lag(m, sr) for m in ms])
    # c_ms = torch.from_numpy(c_ms).to(cmndfs.device)

    y = (
            cmndfs[:, self.c_ms_ceil] -
            cmndfs[:, self.c_ms_floor]
        ) / (
            self.c_ms_ceil - self.c_ms_floor
        ).unsqueeze(0) * (
            self.c_ms - self.c_ms_floor
        ).unsqueeze(0) + cmndfs[:, self.c_ms_floor]
    return y

  def yingram(self, x: torch.Tensor):
    """
    calculates yingram from raw audio (multi segment)

    Args:
        x: raw audio, torch.Tensor of shape (t)
        W: yingram Window Size
        tau_max:
        sr: sampling rate
        w_step: yingram bin step size

    Returns:
        yingram: yingram. torch.Tensor of shape (80 x t')

    """
    # x.shape: t -> B,T, B,T = x.shape
    B, T = x.shape

    frames = self.unfold(x.view(B, 1, 1, T))
    frames = frames.permute(0, 2, 1).contiguous().view(-1, self.W)  # [B* frames, W]

    # If not using gpu, or torch not compatible, implemented numpy batch function is still fine
    dfs = differenceFunctionTorch(frames, frames.shape[-1], self.tau_max)
    cmndfs = cumulativeMeanNormalizedDifferenceFunctionTorch(dfs, self.tau_max)

    yingram = self.yingram_from_cmndf(cmndfs)  # [B*frames,F]
    yingram = yingram.view(B, -1, self.len_midis).permute(0, 2, 1)  # [B,F,T]

    return yingram

  def crop_scope(self, x, yin_start, scope_shift):  # x: tensor [B,C,T] #scope_shift: tensor [B]
    return torch.stack([
      x[i, yin_start + scope_shift[i]:yin_start + self.yin_scope + scope_shift[i], :] for i in range(x.shape[0])
    ], dim=0)


def differenceFunctionTorch(xs: torch.Tensor, N, tau_max) -> torch.Tensor:
  """
  pytorch backend batch-wise differenceFunction
  has 1e-4 level error with input shape of (32, 22050*1.5)

  Args:
      xs:
      N:
      tau_max:
  """

  xs = xs.double()
  w = xs.shape[-1]
  tau_max = min(tau_max, w)
  zeros = torch.zeros((xs.shape[0], 1))

  x_cumsum = torch.cat((
    torch.zeros((xs.shape[0], 1), device=xs.device),
    (xs * xs).cumsum(dim=-1, dtype=torch.double)
  ), dim=-1)  # B x w

  size = w + tau_max
  p2 = (size // 32).bit_length()
  nice_numbers = (16, 18, 20, 24, 25, 27, 30, 32)
  size_pad = min(x * 2 ** p2 for x in nice_numbers if x * 2 ** p2 >= size)

  fcs = torch.fft.rfft(xs, n=size_pad, dim=-1)
  convs = torch.fft.irfft(fcs * fcs.conj())[:, :tau_max]
  y1 = torch.flip(x_cumsum[:, w - tau_max + 1:w + 1], dims=[-1])
  y = y1 + x_cumsum[:, w].unsqueeze(-1) - x_cumsum[:, :tau_max] - 2 * convs

  return y


def cumulativeMeanNormalizedDifferenceFunctionTorch(
    dfs: torch.Tensor,
    N,
    eps=1e-8
) -> torch.Tensor:
  arange = torch.arange(1, N, device=dfs.device, dtype=torch.float64)
  cumsum = torch.cumsum(
    dfs[:, 1:], dim=-1, dtype=torch.float64
  ).to(dfs.device)

  cmndfs = dfs[:, 1:] * arange / (cumsum + eps)
  cmndfs = torch.cat(
    (torch.ones(cmndfs.shape[0], 1, device=dfs.device), cmndfs), dim=-1
  )

  return cmndfs
