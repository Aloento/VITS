import torch
import torch.utils.data
from librosa.filters import mel as librosa_mel_fn

MAX_WAV_VALUE = 32768.0


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
  """
  对输入的张量进行动态范围压缩。
  对输入的张量取对数并乘以一个压缩因子C，再通过 torch.clamp 函数将张量的最小值限制在 clip_val 以及 C 与张量最大值的乘积之间

  :param C: compression factor
  """
  return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
  """
  对已经压缩的张量进行解压缩操作。具体来说，就是对输入的张量先取指数，再除以压缩因子C

  :param C: compression factor used to compress
  """
  return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
  """
  对输入的矩阵进行归一化处理
  """
  output = dynamic_range_compression_torch(magnitudes)
  return output


def spectral_de_normalize_torch(magnitudes):
  """
  对已经归一化的矩阵进行解归一化操作
  """
  output = dynamic_range_decompression_torch(magnitudes)
  return output


mel_basis = {}
hann_window = {}


def spectrogram_torch(y, n_fft, hop_size, win_size, center=False):
  """
  在短时傅里叶变换（STFT）中，hop_size（或称为跳跃长度或步长）是每次计算FFT之间移动的采样点数。
  具体而言，STFT将信号分割为重叠的帧，每个帧的长度为n_fft，相邻帧之间的距离为hop_size。
  hop_size的值通常是n_fft的一小部分，以确保帧之间有足够的重叠，从而避免在信号的边缘出现伪像。

  当center=True时，填充信号以在两端对称地包围原始信号，并在计算STFT之前将其居中。
  当center=False时，填充信号在左侧（前）进行，因此原始信号保留在右侧（后）。
  在这里，使用了center=False的默认值，因为反射填充已经将信号的两端对称地扩展了。

  :param y: 音频信号y
  :param n_fft: STFT 使用的窗口长度，通常为2的幂次方
  :param hop_size: 帧移的长度，即相邻帧之间的采样点数
  :param win_size: 窗口长度
  :param center: 信号序列是否应居中填充
  """

  if torch.min(y) < -1.:
    print('min value is ', torch.min(y))
  if torch.max(y) > 1.:
    print('max value is ', torch.max(y))

  # 为了避免重复创建相同大小和数据类型的窗口
  # 汉宁窗口是一种对称的、具有平滑过渡的窗口，形状类似于一个向两端逐渐变细的钟形曲线，
  # 可以抑制信号频谱中的泄漏现象，使信号在傅里叶变换后的频谱图中更加清晰。
  global hann_window
  dtype_device = str(y.dtype) + '_' + str(y.device)
  wnsize_dtype_device = str(win_size) + '_' + dtype_device

  if wnsize_dtype_device not in hann_window:
    # 创建一个汉宁窗口
    hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(dtype=y.dtype, device=y.device)

  # 对y进行反射填充以适应FFT窗口大小
  # 在y张量的第二个维度上添加填充，使其在输入到STFT函数之前，它的长度扩展到一个n_fft帧长的整数倍。
  # 填充的大小为(n_fft - hop_size) / 2。
  # 这是为了确保STFT计算时输入信号的长度与帧长n_fft之间的比例是整数，因为STFT计算通常要求输入的信号长度是帧长的整数倍。
  # 填充模式为'reflect'，这意味着填充的值是原始信号在边界处的镜像。
  # 例如，如果原始信号的最后一个样本是y[-1]，则在边界填充时，填充的第一个值将为y[-2]，第二个值将为y[-3]，以此类推。
  y = torch.nn.functional.pad(y.unsqueeze(1), (
    int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)), mode='reflect')

  # 去除张量中维度为1的维度，将输入张量从二维降至一维
  y = y.squeeze(1)

  # normalized: 输出 STFT 是否应该被标准化
  # onesided: 输出 STFT 是否应该被截断为单边谱
  # 函数的输出是一个大小为 (channel, freq, time, complex) 的四维张量 spec，
  # 其中 channel 是 STFT 的通道数，通常为 1（表示单声道音频），
  # freq 是 STFT 的频率通道数，time 是 STFT 的时间帧数。
  # complex 是每个频率通道上的复数值，表示每个时间帧上的幅度和相位。
  spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[wnsize_dtype_device],
                    center=center, pad_mode='reflect', normalized=False, onesided=True)

  # 计算了每个时间步的频率幅度的平方和，并对这个和加上一个极小值以防止被零整除，然后再对它们进行开方。
  # 最终，它生成一个只包含频率幅度信息的张量，表示为频谱图
  spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
  return spec


def spec_to_mel_torch(spec, n_fft, num_mels, sampling_rate, fmin, fmax):
  """
  将频谱图（spectrogram）转换为梅尔频谱图（Mel spectrogram）
  :param spec: 频谱图
  :param n_fft: FFT大小
  :param num_mels: Mel滤波器个数
  :param sampling_rate: 采样率
  :param fmin: 最低频率
  :param fmax: 最高频率
  """

  # mel-basis 矩阵是一个对数滤波器组，对输入信号进行频域分解
  global mel_basis
  dtype_device = str(spec.dtype) + '_' + str(spec.device)
  fmax_dtype_device = str(fmax) + '_' + dtype_device

  # 判断当前的数据类型和设备是否已经在mel_basis中存在对应的梅尔滤波器
  if fmax_dtype_device not in mel_basis:
    # 使用 librosa 库将线性频谱转换为梅尔频谱，返回一个梅尔滤波器组成的矩阵
    mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
    mel_basis[fmax_dtype_device] = torch.from_numpy(mel).to(dtype=spec.dtype, device=spec.device)

  # 将输入的 spectrogram 与 mel-basis 矩阵相乘，得到 mel-spectrogram
  spec = torch.matmul(mel_basis[fmax_dtype_device], spec)
  # 对 mel-spectrogram 进行幅度谱的归一化，使得所有的幅度值都在 [0, 1] 范围内。这有助于神经网络训练过程的稳定性和收敛速度
  spec = spectral_normalize_torch(spec)

  return spec


def mel_spectrogram_torch(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
  """
  从音频信号计算Mel spectrogram
  TODO：合并到上面两个方法中

  :param y: 输入的音频信号，作为一个1维张量
  :param n_fft: FFT大小
  :param num_mels: 输出Mel频率图的通道数
  :param sampling_rate: 输入音频信号的采样率
  :param hop_size: 时间跨度
  :param win_size: 窗口大小
  :param fmin: filterbank的最低频率
  :param fmax: filterbank的最高频率
  :param center: 在中心上进行填充
  """

  if torch.min(y) < -1.:
    print('min value is ', torch.min(y))
  if torch.max(y) > 1.:
    print('max value is ', torch.max(y))

  # 已在上面两个方法中解释
  global mel_basis, hann_window
  dtype_device = str(y.dtype) + '_' + str(y.device)
  fmax_dtype_device = str(fmax) + '_' + dtype_device
  wnsize_dtype_device = str(win_size) + '_' + dtype_device

  # spec_to_mel_torch
  if fmax_dtype_device not in mel_basis:
    mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
    mel_basis[fmax_dtype_device] = torch.from_numpy(mel).to(dtype=y.dtype, device=y.device)

  # spectrogram_torch
  if wnsize_dtype_device not in hann_window:
    hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(dtype=y.dtype, device=y.device)

  y = torch.nn.functional.pad(y.unsqueeze(1), (
    int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)), mode='reflect')
  y = y.squeeze(1)

  spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[wnsize_dtype_device],
                    center=center, pad_mode='reflect', normalized=False, onesided=True)

  spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)

  spec = torch.matmul(mel_basis[fmax_dtype_device], spec)
  spec = spectral_normalize_torch(spec)

  return spec
