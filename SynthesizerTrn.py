import math
from typing import List

import torch
from torch import nn, Tensor
from torch.nn import functional as F

import commons
from DurationPredictor import DurationPredictor
from HiFiGANGenerator import HiFiGANGenerator
from Pitch import Pitch
from PosteriorEncoder import PosteriorEncoder
from ResidualCouplingBlock import ResidualCouplingBlock
from StochasticDurationPredictor import StochasticDurationPredictor
from TextEncoder import TextEncoder
from YingDecoder import YingDecoder
from monotonic_align import maximum_path


class SynthesizerTrn(nn.Module):
  """
  Synthesizer for Training
  """

  def __init__(
      self,
      num_chars: int,
      spec_channels: int,
      segment_size: int,
      midi_start,
      midi_end,
      octave_range,
      inter_channels: int,
      hidden_channels: int,
      filter_channels: int,
      n_heads: int,
      n_layers: int,
      kernel_size: int,
      p_dropout: float,
      resblock: str,
      resblock_kernel_sizes: List[int],
      resblock_dilation_sizes: List[List[int]],
      upsample_rates: List[int],
      upsample_initial_channel: int,
      upsample_kernel_sizes: List[int],
      yin_channels,
      yin_start,
      yin_scope,
      yin_shift_range,
      n_speakers=0,
      gin_channels=0,
      use_sdp=True,
      **kwargs
  ):
    """
    :param num_chars: 词汇表的大小
    :param spec_channels: 声谱图的通道数
    :param segment_size: 分段的大小
    :param inter_channels: 编码器和解码器之间的通道数
    :param hidden_channels: 模型中的隐藏通道数
    :param filter_channels: 每个卷积层中的过滤器数
    :param n_heads: 多头自注意力中的头数
    :param n_layers: 编码器和解码器中的卷积层数量
    :param kernel_size: 卷积核的大小
    :param p_dropout: dropout概率
    :param resblock: ResBlock的类型。'1'或'2'
    :param resblock_kernel_sizes: 每个ResBlock的内核大小的列表
    :param resblock_dilation_sizes: 每个ResBlock中每个层的膨胀值的列表
    :param upsample_rates: 每个上采样层的上采样因子（步幅）
    :param upsample_initial_channel: 第一层上采样的通道数。每个连续的上采样层将其除以2
    :param upsample_kernel_sizes: 每个转置卷积的内核大小的列表
    :param n_speakers: 说话人的数量（对于单说话人语音合成为0）
    :param gin_channels: 全局条件嵌入的大小
    :param use_sdp: 是否使用随机持续时间预测器
    :param kwargs: 任意数量的关键字参数
    """

    super().__init__()
    self.n_vocab = num_chars
    self.spec_channels = spec_channels
    self.inter_channels = inter_channels
    self.hidden_channels = hidden_channels
    self.filter_channels = filter_channels
    self.n_heads = n_heads
    self.n_layers = n_layers
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.resblock = resblock
    self.resblock_kernel_sizes = resblock_kernel_sizes
    self.resblock_dilation_sizes = resblock_dilation_sizes
    self.upsample_rates = upsample_rates
    self.upsample_initial_channel = upsample_initial_channel
    self.upsample_kernel_sizes = upsample_kernel_sizes
    self.segment_size = segment_size
    self.n_speakers = n_speakers
    self.gin_channels = gin_channels
    self.use_sdp = use_sdp

    self.yin_channels = yin_channels
    self.yin_start = yin_start
    self.yin_scope = yin_scope

    self.text_encoder = TextEncoder(
      num_chars,
      inter_channels,
      hidden_channels,
      filter_channels,
      n_heads,
      n_layers,
      kernel_size,
      p_dropout
    )

    self.waveform_decoder = HiFiGANGenerator(
      inter_channels - yin_channels +
      yin_scope,
      resblock,
      resblock_kernel_sizes,
      resblock_dilation_sizes,
      upsample_rates,
      upsample_initial_channel,
      upsample_kernel_sizes,
      gin_channels=gin_channels
    )

    self.posterior_encoder = PosteriorEncoder(
      spec_channels,
      inter_channels - yin_channels,
      inter_channels - yin_channels,
      5, 1, 16,
      gin_channels=gin_channels
    )

    self.pitch_encoder = PosteriorEncoder(
      yin_channels,
      yin_channels,
      yin_channels,
      5, 1, 16,
      gin_channels=gin_channels
    )

    self.flow = ResidualCouplingBlock(
      inter_channels,
      hidden_channels,
      5, 1, 4,
      gin_channels=gin_channels
    )

    if use_sdp:
      self.duration_predictor = StochasticDurationPredictor(
        hidden_channels, 192, 3, 0.5, 4, gin_channels=gin_channels
      )
    else:
      self.duration_predictor = DurationPredictor(  # type: ignore
        hidden_channels, 256, 3, 0.5, gin_channels=gin_channels
      )

    self.yin_decoder = YingDecoder(
      yin_scope,
      5, 1, 4,
      yin_start,
      yin_scope,
      yin_shift_range,
      gin_channels=gin_channels
    )

    self.emb_g = nn.Embedding(self.n_speakers, gin_channels)
    self.pitch = Pitch(
      midi_start=midi_start,
      midi_end=midi_end,
      octave_range=octave_range
    )

  def crop_scope(self, x: list, scope_shift=0):  # TODO: need to modify for non-scalar shift
    return [
      i[:, self.yin_start + scope_shift:self.yin_start + self.yin_scope + scope_shift, :] for i in x
    ]

  def crop_scope_tensor(
      self, x: Tensor,  # x: [B,C,T]
      scope_shift: Tensor  # scope_shift: tensor [B]
  ):
    return torch.stack([
      x[i, self.yin_start + scope_shift[i]:self.yin_start + self.yin_scope + scope_shift[i], :] for i in
      range(x.shape[0])
    ], dim=0)

  def yin_dec_infer(self, z_yin, z_mask, sid=None):
    if self.n_speakers > 0:
      g = self.emb_g(sid).unsqueeze(-1)  # [b, h, 1]
    else:
      g = None
    return self.yin_decoder.infer(z_yin, z_mask, g)

  def forward(self, x, t, x_lengths, y, y_lengths, ying, ying_lengths, sid=None, scope_shift=0):
    # 对输入序列进行编码
    x, m_p, logs_p, x_mask = self.text_encoder(x, t, x_lengths)

    if self.n_speakers > 0:
      # 获取一个说话人的嵌入 g 并对其进行变换，以便可以将其与后验编码器的输出相结合
      g = self.emb_g(sid).unsqueeze(-1)  # [b, h, 1]
    else:
      g = None

    z_spec, m_spec, logs_spec, spec_mask = self.posterior_encoder(y, y_lengths, g=g)

    z_yin, m_yin, logs_yin, yin_mask = self.pitch_encoder(ying, y_lengths, g=g)
    z_yin_crop, logs_yin_crop, m_yin_crop = self.crop_scope([z_yin, logs_yin, m_yin], scope_shift)

    # yin dec loss
    yin_gt_crop, yin_gt_shifted_crop, yin_dec_crop, z_yin_crop_shifted, scope_shift = self.yin_decoder(
      z_yin, ying, yin_mask, g)

    z = torch.cat([z_spec, z_yin], dim=1)
    logs_q = torch.cat([logs_spec, logs_yin], dim=1)
    m_q = torch.cat([m_spec, m_yin], dim=1)

    # 通过 Spline Flow 将输出 z_spec 从后验分布转换为先验分布
    z_p = self.flow(z, spec_mask, g=g)

    z_dec = torch.cat([z_spec, z_yin_crop], dim=1)

    z_dec_shifted = torch.cat([z_spec.detach(), z_yin_crop_shifted], dim=1)
    z_dec_ = torch.cat([z_dec, z_dec_shifted], dim=0)

    # 计算注意力分布。此部分使用负交叉熵损失来计算注意力分布，以便在生成语音时自动学习对齐模式。
    # 计算了4个不同的贡献，然后将它们相加以获得最终负交叉熵损失。
    # 注意分布在这里被计算并返回。
    with torch.no_grad():
      # negative cross-entropy
      # [b, d, t]
      s_p_sq_r = torch.exp(-2 * logs_p)
      # [b, 1, t_s]
      neg_cent1 = torch.sum(-0.5 * math.log(2 * math.pi) - logs_p, [1], keepdim=True)
      # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s], z_p: [b,d,t]
      neg_cent2 = torch.einsum('bdt, bds -> bts', -0.5 * (z_p ** 2), s_p_sq_r)
      # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
      neg_cent3 = torch.einsum('bdt, bds -> bts', z_p, (m_p * s_p_sq_r))
      # [b, 1, t_s]
      neg_cent4 = torch.sum(-0.5 * (m_p ** 2) * s_p_sq_r, [1], keepdim=True)
      neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4

      attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(spec_mask, -1)
      attn = maximum_path(neg_cent, attn_mask.squeeze(1)).unsqueeze(1).detach()

    w = attn.sum(2)
    # 时长预测
    # 根据注意力分布计算权重向量，以调整 x 序列的持续时间。
    # 如果使用 Soft DTW，则计算一个序列的持续时间，
    # 否则计算一个标量损失，用于评估所生成语音的质量。
    if self.use_sdp:
      l_length = self.duration_predictor(x, x_mask, w, g=g)
      l_length = l_length / torch.sum(x_mask)
    else:
      logw_ = torch.log(w + 1e-6) * x_mask
      logw = self.duration_predictor(x, x_mask, g=g)
      l_length = torch.sum((logw - logw_) ** 2, [1, 2]) / torch.sum(x_mask)  # for averaging

    # expand prior
    # 使用注意力分布对先验分布的均值和方差进行加权，以获得特定时间步的平均均值和方差，从而用于音频生成
    m_p = torch.einsum('bctn, bdn -> bdt', attn, m_p)
    logs_p = torch.einsum('bctn, bdn -> bdt', attn, logs_p)

    # 随机选择一个小的时间片段 z_slice 和一个对应的 ID 序列 ids_slice，
    # 并将它们传递给波形解码器 waveform_decoder，用于生成音频
    z_slice, ids_slice = commons.rand_slice_segments_for_cat(
      z_dec_, torch.cat([y_lengths, y_lengths], dim=0), self.segment_size
    )

    o_ = self.waveform_decoder.hier_forward(z_slice, g=torch.cat([g, g], dim=0))

    o = [torch.chunk(o_hier, 2, dim=0)[0] for o_hier in o_]
    # 返回生成的音频 o，持续时间的损失 l_length，注意力分布 attn，
    # 随机选择的时间片段的 ID 序列 ids_slice，以及用于将来计算注意力的 x_mask 和 `y
    o_pad = F.pad(
      o_[-1],
      (768, 768 + (-o_[-1].shape[-1]) % 256 + 256 * (o_[-1].shape[-1] % 256 == 0)),
      mode='constant'
    ).squeeze(1)

    yin_hat = self.pitch.yingram(o_pad)
    yin_hat_crop = self.crop_scope([yin_hat])[0]
    yin_hat_shifted = self.crop_scope_tensor(
      torch.chunk(yin_hat, 2, dim=0)[0], scope_shift
    )

    return o, l_length, attn, ids_slice, x_mask, spec_mask, o_, \
      (z, z_p, m_p, logs_p, m_q, logs_q), \
      z_dec_, \
      (z_spec, m_spec, logs_spec, spec_mask, z_yin, m_yin, logs_yin, yin_mask), \
      (yin_gt_crop, yin_gt_shifted_crop, yin_dec_crop, yin_hat_crop, scope_shift, yin_hat_shifted)

  def infer(
      self,
      x,
      t,
      x_lengths,
      sid=None,
      noise_scale=1,
      length_scale=1,
      noise_scale_w=1.,
      max_len=None,
      scope_shift=0  # TODO: need to fix vector scope shift needed
  ):
    x, m_p, logs_p, x_mask = self.text_encoder(x, x_lengths)

    if self.n_speakers > 0:
      g = self.emb_g(sid).unsqueeze(-1)  # [b, h, 1]
    else:
      g = None

    if self.use_sdp:
      logw = self.duration_predictor(x, x_mask, g=g, reverse=True, noise_scale=noise_scale_w)
    else:
      logw = self.duration_predictor(x, x_mask, g=g)

    w = torch.exp(logw) * x_mask * length_scale
    w_ceil = torch.ceil(w)
    y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
    y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, None), 1).to(x_mask.dtype)
    attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
    attn = commons.generate_path(w_ceil, attn_mask)

    # [b, t', t], [b, t, d] -> [b, d, t']
    m_p = torch.einsum('bctn, bdn -> bdt', attn, m_p)
    # [b, t', t], [b, t, d] -> [b, d, t']
    logs_p = torch.einsum('bctn, bdn -> bdt', attn, logs_p)

    z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
    z = self.flow(z_p, y_mask, g=g, reverse=True)
    z_spec, z_yin = torch.split(
      z,
      self.inter_channels - self.yin_channels,
      dim=1
    )
    z_yin_crop = self.crop_scope([z_yin], scope_shift)[0]
    z_crop = torch.cat([z_spec, z_yin_crop], dim=1)
    o = self.waveform_decoder((z_crop * y_mask)[:, :, :max_len], g=g)

    return o, attn, y_mask, (z_crop, z, z_p, m_p, logs_p)

  def infer_pre_decoder(
      self,
      x,
      t,
      x_lengths,
      sid=None,
      noise_scale=1.,
      length_scale=1.,
      noise_scale_w=1.,
      max_len=None,
      scope_shift=0
  ):
    x, m_p, logs_p, x_mask = self.text_encoder(x, t, x_lengths)

    if self.n_speakers > 0:
      g = self.emb_g(sid).unsqueeze(-1)  # [b, h, 1]
    else:
      g = None

    if self.use_sdp:
      logw = self.duration_predictor(
        x,
        x_mask,
        g=g,
        reverse=True,
        noise_scale=noise_scale_w
      )
    else:
      logw = self.duration_predictor(x, x_mask, g=g)

    w = torch.exp(logw) * x_mask * length_scale
    w_ceil = torch.ceil(w)
    y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
    y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, None), 1).to(x_mask.dtype)
    attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
    attn = commons.generate_path(w_ceil, attn_mask)

    m_p = torch.einsum('bctn, bdn -> bdt', attn, m_p)
    logs_p = torch.einsum('bctn, bdn -> bdt', attn, logs_p)

    z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
    z = self.flow(z_p, y_mask, g=g, reverse=True)
    z_spec, z_yin = torch.split(
      z,
      self.inter_channels - self.yin_channels,
      dim=1
    )
    z_yin_crop = self.crop_scope([z_yin], scope_shift)[0]
    z_crop = torch.cat([z_spec, z_yin_crop], dim=1)
    decoder_inputs = z_crop * y_mask

    return decoder_inputs, attn, y_mask, (z_crop, z, z_p, m_p, logs_p)

  def infer_pre_lr(
      self,
      x,
      t,
      x_lengths,
      sid=None,
      length_scale=1,
      noise_scale_w=1.,
  ):
    x, m_p, logs_p, x_mask = self.text_encoder(x, t, x_lengths)

    if self.n_speakers > 0:
      g = self.emb_g(sid).unsqueeze(-1)  # [b, h, 1]
    else:
      g = None

    if self.use_sdp:
      logw = self.duration_predictor(
        x,
        x_mask,
        g=g,
        reverse=True,
        noise_scale=noise_scale_w
      )
    else:
      logw = self.duration_predictor(x, x_mask, g=g)

    w = torch.exp(logw) * x_mask * length_scale
    w_ceil = torch.ceil(w)

    return w_ceil, x, m_p, logs_p, x_mask, g

  def infer_lr(self, w_ceil, x, m_p, logs_p, x_mask):
    y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
    y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, None), 1).to(x_mask.dtype)
    attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
    attn = commons.generate_path(w_ceil, attn_mask)

    m_p = torch.einsum('bctn, bdn -> bdt', attn, m_p)
    logs_p = torch.einsum('bctn, bdn -> bdt', attn, logs_p)

    return m_p, logs_p, y_mask

  def infer_post_lr_pre_decoder(
      self,
      m_p,
      logs_p,
      g,
      y_mask,
      noise_scale=1,
      scope_shift=0
  ):
    z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
    z = self.flow(z_p, y_mask, g=g, reverse=True)
    z_spec, z_yin = torch.split(
      z,
      self.inter_channels - self.yin_channels,
      dim=1
    )

    z_yin_crop = self.crop_scope([z_yin], scope_shift)[0]
    z_crop = torch.cat([z_spec, z_yin_crop], dim=1)
    decoder_inputs = z_crop * y_mask

    return decoder_inputs, y_mask, (z_crop, z, z_p, m_p, logs_p)

  def infer_decode_chunk(self, decoder_inputs, sid=None):
    if self.n_speakers > 0:
      g = self.emb_g(sid).unsqueeze(-1)  # [b, h, 1]
    else:
      g = None
    return self.waveform_decoder(decoder_inputs, g=g)

  def voice_conversion(self, y, y_lengths, sid_src, sid_tgt):
    assert self.n_speakers > 0, "n_speakers have to be larger than 0."

    g_src = self.emb_g(sid_src).unsqueeze(-1)
    g_tgt = self.emb_g(sid_tgt).unsqueeze(-1)
    z, m_q, logs_q, y_mask = self.posterior_encoder(y, y_lengths, g=g_src)
    z_p = self.flow(z, y_mask, g=g_src)
    z_hat = self.flow(z_p, y_mask, g=g_tgt, reverse=True)
    o_hat = self.waveform_decoder(z_hat * y_mask, g=g_tgt)

    return o_hat, y_mask, (z, z_p, z_hat)
