import torch


class TextAudioSpeakerCollate:
  """
  Zero-pads model inputs and targets
  准备模型的输入数据和标签数据，确保它们的长度一致
  """

  def __init__(self, return_ids=False):
    """
    :param return_ids: 是否返回原始数据的索引
    """
    self.return_ids = return_ids

  def __call__(self, batch):
    """
    Collates training batch from normalized text and audio

    :param batch: [text_normalized, spec_normalized, wav_normalized]
    """

    # Right zero-pad all one-hot text sequences to max input length
    # 对输入的文本序列进行了右侧的零填充以便于统一长度
    # 每个元素在原始输入张量中的位置索引 被放弃 _
    _, ids_sorted_decreasing = torch.sort(
      # 获取batch中每个样本的第二个元素（即spec_normalized）的第二个维度大小（即时间步数）
      torch.LongTensor([x[1].size(1) for x in batch]),
      dim=0, descending=True
    )

    # 批次中所有文本序列的最大长度
    max_text_len = max([len(x[0]) for x in batch])
    # 所有频谱图最大长度
    max_spec_len = max([x[1].size(1) for x in batch])
    # pitch最大长度
    max_ying_len = max([x[2].size(1) for x in batch])
    # 波形数据最大长度
    max_wav_len = max([x[3].size(1) for x in batch])

    # 它们都具有批次大小的长度，用于存储每个样本的实际长度
    text_lengths = torch.LongTensor(len(batch))
    spec_lengths = torch.LongTensor(len(batch))
    ying_lengths = torch.LongTensor(len(batch))
    wav_lengths = torch.LongTensor(len(batch))
    sid = torch.LongTensor(len(batch))

    # 存储文本序列的零填充后的结果
    text_padded = torch.LongTensor(len(batch), max_text_len)
    tone_padded = torch.LongTensor(len(batch), max_text_len)
    # 存储 Mel-Spectrogram 的零填充后的结果
    # batch[0][1].size(0) 是一个常量，表示 Mel-Spectrogram 的数量
    spec_padded = torch.FloatTensor(len(batch), batch[0][1].size(0), max_spec_len)
    ying_padded = torch.FloatTensor(len(batch), batch[0][2].size(0), max_ying_len)
    # 音频序列的零填充后的结果
    wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)

    # 将这些张量的所有元素设置为零，以确保它们的初始值都是零
    text_padded.zero_()
    tone_padded.zero_()
    spec_padded.zero_()
    ying_padded.zero_()
    wav_padded.zero_()

    # ids_sorted_decreasing (batch 中的索引) 中的排序索引将当前样本放置到正确的位置，并进行填充
    for i in range(len(ids_sorted_decreasing)):
      # 将当前样本的文本、音频频谱、语音波形进行填充，并记录对应的长度
      # 在每个填充后的张量中，实际上只有当前样本所对应的那一维是被填充过的，其它维度都是零。
      # 这样做是为了保证每个张量在维度上都是一致的，方便后续的模型输入
      row = batch[ids_sorted_decreasing[i]]

      text = row[0]
      text_padded[i, :text.size(0)] = text
      text_lengths[i] = text.size(0)

      spec = row[1]
      spec_padded[i, :, :spec.size(1)] = spec
      spec_lengths[i] = spec.size(1)

      ying = row[2]
      ying_padded[i, :, :ying.size(1)] = ying
      ying_lengths[i] = ying.size(1)

      wav = row[3]
      wav_padded[i, :, :wav.size(1)] = wav
      wav_lengths[i] = wav.size(1)

      tone = row[5]
      tone_padded[i, :text.size(0)] = tone

      sid[i] = row[4]

    if self.return_ids:
      return text_padded, text_lengths, spec_padded, spec_lengths, ying_padded, ying_lengths, wav_padded, wav_lengths, sid, tone_padded, ids_sorted_decreasing

    return text_padded, text_lengths, spec_padded, spec_lengths, ying_padded, ying_lengths, wav_padded, wav_lengths, sid, tone_padded
