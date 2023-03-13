import os
import random

import torch
import torch.utils.data

import commons
from mel_processing import spectrogram_torch
from text import text_to_sequence, cleaned_text_to_sequence
from utils import load_wav_to_torch, load_filepaths_and_text


class TextAudioLoader(torch.utils.data.Dataset):
  """
  1) loads audio, text pairs 加载音频和文本对
  2) normalizes text and converts them to sequences of integers 对文本进行规范化并将其转换为整数序列
  3) computes spectrogram's from audio files. 从音频文件计算出其对应的声谱图
  """

  def __init__(self, audiopaths_and_text, hparams):
    self.audiopaths_and_text = load_filepaths_and_text(audiopaths_and_text)
    self.text_cleaners = hparams.text_cleaners
    self.max_wav_value = hparams.max_wav_value
    self.sampling_rate = hparams.sampling_rate
    self.filter_length = hparams.filter_length
    self.hop_length = hparams.hop_length
    self.win_length = hparams.win_length
    self.sampling_rate = hparams.sampling_rate

    self.cleaned_text = getattr(hparams, "cleaned_text", False)

    self.add_blank = hparams.add_blank
    self.min_text_len = getattr(hparams, "min_text_len", 1)
    self.max_text_len = getattr(hparams, "max_text_len", 190)

    random.seed(1234)
    # 随机打乱
    random.shuffle(self.audiopaths_and_text)
    # 将不符合文本长度要求的音频和文本对剔除
    self._filter()

  def _filter(self):
    """
    Filter text & store spec lengths
    筛选输入数据的音频和文本信息，并为后续分桶操作存储音频的长度
    """
    # Store spectrogram lengths for Bucketing
    # wav_length ~= file_size / (wav_channels * Bytes per dim) = file_size / (1 * 2)
    # spec_length = wav_length // hop_length

    audiopaths_and_text_new = []
    lengths = []
    # 遍历输入数据中的每个音频文件和对应的文本信息
    for audiopath, text in self.audiopaths_and_text:
      # 判断文本长度是否在给定范围内
      if self.min_text_len <= len(text) <= self.max_text_len:
        # 将这个音频和文本对加入一个新的列表，并计算该音频对应的长度（单位是以采样点数为基础的帧数）
        audiopaths_and_text_new.append([audiopath, text])
        lengths.append(os.path.getsize(audiopath) // (2 * self.hop_length))

    self.audiopaths_and_text = audiopaths_and_text_new
    self.lengths = lengths

  def get_audio_text_pair(self, audiopath_and_text):
    # separate filename and text 分离出音频文件路径和文本
    audiopath, text = audiopath_and_text[0], audiopath_and_text[1]
    text = self.get_text(text)
    spec, wav = self.get_audio(audiopath)
    return text, spec, wav

  def get_audio(self, filename):
    # 加载音频文件，并获取音频的采样率
    audio, sampling_rate = load_wav_to_torch(filename)

    # 检查采样率是否与目标采样率相同
    if sampling_rate != self.sampling_rate:
      raise ValueError("{} SR doesn't match target {} SR".format(
        sampling_rate, self.sampling_rate))

    # 将音频归一化，将音频值除以最大音频值，以使所有音频值在[-1, 1]之间
    audio_norm = audio / self.max_wav_value
    audio_norm = audio_norm.unsqueeze(0)

    # 频谱文件
    spec_filename = filename.replace(".wav", ".spec.pt")
    if os.path.exists(spec_filename):
      # 从该文件中加载频谱数据
      spec = torch.load(spec_filename)
    else:
      # 计算音频的频谱
      spec = spectrogram_torch(audio_norm, self.filter_length,
                               self.hop_length, self.win_length,
                               center=False)
      # 将添加的维度压缩掉
      spec = torch.squeeze(spec, 0)
      torch.save(spec, spec_filename)
    # 频谱数据和归一化后的音频数据
    return spec, audio_norm

  def get_text(self, text):
    # 将文本转换为数字序列，每个字符都映射到一个唯一的整数
    if self.cleaned_text:
      text_norm = cleaned_text_to_sequence(text)
    else:
      text_norm = text_to_sequence(text, self.text_cleaners)

    # 空白符通常用于标记单词之间的空格
    if self.add_blank:
      text_norm = commons.intersperse(text_norm, 0)

    text_norm = torch.LongTensor(text_norm)
    return text_norm

  def __getitem__(self, index):
    return self.get_audio_text_pair(self.audiopaths_and_text[index])

  def __len__(self):
    return len(self.audiopaths_and_text)


class TextAudioCollate:
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
      # 沿第0个维度排序（即按行进行排序），降序排列
      dim=0, descending=True)

    # 批次中所有文本序列的最大长度
    max_text_len = max([len(x[0]) for x in batch])
    # 所有频谱图最大长度
    max_spec_len = max([x[1].size(1) for x in batch])
    # 波形数据最大长度
    max_wav_len = max([x[2].size(1) for x in batch])

    # 它们都具有批次大小的长度，用于存储每个样本的实际长度
    text_lengths = torch.LongTensor(len(batch))
    spec_lengths = torch.LongTensor(len(batch))
    wav_lengths = torch.LongTensor(len(batch))

    # 存储文本序列的零填充后的结果
    text_padded = torch.LongTensor(len(batch), max_text_len)
    # 存储 Mel-Spectrogram 的零填充后的结果
    # batch[0][1].size(0) 是一个常量，表示 Mel-Spectrogram 的数量
    spec_padded = torch.FloatTensor(len(batch), batch[0][1].size(0), max_spec_len)
    # 音频序列的零填充后的结果
    wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)
    # 将这些张量的所有元素设置为零，以确保它们的初始值都是零
    text_padded.zero_()
    spec_padded.zero_()
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

      wav = row[2]
      wav_padded[i, :, :wav.size(1)] = wav
      wav_lengths[i] = wav.size(1)

    if self.return_ids:
      return text_padded, text_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths, ids_sorted_decreasing

    return text_padded, text_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths


class TextAudioSpeakerLoader(torch.utils.data.Dataset):
  """
  Multi speaker version
  1) loads audio, speaker_id, text pairs
  2) normalizes text and converts them to sequences of integers
  3) computes spectrograms from audio files.
  """

  def __init__(self, audiopaths_sid_text, hparams):
    self.audiopaths_sid_text = load_filepaths_and_text(audiopaths_sid_text)
    self.text_cleaners = hparams.text_cleaners
    self.max_wav_value = hparams.max_wav_value
    self.sampling_rate = hparams.sampling_rate
    self.filter_length = hparams.filter_length
    self.hop_length = hparams.hop_length
    self.win_length = hparams.win_length
    self.sampling_rate = hparams.sampling_rate

    self.cleaned_text = getattr(hparams, "cleaned_text", False)

    self.add_blank = hparams.add_blank
    self.min_text_len = getattr(hparams, "min_text_len", 1)
    self.max_text_len = getattr(hparams, "max_text_len", 190)

    random.seed(1234)
    random.shuffle(self.audiopaths_sid_text)
    self._filter()

  def _filter(self):
    """
    Filter text & store spec lengths
    """
    # Store spectrogram lengths for Bucketing
    # wav_length ~= file_size / (wav_channels * Bytes per dim) = file_size / (1 * 2)
    # spec_length = wav_length // hop_length

    audiopaths_sid_text_new = []
    lengths = []
    for audiopath, sid, text in self.audiopaths_sid_text:
      if self.min_text_len <= len(text) and len(text) <= self.max_text_len:
        audiopaths_sid_text_new.append([audiopath, sid, text])
        lengths.append(os.path.getsize(audiopath) // (2 * self.hop_length))
    self.audiopaths_sid_text = audiopaths_sid_text_new
    self.lengths = lengths

  def get_audio_text_speaker_pair(self, audiopath_sid_text):
    # separate filename, speaker_id and text
    audiopath, sid, text = audiopath_sid_text[0], audiopath_sid_text[1], audiopath_sid_text[2]
    text = self.get_text(text)
    spec, wav = self.get_audio(audiopath)
    sid = self.get_sid(sid)
    return (text, spec, wav, sid)

  def get_audio(self, filename):
    audio, sampling_rate = load_wav_to_torch(filename)
    if sampling_rate != self.sampling_rate:
      raise ValueError("{} SR doesn't match target {} SR".format(
        sampling_rate, self.sampling_rate))
    audio_norm = audio / self.max_wav_value
    audio_norm = audio_norm.unsqueeze(0)
    spec_filename = filename.replace(".wav", ".spec.pt")
    if os.path.exists(spec_filename):
      spec = torch.load(spec_filename)
    else:
      spec = spectrogram_torch(audio_norm, self.filter_length,
                               self.hop_length, self.win_length,
                               center=False)
      spec = torch.squeeze(spec, 0)
      torch.save(spec, spec_filename)
    return spec, audio_norm

  def get_text(self, text):
    if self.cleaned_text:
      text_norm = cleaned_text_to_sequence(text)
    else:
      text_norm = text_to_sequence(text, self.text_cleaners)
    if self.add_blank:
      text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

  def get_sid(self, sid):
    sid = torch.LongTensor([int(sid)])
    return sid

  def __getitem__(self, index):
    return self.get_audio_text_speaker_pair(self.audiopaths_sid_text[index])

  def __len__(self):
    return len(self.audiopaths_sid_text)


class TextAudioSpeakerCollate:
  """ Zero-pads model inputs and targets
  """

  def __init__(self, return_ids=False):
    self.return_ids = return_ids

  def __call__(self, batch):
    """Collate's training batch from normalized text, audio and speaker identities
    PARAMS
    ------
    batch: [text_normalized, spec_normalized, wav_normalized, sid]
    """
    # Right zero-pad all one-hot text sequences to max input length
    _, ids_sorted_decreasing = torch.sort(
      torch.LongTensor([x[1].size(1) for x in batch]),
      dim=0, descending=True)

    max_text_len = max([len(x[0]) for x in batch])
    max_spec_len = max([x[1].size(1) for x in batch])
    max_wav_len = max([x[2].size(1) for x in batch])

    text_lengths = torch.LongTensor(len(batch))
    spec_lengths = torch.LongTensor(len(batch))
    wav_lengths = torch.LongTensor(len(batch))
    sid = torch.LongTensor(len(batch))

    text_padded = torch.LongTensor(len(batch), max_text_len)
    spec_padded = torch.FloatTensor(len(batch), batch[0][1].size(0), max_spec_len)
    wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)
    text_padded.zero_()
    spec_padded.zero_()
    wav_padded.zero_()
    for i in range(len(ids_sorted_decreasing)):
      row = batch[ids_sorted_decreasing[i]]

      text = row[0]
      text_padded[i, :text.size(0)] = text
      text_lengths[i] = text.size(0)

      spec = row[1]
      spec_padded[i, :, :spec.size(1)] = spec
      spec_lengths[i] = spec.size(1)

      wav = row[2]
      wav_padded[i, :, :wav.size(1)] = wav
      wav_lengths[i] = wav.size(1)

      sid[i] = row[3]

    if self.return_ids:
      return text_padded, text_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths, sid, ids_sorted_decreasing
    return text_padded, text_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths, sid


class DistributedBucketSampler(torch.utils.data.distributed.DistributedSampler):
  """
  用于在多个GPU之间分配batch的数据加载器，会尽量使得每个batch中的样本长度相似，
  通过指定length boundaries来定义长度组，任何一个batch都是包含在两个连续的length boundaries之间的。

  继承自PyTorch的DistributedSampler，它维护了数据集的一些基本信息，例如长度和样本数量，
  构造函数的参数包括数据集、batch size、length boundaries、replicas数量和rank，
  其中replicas数量和rank用于指定当前GPU的数量和ID。

  尽量使得每个batch中的样本长度相似是为了避免在训练过程中产生过多的padding，从而降低训练效率和模型性能。
  在深度学习中，通常会将一个batch中的多个样本同时输入到模型中进行训练，而每个样本的长度不同，
  如果直接将它们拼接在一起形成一个batch进行训练，则需要在长度较短的样本上进行padding，
  使得它们的长度与长度最长的样本相同。这样就会造成一些浪费，
  因为在进行padding时，填充的信息对模型并没有贡献，同时也会增加训练的计算量。
  如果能够使得每个batch中的样本长度相似，则可以减少padding的数量，从而提高训练效率和模型性能。

  TODO：在这里需要修改为自动确定 boundaries

  在一个batch中保持相似的输入长度。
  长度组由边界指定。
  例如）边界 = [b1，b2，b3] -> 任何一个批次都包括{x | b1 < length(x) <= b2}或{x | b2 < length(x) <= b3}。

  它会删除不在边界范围内的样本。
  例如）边界 = [b1，b2，b3] -> 所有满足length(x) <= b1或length(x) > b3的x都会被删除。
  """

  def __init__(self, dataset, batch_size, boundaries, num_replicas=None, rank=None, shuffle=True):
    """
    :param dataset: 数据集 (TextAudioLoader)
    :param batch_size: 每个批次中样本的数量
    :param boundaries: 长度的边界，指定每个批次的长度范围
    :param num_replicas: 进程数，即将数据集分成多少份
    :param rank: 进程的标识符，从0开始
    :param shuffle: 是否在每个epoch开始时打乱数据集
    """

    # 初始化基类
    super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)

    self.lengths = dataset.lengths
    self.batch_size = batch_size
    self.boundaries = boundaries

    # 根据边界将数据集分成不同的“桶”，并计算每个“桶”包含的样本数
    self.buckets, self.num_samples_per_bucket = self._create_buckets()
    # 计算整个数据集包含的样本数
    self.total_size = sum(self.num_samples_per_bucket)
    # 每个进程应该处理的样本数
    self.num_samples = self.total_size // self.num_replicas

  def _create_buckets(self):
    """
    将数据集分成不同的桶，每个桶内包含长度在一定范围内的样本
    """

    # 如果边界为[32, 64, 128, 256]，则将创建三个桶：
    # 第一个桶用于包含长度在(32, 64]之间的样本，
    # 第二个桶用于包含长度在(64, 128]之间的样本，
    # 第三个桶用于包含长度在(128, 256]之间的样本。
    buckets = [[] for _ in range(len(self.boundaries) - 1)]

    for i in range(len(self.lengths)):
      length = self.lengths[i]
      # 样本应该被分配到的桶的索引号
      idx_bucket = self._bisect(length)
      # 样本的长度小于最小的范围或大于最大的范围，则返回-1，表示不分配到任何桶中
      if idx_bucket != -1:
        buckets[idx_bucket].append(i)

    # 最后一个bucket开始遍历
    for i in range(len(buckets) - 1, 0, -1):
      if len(buckets[i]) == 0:
        # 移除空桶
        buckets.pop(i)
        # 以便后面的代码可以正确地计算样本的长度范围
        self.boundaries.pop(i + 1)

    num_samples_per_bucket = []
    for i in range(len(buckets)):
      len_bucket = len(buckets[i])
      # 总的 batch size：每个 replica 上的 batch size 乘以 replica 的数量
      total_batch_size = self.num_replicas * self.batch_size
      # 当前 bucket 内需要额外添加多少个样本才能达到一个完整的 batch
      rem = (total_batch_size - (len_bucket % total_batch_size)) % total_batch_size
      # 当前 bucket 内需要取 len_bucket + rem 个样本
      num_samples_per_bucket.append(len_bucket + rem)

    # 桶的列表和每个桶中样本的数量列表
    return buckets, num_samples_per_bucket

  def __iter__(self):
    # deterministically shuffle based on epoch
    # 随机数生成器
    g = torch.Generator()
    # 根据固定的种子确定一个确定性随机序列
    g.manual_seed(self.epoch)

    indices = []
    if self.shuffle:
      for bucket in self.buckets:
        # 打乱数据顺序
        indices.append(torch.randperm(len(bucket), generator=g).tolist())
    else:
      for bucket in self.buckets:
        # 每个样本在bucket中的索引
        indices.append(list(range(len(bucket))))

    batches = []
    for i in range(len(self.buckets)):
      bucket = self.buckets[i]
      # 当前bucket中的样本数量
      len_bucket = len(bucket)
      ids_bucket = indices[i]
      # 每个bucket需要的样本数量
      num_samples_bucket = self.num_samples_per_bucket[i]

      # add extra samples to make it evenly divisible
      # 需要添加的额外样本数量rem
      rem = num_samples_bucket - len_bucket
      # 使得每个bucket的样本数都可以被batch size整除
      ids_bucket = ids_bucket + ids_bucket * (rem // len_bucket) + ids_bucket[:(rem % len_bucket)]

      # subsample
      # 均匀采样，得到当前进程需要处理的样本id
      ids_bucket = ids_bucket[self.rank::self.num_replicas]

      # batching
      # 将ids_bucket分成若干个batch，每个batch包含batch_size个样本
      for j in range(len(ids_bucket) // self.batch_size):
        batch = [bucket[idx] for idx in ids_bucket[j * self.batch_size:(j + 1) * self.batch_size]]
        batches.append(batch)

    if self.shuffle:
      # 根据随机数生成的batch_ids列表打乱batches的顺序
      batch_ids = torch.randperm(len(batches), generator=g).tolist()
      batches = [batches[i] for i in batch_ids]
    self.batches = batches

    # 确保生成的所有batch的样本数之和等于num_samples（数据集中的样本总数）
    assert len(self.batches) * self.batch_size == self.num_samples
    # batches的迭代器
    return iter(self.batches)

  def _bisect(self, x, lo=0, hi=None):
    """
    二分查找算法来查找一个数字在已排序的列表中的位置
    :param x: 要查找的数字
    :param lo:
    :param hi: 搜索范围的右端点
    :return:
    """
    if hi is None:
      hi = len(self.boundaries) - 1

    if hi > lo:
      # 中间位置的索引
      mid = (hi + lo) // 2
      if self.boundaries[mid] < x <= self.boundaries[mid + 1]:
        return mid
      elif x <= self.boundaries[mid]:
        # 在搜索范围的左半部分递归
        return self._bisect(x, lo, mid)
      else:
        # 在搜索范围的右半部分递归
        return self._bisect(x, mid + 1, hi)
    else:
      return -1

  def __len__(self):
    # 数据集的长度为多少个batch
    # 如果 num_samples 是 1000，batch_size 是 64，那么将返回 15，表示数据集可以分成 15 个大小为 64 的批次，以处理所有的 1000 个样本
    return self.num_samples // self.batch_size
