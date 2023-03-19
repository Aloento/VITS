import os
import random

import torch
import torch.utils
from torch.utils.data import Dataset

import commons
from Pitch import Pitch
from mel_processing import spectrogram_torch
from text import cleaned_text_to_sequence
from utils import load_filepaths_and_text, load_wav_to_torch


class TextAudioSpeakerLoader(Dataset):
  """
  Multi speaker version
  1) loads audio, speaker_id, text pairs 加载音频，说话人，和文本对
  2) normalizes text and converts them to sequences of integers 对文本进行规范化并将其转换为整数序列
  3) computes spectrogram's from audio files. 从音频文件计算出其对应的声谱图
  """

  def __init__(self, wav_paths_sid_text, hparams, pt_run=False):
    self.wav_paths_sid_text = load_filepaths_and_text(wav_paths_sid_text)
    self.sampling_rate = hparams.sampling_rate
    self.filter_length = hparams.filter_length
    self.hop_length = hparams.hop_length
    self.win_length = hparams.win_length

    self.add_blank = hparams.add_blank
    self.min_text_len = getattr(hparams, "min_text_len", 1)
    self.max_text_len = getattr(hparams, "max_text_len", 190)

    self.speaker_dict = {
      speaker: idx
      for idx, speaker in enumerate(hparams.speakers)
    }
    self.data_path = hparams.data_path

    self.pitch = Pitch(
      sr=hparams.sampling_rate,
      W=hparams.tau_max,
      tau_max=hparams.tau_max,
      midi_start=hparams.midi_start,
      midi_end=hparams.midi_end,
      octave_range=hparams.octave_range
    )

    random.seed(114514)
    # 随机打乱
    random.shuffle(self.wav_paths_sid_text)
    # 将不符合文本长度要求的音频和文本对剔除
    self._filter()

    if pt_run:
      for _wav_paths_sid_text in self.wav_paths_sid_text:
        _ = self.get_audio_text_speaker_pair(
          _wav_paths_sid_text, True
        )

  def _filter(self):
    """
    Filter text & store spec lengths
    筛选输入数据的音频和文本信息，并为后续分桶操作存储音频的长度
    """
    # Store spectrogram lengths for Bucketing
    # wav_length ~= file_size / (wav_channels * Bytes per dim) = file_size / (1 * 2)
    # spec_length = wav_length // hop_length

    wav_paths_sid_text_new = []
    lengths = []

    # 遍历输入数据中的每个音频文件和对应的文本信息
    for wav_path, spk, text, lang in self.wav_paths_sid_text:
      # 判断文本长度是否在给定范围内
      if self.min_text_len <= len(text) <= self.max_text_len:
        wav_path = os.path.join(self.data_path, wav_path)

        if not os.path.exists(wav_path):
          print(wav_path, "not exist!")
          continue
        try:
          _, _ = load_wav_to_torch(wav_path)
        except:
          print(wav_path, "load error!")
          continue

        # 将这个音频和文本对加入一个新的列表，并计算该音频对应的长度（单位是以采样点数为基础的帧数）
        wav_paths_sid_text_new.append([wav_path, spk, text, lang])
        lengths.append(os.path.getsize(wav_path) // (2 * self.hop_length))

    self.wav_paths_sid_text = wav_paths_sid_text_new
    self.lengths = lengths

  def get_audio_text_speaker_pair(self, wav_path_sid_text, pt_run=False):
    # separate filename, speaker_id and text
    wav_path, spk, text, lang = wav_path_sid_text
    text, lang = self.get_text(text, lang)

    spec, ying, wav = self.get_audio(wav_path, pt_run)
    sid = self.get_sid(self.speaker_dict[spk])

    return text, spec, ying, wav, sid, lang

  def get_audio(self, filename, pt_run=False):
    # 加载音频文件，并获取音频的采样率
    audio, sampling_rate = load_wav_to_torch(filename)

    if sampling_rate != self.sampling_rate:
      raise ValueError("{} SR doesn't match target {} SR".format(
        sampling_rate, self.sampling_rate))

    # 将音频归一化
    audio_norm = audio.unsqueeze(0)

    # 频谱文件
    spec_filename = filename.replace(".wav", ".spec.pt")

    if os.path.exists(spec_filename) and not pt_run:
      # 从该文件中加载频谱数据
      spec = torch.load(spec_filename, map_location='cpu')
    else:
      # 计算音频的频谱
      spec = spectrogram_torch(
        audio_norm,
        self.filter_length,
        self.hop_length,
        self.win_length,
        center=False
      )
      # 将添加的维度压缩掉
      spec = torch.squeeze(spec, 0)
      torch.save(spec, spec_filename)

    ying_filename = filename.replace(".wav", ".ying.pt")

    if os.path.exists(ying_filename) and not pt_run:
      ying = torch.load(ying_filename, map_location='cpu')
    else:
      wav = torch.nn.functional.pad(
        audio_norm.unsqueeze(0),
        (
          self.filter_length - self.hop_length,
          self.filter_length - self.hop_length +
          (-audio_norm.shape[1]) % self.hop_length + self.hop_length *
          (audio_norm.shape[1] % self.hop_length == 0)
        ),
        mode='constant').squeeze(0)

      ying = self.pitch.yingram(wav)[0]
      torch.save(ying, ying_filename)

    # 频谱数据和归一化后的音频数据
    return spec, ying, audio_norm

  def get_text(self, text, lang):
    # 将文本转换为数字序列，每个字符都映射到一个唯一的整数
    text_norm = cleaned_text_to_sequence(text)
    lang = [int(i) for i in lang.split(" ")]

    # 空白符通常用于标记单词之间的空格
    if self.add_blank:
      text_norm, lang = commons.intersperse_with_language_id(text_norm, lang, 0)

    text_norm = torch.LongTensor(text_norm)
    lang = torch.LongTensor(lang)

    return text_norm, lang

  def get_sid(self, sid):
    sid = torch.LongTensor([int(sid)])
    return sid

  def __getitem__(self, index):
    return self.get_audio_text_speaker_pair(self.wav_paths_sid_text[index])

  def __len__(self):
    return len(self.wav_paths_sid_text)
