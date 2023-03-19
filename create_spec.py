import os

import torch
import torch.utils.data
from tqdm import tqdm

from mel_processing import spectrogram_torch
from utils import load_wav_to_torch, load_filepaths_and_text


def create_spec(wav_paths_sid_text, hparams):
  wav_paths_sid_text = load_filepaths_and_text(wav_paths_sid_text)

  for wav_path, _, _, _ in tqdm(wav_paths_sid_text):
    wav_path = os.path.join(hparams.data_path, wav_path)

    if not os.path.exists(wav_path):
      print(wav_path, "not exist!")
      continue
    try:
      audio, sampling_rate = load_wav_to_torch(wav_path)
    except:
      print(wav_path, "load error!")
      continue

    if sampling_rate != hparams.sampling_rate:
      raise ValueError("{} SR doesn't match target {} SR".format(
        sampling_rate, hparams.sampling_rate)
      )

    audio_norm = audio.unsqueeze(0)
    spec_path = wav_path.replace(".wav", ".spec.pt")

    if not os.path.exists(spec_path):
      spec = spectrogram_torch(
        audio_norm,
        hparams.filter_length,
        hparams.hop_length,
        hparams.win_length,
        center=False
      )
      spec = torch.squeeze(spec, 0)
      torch.save(spec, spec_path)
