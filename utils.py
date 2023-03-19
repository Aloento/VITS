import glob
import logging
import os
import subprocess
import sys

import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.io.wavfile import read

MATPLOTLIB_FLAG = False

logging.basicConfig(
  stream=sys.stdout,
  level=logging.INFO,
  format='[%(levelname)s|%(filename)s:%(lineno)s][%(asctime)s] >>> %(message)s'
)
logger = logging


def load_checkpoint(checkpoint_path, rank=0, model_g=None, model_d=None, optim_g=None, optim_d=None):
  is_train = os.path.isdir(checkpoint_path)

  if is_train:
    train = latest_checkpoint_path(checkpoint_path, "*_Train_*.pth")
    val = latest_checkpoint_path(checkpoint_path, "*_Eval_*.pth")

    assert train is not None
    assert val is not None

    train_dict = torch.load(train, map_location='cpu')
    iteration = train_dict['iteration']
  else:
    assert os.path.isfile(checkpoint_path)
    val = checkpoint_path

  val_dict = torch.load(val, map_location='cpu')
  config = val_dict['config']

  assert model_g is not None
  model_g = load_model(
    model_g,
    val_dict['model_g']
  )

  if is_train:
    if optim_g is not None:
      optim_g.load_state_dict(train_dict['optimizer_g'])

    if model_d is not None:
      model_d = load_model(
        model_d,
        train_dict['model_d']
      )

    if optim_d is not None:
      optim_d.load_state_dict(train_dict['optimizer_d'])

  if rank == 0:
    logger.info(
      "Loaded checkpoint '{}' (iteration {})".format(
        checkpoint_path,
        iteration if is_train else "Eval"
      )
    )

  return model_g, model_d, optim_g, optim_d, iteration, config


def load_model(model, model_state_dict):
  if hasattr(model, 'module'):
    state_dict = model.module.state_dict()
  else:
    state_dict = model.state_dict()

  for k, v in model_state_dict.items():
    if k in state_dict and state_dict[k].size() == v.size():
      state_dict[k] = v

  if hasattr(model, 'module'):
    model.module.load_state_dict(state_dict)
  else:
    model.load_state_dict(state_dict)

  return model


def save_checkpoint(net_g, optim_g, net_d, optim_d, hps, epoch, global_step):
  def get_state_dict(model):
    if hasattr(model, 'module'):
      state_dict = model.module.state_dict()
    else:
      state_dict = model.state_dict()
    return state_dict

  torch.save(
    {
      'model_d': get_state_dict(net_d),
      'optimizer_g': optim_g.state_dict(),
      'optimizer_d': optim_d.state_dict(),
      'iteration': epoch,
    }, os.path.join(
      hps.model_dir, "{}_Train_{}.pth".format(hps.model_name, global_step)
    )
  )

  torch.save(
    {
      'model_g': get_state_dict(net_g),
      'config': str(hps),
    }, os.path.join(
      hps.model_dir, "{}_Eval_{}.pth".format(hps.model_name, global_step)
    )
  )


def summarize(writer, global_step, scalars={}, histograms={}, images={}, audios={}, audio_sampling_rate=22050):
  for k, v in scalars.items():
    writer.add_scalar(k, v, global_step)

  for k, v in histograms.items():
    writer.add_histogram(k, v, global_step)

  for k, v in images.items():
    writer.add_image(k, v, global_step, dataformats='HWC')

  for k, v in audios.items():
    writer.add_audio(k, v, global_step, audio_sampling_rate)


def latest_checkpoint_path(dir_path, regex="*.pth"):
  f_list = glob.glob(os.path.join(dir_path, regex))
  f_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
  x = f_list[-1]
  print(x)
  return x


def plot_spectrogram_to_numpy(spectrogram):
  global MATPLOTLIB_FLAG

  if not MATPLOTLIB_FLAG:
    import matplotlib
    matplotlib.use("Agg")
    MATPLOTLIB_FLAG = True
    mpl_logger = logging.getLogger('matplotlib')
    mpl_logger.setLevel(logging.WARNING)

  import matplotlib.pylab as plt
  import numpy as np

  fig, ax = plt.subplots(figsize=(10, 2))
  im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation='none')

  plt.colorbar(im, ax=ax)
  plt.xlabel("Frames")
  plt.ylabel("Channels")
  plt.tight_layout()

  fig.canvas.draw()
  data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
  data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
  plt.close()

  return data


def load_wav_to_torch(full_path):
  """
  使用read方法从音频文件中读取采样率和音频数据，并将数据类型转换为float32类型。
  然后，使用torch.FloatTensor方法将数据转换为PyTorch的FloatTensor类型，并返回数据和采样率。
  """
  sampling_rate, wav = read(full_path)

  if len(wav.shape) == 2:
    wav = wav[:, 0]

  if wav.dtype == np.int16:
    wav = wav / 32768.0
  elif wav.dtype == np.int32:
    wav = wav / 2147483648.0
  elif wav.dtype == np.uint8:
    wav = (wav - 128) / 128.0

  wav = wav.astype(np.float32)
  return torch.FloatTensor(wav), sampling_rate


def load_filepaths_and_text(filename, split="|"):
  with open(filename, encoding='utf-8') as f:
    filepaths_and_text = [line.strip().split(split) for line in f]
  return filepaths_and_text


def get_hparams(args, init=True):
  config = OmegaConf.load(args.config)
  hparams = HParams(**config)
  model_dir = os.path.join(hparams.train.log_path, args.model)

  if not os.path.exists(model_dir):
    os.makedirs(model_dir)
  hparams.model_name = args.model
  hparams.model_dir = model_dir
  config_save_path = os.path.join(model_dir, "config.yaml")

  if init:
    OmegaConf.save(config, config_save_path)

  return hparams


def get_hparams_from_file(config_path):
  config = OmegaConf.load(config_path)
  hparams = HParams(**config)
  return hparams


def check_git_hash(model_dir):
  """
  :param model_dir: 模型文件存储的路径
  :return:
  """

  # 获取当前脚本文件的绝对路径
  # 获取当前脚本文件所在目录的路径
  source_dir = os.path.dirname(os.path.realpath(__file__))
  # 判断源代码所在目录是否为 git 仓库
  if not os.path.exists(os.path.join(source_dir, ".git")):
    logger.warn("{} is not a git repository, therefore hash value comparison will be ignored.".format(
      source_dir
    ))
    return

  # 获取当前代码仓库的 HEAD 值
  cur_hash = subprocess.getoutput("git rev-parse HEAD")

  path = os.path.join(model_dir, "githash")
  #  githash 文件
  if os.path.exists(path):
    saved_hash = open(path).read()
    # 将文件中保存的版本号与当前版本号进行比较
    if saved_hash != cur_hash:
      logger.warn("git hash values are different. {}(saved) != {}(current)".format(
        saved_hash[:8], cur_hash[:8]))
  else:
    # 新建一个文件并保存当前版本号
    open(path, "w").write(cur_hash)


def get_logger(model_dir, filename="train.log"):
  global logger

  logger = logging.getLogger(os.path.basename(model_dir))
  logger.setLevel(logging.DEBUG)

  formatter = logging.Formatter("%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")

  if not os.path.exists(model_dir):
    os.makedirs(model_dir)

  h = logging.FileHandler(os.path.join(model_dir, filename))
  h.setLevel(logging.DEBUG)
  h.setFormatter(formatter)
  logger.addHandler(h)

  return logger


class HParams():
  def __init__(self, **kwargs):
    for k, v in kwargs.items():
      if type(v) == dict:
        v = HParams(**v)
      self[k] = v

  def keys(self):
    return self.__dict__.keys()

  def items(self):
    return self.__dict__.items()

  def values(self):
    return self.__dict__.values()

  def __len__(self):
    return len(self.__dict__)

  def __getitem__(self, key):
    return getattr(self, key)

  def __setitem__(self, key, value):
    return setattr(self, key, value)

  def __contains__(self, key):
    return key in self.__dict__

  def __repr__(self):
    return self.__dict__.__repr__()
