import argparse
import os

import torch
import torch.distributed as dist  # 用于在多个GPU上进行分布式训练的模块
import torch.multiprocessing as mp  # 用于在多个进程中运行代码的模块
from torch.backends import cudnn
from torch.cuda.amp import GradScaler  # 用于混合精度训练的模块
from torch.nn.parallel import DistributedDataParallel as DDP  # 实现了多GPU分布式训练
from torch.utils.data import DataLoader  # 用于数据加载和处理的模块
from torch.utils.tensorboard import SummaryWriter  # 用于可视化训练和评估指标的模块
from tqdm import tqdm

import utils
from Avocodo import AvocodoDiscriminator
from DistributedBucketSampler import DistributedBucketSampler
from SynthesizerTrn import SynthesizerTrn
from TextAudioSpeakerCollate import TextAudioSpeakerCollate
from TextAudioSpeakerLoader import TextAudioSpeakerLoader
from create_spec import create_spec
from text.symbols import symbols
from train_and_evaluate import train_and_evaluate

# PyTorch使用CUDNN加速库
cudnn.benchmark = True

def main(args):
  # 检查CUDA是否可用，如果不可用，则抛出异常
  assert torch.cuda.is_available(), "CPU training is not allowed."

  # 获取当前可用的GPU数量
  num_gpus = torch.cuda.device_count()
  # 获取配置
  hps = utils.get_hparams(args)

  # create spectrogram files
  create_spec(hps.data.training_files, hps.data)
  create_spec(hps.data.validation_files, hps.data)

  os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
  # 用于在多个GPU之间进行通信
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = '8000'

  # 该函数将在每个GPU上运行
  mp.spawn(run, nprocs=num_gpus, args=(num_gpus, hps, args))


def count_parameters(model, scale=1000000):
  return sum(p.numel()
             for p in model.parameters() if p.requires_grad) / scale


def run(rank, num_gpus, hps, args):
  """
  :param rank: 当前进程的 ID
  :param num_gpus: 可用的 GPU 数量
  :param hps: 配置
  """

  if rank == 0:
    # 日志记录器
    logger = utils.get_logger(hps.model_dir)
    # 输出配置
    logger.info('MODEL NAME: {} in {}'.format(args.model, hps.model_dir))
    logger.info(
      'GPU: Use {} gpu(s) with batch size {} (FP16 running: {})'.format(
        num_gpus, hps.train.batch_size, hps.train.fp16_run)
    )

    # 检查代码仓库的版本信息是否与当前运行的代码一致
    utils.check_git_hash(hps.model_dir)
    # 记录训练日志
    writer = SummaryWriter(log_dir=hps.model_dir)

  # 初始化进程组
  # backend: 使用 NVIDIA 提供的通信库 NCCL 进行通信
  # init_method: 使用环境变量的方式初始化进程组
  # world_size: 进程组的总数
  dist.init_process_group(
    backend='gloo',
    init_method='env://',
    world_size=num_gpus,
    rank=rank,
    group_name=args.model
  )

  # 设置随机数种子，使得每次运行代码时生成的随机数序列相同
  torch.manual_seed(hps.train.seed)
  # 设置当前使用的 GPU 设备，将当前进程绑定到 rank 对应的 GPU 设备上
  torch.cuda.set_device(rank)

  # 从音频和文本文件中加载数据
  train_dataset = TextAudioSpeakerLoader(
    hps.data.training_files, hps.data,
    rank == 0 and args.initial_run
  )
  # 对数据进行采样
  train_sampler = DistributedBucketSampler(
    train_dataset,
    hps.train.batch_size,
    [32, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1300, 1400, 1500],
    num_replicas=num_gpus,
    rank=rank,
    shuffle=True
  )

  collate_fn = TextAudioSpeakerCollate()
  # train_dataset：表示要加载的训练数据集。
  # num_workers=8：表示使用 8 个进程来读取数据集。
  # shuffle=False：表示不对数据进行打乱。
  # pin_memory=True：表示将数据加载到 GPU 内存中，可以提高训练速度。
  # collate_fn=collate_fn：表示用于将数据集中的样本组合成一个 batch 的函数，这里采用了自定义的 TextAudioCollate 函数。
  # batch_sampler=train_sampler：表示使用自定义的 DistributedBucketSampler 对 batch 进行采样。
  train_loader = DataLoader(
    train_dataset,
    num_workers=8,
    shuffle=False,
    collate_fn=collate_fn,
    batch_sampler=train_sampler,
  )

  if rank == 0:
    # 加载验证数据
    # 不会丢弃最后一个 batch
    eval_dataset = TextAudioSpeakerLoader(hps.data.validation_files, hps.data, args.initial_run)
    eval_loader = DataLoader(
      eval_dataset,
      num_workers=4,
      shuffle=False,
      batch_size=hps.train.batch_size,
      drop_last=False,
      collate_fn=collate_fn,
    )
    logger.info('Training Started')
  elif args.initial_run:
    print(f'rank: {rank} is waiting...')

  dist.barrier()

  net_g = SynthesizerTrn(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    n_speakers=len(hps.data.speakers),
    midi_start=hps.data.midi_start,
    midi_end=hps.data.midi_end,
    octave_range=hps.data.octave_range,
    **hps.model
  ).cuda(rank)

  net_d = AvocodoDiscriminator(hps.model.use_spectral_norm).cuda(rank)

  if rank == 0:
    logger.info('MODEL SIZE: G {:.2f}M and D {:.2f}M'.format(
      count_parameters(net_g),
      count_parameters(net_d),
    ))

  optim_g = torch.optim.AdamW(
    net_g.parameters(),
    hps.train.learning_rate,
    betas=hps.train.betas,
    eps=hps.train.eps
  )

  optim_d = torch.optim.AdamW(
    net_d.parameters(),
    hps.train.learning_rate,
    betas=hps.train.betas,
    eps=hps.train.eps
  )

  net_g = DDP(net_g, device_ids=[rank])
  net_d = DDP(net_d, device_ids=[rank])

  try:
    _, _, _, _, epoch_save, _ = utils.load_checkpoint(
      hps.model_dir,
      rank, net_g, net_d, optim_g, optim_d
    )

    epoch_str = epoch_save + 1
    utils.global_step = epoch_save * len(train_loader) + 1
  except:
    epoch_str = 1
    utils.global_step = 0

  scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
    optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
  )
  scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
    optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
  )

  scaler = GradScaler(enabled=hps.train.fp16_run)

  if rank == 0:
    outer_bar = tqdm(
      total=hps.train.epochs,
      desc="Training",
      position=0,
      leave=False
    )
    outer_bar.update(epoch_str)

  for epoch in range(epoch_str, hps.train.epochs + 1):
    if rank == 0:
      train_and_evaluate(
        rank, epoch, hps, [net_g, net_d],
        [optim_g, optim_d], scaler,
        [train_loader, eval_loader], writer
      )
    else:
      train_and_evaluate(
        rank, epoch, hps, [net_g, net_d],
        [optim_g, optim_d], scaler,
        [train_loader, None], None
      )

    scheduler_g.step()
    scheduler_d.step()

    if rank == 0:
      outer_bar.update(1)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '-c',
    '--config',
    type=str,
    default="./configs/config_cje.yaml",
    help='Path to configuration file'
  )
  parser.add_argument(
    '-m',
    '--model',
    type=str,
    default="9Nine",
    help='Model name'
  )
  parser.add_argument(
    '-r',
    '--resume',
    type=str,
    help='Path to checkpoint for resume'
  )
  parser.add_argument(
    '-f',
    '--force_resume',
    type=str,
    help='Path to checkpoint for force resume'
  )
  parser.add_argument(
    '-t',
    '--transfer',
    type=str,
    help='Path to baseline checkpoint for transfer'
  )
  parser.add_argument(
    '-w',
    '--ignore_warning',
    action="store_true",
    help='Ignore warning message'
  )
  parser.add_argument(
    '-i',
    '--initial_run',
    action="store_true",
    help='Inintial run for saving pt files'
  )

  args = parser.parse_args()

  if args.ignore_warning:
    import warnings

    warnings.filterwarnings(action='ignore')

  main(args)
