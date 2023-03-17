import argparse
import math
import os

import torch
import torch.distributed as dist  # 用于在多个GPU上进行分布式训练的模块
import torch.multiprocessing as mp  # 用于在多个进程中运行代码的模块
from phaseaug.phaseaug import PhaseAug
from torch.cuda.amp import autocast, GradScaler  # 用于混合精度训练的模块
from torch.nn import functional as F  # 包含了许多用于构建神经网络的函数，如激活函数、池化等
from torch.nn.parallel import DistributedDataParallel as DDP  # 实现了多GPU分布式训练
from torch.utils.data import DataLoader  # 用于数据加载和处理的模块
from torch.utils.tensorboard import SummaryWriter  # 用于可视化训练和评估指标的模块
from tqdm import tqdm

import commons
import utils
from data_utils import (
  TextAudioSpeakerLoader,
  TextAudioSpeakerCollate,
  DistributedBucketSampler, create_spec
)
from losses import (
  generator_loss,
  discriminator_loss,
  feature_loss,
  kl_loss
)
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from models import (
  SynthesizerTrn,
  AvocodoDiscriminator
)
from text.symbols import symbols

# PyTorch使用CUDNN加速库
torch.backends.cudnn.benchmark = True
# 跟踪训练步数
global_step = 0


def main(args):
  # 检查CUDA是否可用，如果不可用，则抛出异常
  assert torch.cuda.is_available(), "CPU training is not allowed."

  # 获取当前可用的GPU数量
  num_gpus = torch.cuda.device_count()
  # 获取配置
  hps = utils.get_hparams()

  # create spectrogram files
  create_spec(hps.data.training_files, hps.data)
  create_spec(hps.data.validation_files, hps.data)

  if num_gpus == 1:
    return run(0, 1, hps, args)

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

  # 记录模型训练的步数
  global global_step

  if rank == 0:
    # 日志记录器
    logger = utils.get_logger(hps.model_dir)
    # 输出配置
    logger.info('MODEL NAME: {} in {}'.format(args.model, hps.model_dir))
    logger.info(
      'GPU: Use {} gpu(s) with batch size {} (FP16 running: {})'.format(
        num_gpus, hps.train.batch_size, hps.train.fp16_run)
    )

    logger.debug(hps)

    # 检查代码仓库的版本信息是否与当前运行的代码一致
    utils.check_git_hash(hps.model_dir)
    # 记录训练日志
    writer = SummaryWriter(log_dir=hps.model_dir)

  if num_gpus > 1:
    # 初始化进程组
    # backend: 使用 NVIDIA 提供的通信库 NCCL 进行通信
    # init_method: 使用环境变量的方式初始化进程组
    # world_size: 进程组的总数
    dist.init_process_group(
      backend='nccl',
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
  train_dataset = TextAudioSpeakerLoader(hps.data.training_files, hps.data,
                                         rank == 0 and args.initial_run)
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
    pin_memory=False,
    collate_fn=collate_fn,
    persistent_workers=True,
    batch_sampler=train_sampler,
  )

  if rank == 0:
    # 加载验证数据
    # 不会丢弃最后一个 batch
    eval_dataset = TextAudioSpeakerLoader(hps.data.validation_files, hps.data)
    eval_loader = DataLoader(
      eval_dataset,
      num_workers=8,
      shuffle=False,
      batch_size=hps.train.batch_size,
      pin_memory=False,
      drop_last=False,
      collate_fn=collate_fn,
      persistent_workers=True,
    )

  net_g = SynthesizerTrn(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    n_speakers=len(hps.data.speakers),
    midi_start=hps.data.midi_start,
    midi_end=hps.data.midi_end,
    octave_range=hps.data.octave_range,
    **hps.model).cuda(rank)

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

  net_g = DDP(net_g, device_ids=[rank], find_unused_parameters=True)
  net_d = DDP(net_d, device_ids=[rank], find_unused_parameters=True)

  try:
    _, _, _, _, _, epoch_save, _ = utils.load_checkpoint(
      utils.latest_checkpoint_path(hps.model_dir, "*.pth"),
      rank, net_g, net_d, optim_g, optim_d
    )

    epoch_str = epoch_save + 1
    global_step = epoch_save * len(train_loader) + 1
  except:
    epoch_str = 1
    global_step = 0

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


def train_and_evaluate(rank, epoch, hps, nets, optims, scaler, loaders, logger, writer):
  net_g, net_d = nets
  optim_g, optim_d = optims

  train_loader, eval_loader = loaders
  train_loader.batch_sampler.set_epoch(epoch)

  aug = PhaseAug().cuda(rank)

  global global_step

  net_g.train()
  net_d.train()

  if rank == 0:
    inner_bar = tqdm(
      total=len(train_loader),
      desc="Epoch {}".format(epoch),
      position=1,
      leave=False
    )

  for batch_idx, (x, x_lengths, spec, spec_lengths, ying, ying_lengths, y,
                  y_lengths, speakers, tone) in enumerate(train_loader):

    x = x.cuda(rank, non_blocking=True)
    x_lengths = x_lengths.cuda(rank, non_blocking=True)

    spec = spec.cuda(rank, non_blocking=True)
    spec_lengths = spec_lengths.cuda(rank, non_blocking=True)

    ying = ying.cuda(rank, non_blocking=True)
    ying_lengths = ying_lengths.cuda(rank, non_blocking=True)

    y = y.cuda(rank, non_blocking=True)
    y_lengths = y_lengths.cuda(rank, non_blocking=True)

    speakers = speakers.cuda(rank, non_blocking=True)
    tone = tone.cuda(rank, non_blocking=True)

    with autocast(enabled=hps.train.fp16_run):
      y_hat, l_length, attn, ids_slice, x_mask, z_mask, y_hat_, \
        (z, z_p, m_p, logs_p, m_q, logs_q), _, \
        (z_spec, m_spec, logs_spec, spec_mask, z_yin, m_yin, logs_yin, yin_mask), \
        (yin_gt_crop, yin_gt_shifted_crop, yin_dec_crop, yin_hat_crop, scope_shift, yin_hat_shifted) \
        = net_g(x, tone, x_lengths, spec, spec_lengths, ying, ying_lengths, speakers)

      mel = spec_to_mel_torch(
        spec, hps.data.filter_length,
        hps.data.n_mel_channels,
        hps.data.sampling_rate, hps.data.mel_fmin,
        hps.data.mel_fmax
      )

      y_mel = commons.slice_segments(
        mel, ids_slice, hps.train.segment_size // hps.data.hop_length
      )

      y_hat_mel = mel_spectrogram_torch(
        y_hat[-1].squeeze(1), hps.data.filter_length,
        hps.data.n_mel_channels, hps.data.sampling_rate,
        hps.data.hop_length, hps.data.win_length, hps.data.mel_fmin,
        hps.data.mel_fmax
      )

      yin_gt_crop = commons.slice_segments(
        torch.cat([yin_gt_crop, yin_gt_shifted_crop], dim=0),
        ids_slice, hps.train.segment_size // hps.data.hop_length
      )

      y_ = commons.slice_segments(
        torch.cat([y, y], dim=0),
        ids_slice * hps.data.hop_length,
        hps.train.segment_size
      )  # sliced

      # Discriminator
      with autocast(enabled=False):
        aug_y_, aug_y_hat_last = aug.forward_sync(
          y_, y_hat_[-1].detach()
        )
        aug_y_hat_ = [_y.detach() for _y in y_hat_[:-1]]
        aug_y_hat_.append(aug_y_hat_last)

      y_d_hat_r, y_d_hat_g, _, _ = net_d(aug_y_, aug_y_hat_)

      with autocast(enabled=False):
        loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
          y_d_hat_r, y_d_hat_g
        )
        loss_disc_all = loss_disc

    optim_d.zero_grad()
    scaler.scale(loss_disc_all).backward()
    scaler.unscale_(optim_d)
    grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
    scaler.step(optim_d)

    p = float(batch_idx + epoch *
              len(train_loader)) / hps.train.alpha / len(train_loader)
    alpha = 2. / (1. + math.exp(-20 * p)) - 1

    with autocast(enabled=hps.train.fp16_run):
      # Generator
      with autocast(enabled=False):
        aug_y_, aug_y_hat_last = aug.forward_sync(y_, y_hat_[-1])
        aug_y_hat_ = y_hat_
        aug_y_hat_[-1] = aug_y_hat_last

      y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(aug_y_, aug_y_hat_)

      with autocast(enabled=False):
        loss_dur = torch.sum(l_length.float())
        loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel

        loss_kl = kl_loss(
          z_p, logs_q, m_p, logs_p, z_mask
        ) * hps.train.c_kl

        loss_yin_dec = F.l1_loss(
          yin_gt_shifted_crop,
          yin_dec_crop
        ) * hps.train.c_yin

        loss_yin_shift = F.l1_loss(
          torch.exp(-yin_gt_crop),
          torch.exp(-yin_hat_crop)
        ) * hps.train.c_yin + F.l1_loss(
          torch.exp(-yin_hat_shifted),
          torch.exp(-(torch.chunk(yin_hat_crop, 2, dim=0)[1]))
        ) * hps.train.c_yin

        loss_fm = feature_loss(fmap_r, fmap_g)
        loss_gen, losses_gen = generator_loss(y_d_hat_g)
        loss_gen_all = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl + loss_yin_shift + loss_yin_dec

    optim_g.zero_grad()
    scaler.scale(loss_gen_all).backward()
    scaler.unscale_(optim_g)
    grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
    scaler.step(optim_g)
    scaler.update()

    if rank == 0:
      inner_bar.update(1)
      inner_bar.set_description(
        "Epoch {} | g {: .04f} d {: .04f}|".format(
          epoch, loss_gen_all, loss_disc_all))
      if global_step % hps.train.log_interval == 0:
        lr = optim_g.param_groups[0]['lr']

        scalar_dict = {
          "learning_rate": lr,
          "loss/g/score": sum(losses_gen),
          "loss/g/fm": loss_fm,
          "loss/g/mel": loss_mel,
          "loss/g/dur": loss_dur,
          "loss/g/kl": loss_kl,
          "loss/g/yindec": loss_yin_dec,
          "loss/g/yinshift": loss_yin_shift,
          "loss/g/total": loss_gen_all,
          "loss/d/real": sum(losses_disc_r),
          "loss/d/gen": sum(losses_disc_g),
          "loss/d/total": loss_disc_all,
        }

        utils.summarize(
          writer=writer,
          global_step=global_step,
          scalars=scalar_dict
        )

        print([global_step, loss_gen_all.item(), loss_disc_all.item(), lr])

      if global_step % hps.train.eval_interval == 0:
        evaluate(hps, global_step, epoch, net_g, eval_loader, writer)

      if global_step % hps.train.save_interval == 0:
        utils.save_checkpoint(
          net_g, optim_g, net_d, optim_d, hps, epoch,
          hps.train.learning_rate,
          os.path.join(hps.model_dir, "{}_{}.pth".format(hps.model_name, global_step))
        )

        try:
          keep_ckpts = getattr(hps.train, 'keep_ckpts', 0)

          if keep_ckpts > 0:
            rm_path = os.path.join(
              hps.model_dir,
              "{}_{}.pth".format(hps.model_name, global_step - hps.train.save_interval * keep_ckpts)
            )

            os.remove(rm_path)
            print()
            print("remove ", rm_path)

        except:
          pass

    global_step += 1


def evaluate(hps, current_step, epoch, generator, eval_loader, writer):
  generator.eval()
  n_sample = hps.train.n_sample

  with torch.no_grad():
    loss_val_mel = 0
    loss_val_yin = 0
    val_bar = tqdm(
      total=len(eval_loader),
      desc="Validation (Step {})".format(current_step),
      position=1,
      leave=False
    )

    for batch_idx, (x, x_lengths, spec, spec_lengths,
                    ying, ying_lengths, y, y_lengths, speakers,
                    tone) in enumerate(eval_loader):

      x = x.cuda(0, non_blocking=True),
      x_lengths = x_lengths.cuda(0, non_blocking=True)

      spec, = spec.cuda(0, non_blocking=True)
      spec_lengths = spec_lengths.cuda(0, non_blocking=True)

      ying, = ying.cuda(0, non_blocking=True)
      ying_lengths = ying_lengths.cuda(0, non_blocking=True)

      y = y.cuda(0, non_blocking=True)
      y_lengths = y_lengths.cuda(0, non_blocking=True)

      speakers = speakers.cuda(0, non_blocking=True)
      tone = tone.cuda(0, non_blocking=True)

      with autocast(enabled=hps.train.fp16_run):
        y_hat, l_length, attn, ids_slice, x_mask, z_mask, y_hat_, \
          (z, z_p, m_p, logs_p, m_q, logs_q), \
          _, \
          (z_spec, m_spec, logs_spec, spec_mask, z_yin, m_yin, logs_yin, yin_mask), \
          (yin_gt_crop, yin_gt_shifted_crop, yin_dec_crop, yin_hat_crop, scope_shift, yin_hat_shifted) \
          = generator.module(x, tone, x_lengths, spec, spec_lengths, ying, ying_lengths, speakers)

        mel = spec_to_mel_torch(
          spec, hps.data.filter_length,
          hps.data.n_mel_channels,
          hps.data.sampling_rate,
          hps.data.mel_fmin, hps.data.mel_fmax
        )

        y_mel = commons.slice_segments(
          mel, ids_slice,
          hps.train.segment_size // hps.data.hop_length
        )

        y_hat_mel = mel_spectrogram_torch(
          y_hat[-1].squeeze(1), hps.data.filter_length,
          hps.data.n_mel_channels, hps.data.sampling_rate,
          hps.data.hop_length, hps.data.win_length,
          hps.data.mel_fmin, hps.data.mel_fmax
        )

        with autocast(enabled=False):
          loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
          loss_val_mel += loss_mel.item()

          loss_yin = F.l1_loss(
            yin_gt_shifted_crop,
            yin_dec_crop
          ) * hps.train.c_yin

          loss_val_yin += loss_yin.item()

      if batch_idx == 0:
        x = x[:n_sample]
        x_lengths = x_lengths[:n_sample]

        spec = spec[:n_sample]
        spec_lengths = spec_lengths[:n_sample]

        ying = ying[:n_sample]
        ying_lengths = ying_lengths[:n_sample]

        y = y[:n_sample]
        y_lengths = y_lengths[:n_sample]

        speakers = speakers[:n_sample]
        tone = tone[:1]

        decoder_inputs, _, mask, (z_crop, z, *_) \
          = generator.module.infer_pre_decoder(x, tone, x_lengths, speakers, max_len=2000)

        y_hat = generator.module.infer_decode_chunk(decoder_inputs, speakers)

        # scope-shifted
        z_spec, z_yin = torch.split(
          z,
          hps.model.inter_channels -
          hps.model.yin_channels,
          dim=1
        )

        z_yin_crop = generator.module.crop_scope([z_yin], 6)[0]
        z_crop_shift = torch.cat([z_spec, z_yin_crop], dim=1)

        decoder_inputs_shift = z_crop_shift * mask
        y_hat_shift = generator.module.infer_decode_chunk(decoder_inputs_shift, speakers)

        z_yin = z_yin * mask
        yin_hat = generator.module.yin_dec_infer(z_yin, mask, speakers)

        y_hat_mel_length = mask.sum([1, 2]).long()
        y_hat_lengths = y_hat_mel_length * hps.data.hop_length

        mel = spec_to_mel_torch(
          spec, hps.data.filter_length,
          hps.data.n_mel_channels,
          hps.data.sampling_rate,
          hps.data.mel_fmin, hps.data.mel_fmax
        )

        y_hat_mel = mel_spectrogram_torch(
          y_hat.squeeze(1).float(), hps.data.filter_length,
          hps.data.n_mel_channels, hps.data.sampling_rate,
          hps.data.hop_length, hps.data.win_length,
          hps.data.mel_fmin, hps.data.mel_fmax
        )

        y_hat_shift_mel = mel_spectrogram_torch(
          y_hat_shift.squeeze(1).float(), hps.data.filter_length,
          hps.data.n_mel_channels, hps.data.sampling_rate,
          hps.data.hop_length, hps.data.win_length,
          hps.data.mel_fmin, hps.data.mel_fmax
        )

        y_hat_pad = F.pad(
          y_hat, (
            hps.data.filter_length - hps.data.hop_length,
            hps.data.filter_length - hps.data.hop_length +
            (-y_hat.shape[-1]) % hps.data.hop_length +
            hps.data.hop_length *
            (y_hat.shape[-1] % hps.data.hop_length == 0)
          ),
          mode='reflect'
        ).squeeze(1)

        y_hat_shift_pad = F.pad(
          y_hat_shift, (
            hps.data.filter_length - hps.data.hop_length,
            hps.data.filter_length - hps.data.hop_length +
            (-y_hat.shape[-1]) % hps.data.hop_length +
            hps.data.hop_length *
            (y_hat.shape[-1] % hps.data.hop_length == 0)
          ),
          mode='reflect'
        ).squeeze(1)

        ying_hat = generator.module.pitch.yingram(y_hat_pad)
        ying_hat_shift = generator.module.pitch.yingram(y_hat_shift_pad)

        if y_hat_mel.size(2) < mel.size(2):
          zero = torch.full((
            n_sample, y_hat_mel.size(1), mel.size(2) - y_hat_mel.size(2)
          ), -11.5129
          ).to(y_hat_mel.device)

          y_hat_mel = torch.cat((y_hat_mel, zero), dim=2)
          y_hat_shift_mel = torch.cat((y_hat_shift_mel, zero), dim=2)

          zero = torch.full((
            n_sample, yin_hat.size(1), mel.size(2) - yin_hat.size(2)
          ), 0
          ).to(y_hat_mel.device)

          yin_hat = torch.cat((yin_hat, zero), dim=2)
          zero = torch.full((
            n_sample, ying_hat.size(1),
            mel.size(2) - ying_hat.size(2)
          ), 0
          ).to(y_hat_mel.device)

          ying_hat = torch.cat((ying_hat, zero), dim=2)
          ying_hat_shift = torch.cat((ying_hat_shift, zero), dim=2)

          zero = torch.full((
            n_sample, z_yin.size(1), mel.size(2) - z_yin.size(2)
          ), 0
          ).to(y_hat_mel.device)

          z_yin = torch.cat((z_yin, zero), dim=2)

          ids = torch.arange(0, mel.size(2)).unsqueeze(0).expand(
            mel.size(1), -1
          ).unsqueeze(0).expand(
            n_sample, -1, -1
          ).to(y_hat_mel_length.device)

          mask = ids > y_hat_mel_length.unsqueeze(1).expand(
            -1, mel.size(1)
          ).unsqueeze(2).expand(
            -1, -1, mel.size(2)
          )

          y_hat_mel.masked_fill_(mask, -11.5129)
          y_hat_shift_mel.masked_fill_(mask, -11.5129)

        image_dict = dict()
        audio_dict = dict()

        for i in range(n_sample):
          image_dict.update({
            "gen/{}_mel".format(i):
              utils.plot_spectrogram_to_numpy(
                y_hat_mel[i].cpu().numpy()
              )
          })

          audio_dict.update({
            "gen/{}_audio".format(i):
              y_hat[i, :, :y_hat_lengths[i]]
          })

          image_dict.update({
            "gen/{}_mel_shift".format(i):
              utils.plot_spectrogram_to_numpy(
                y_hat_shift_mel[i].cpu().numpy()
              )
          })

          audio_dict.update({
            "gen/{}_audio_shift".format(i):
              y_hat_shift[i, :, :y_hat_lengths[i]]
          })

          image_dict.update({
            "gen/{}_z_yin".format(i):
              utils.plot_spectrogram_to_numpy(z_yin[i].cpu().numpy())
          })

          image_dict.update({
            "gen/{}_yin_dec".format(i):
              utils.plot_spectrogram_to_numpy(
                yin_hat[i].cpu().numpy()
              )
          })

          image_dict.update({
            "gen/{}_ying".format(i):
              utils.plot_spectrogram_to_numpy(
                ying_hat[i].cpu().numpy()
              )
          })

          image_dict.update({
            "gen/{}_ying_shift".format(i):
              utils.plot_spectrogram_to_numpy(
                ying_hat_shift[i].cpu().numpy()
              )
          })

        if current_step == 0:
          for i in range(n_sample):
            image_dict.update({
              "gt/{}_mel".format(i):
                utils.plot_spectrogram_to_numpy(
                  mel[i].cpu().numpy()
                )
            })

            image_dict.update({
              "gt/{}_ying".format(i):
                utils.plot_spectrogram_to_numpy(
                  ying[i].cpu().numpy()
                )
            })

            audio_dict.update(
              {"gt/{}_audio".format(i): y[i, :, :y_lengths[i]]}
            )

        utils.summarize(
          writer=writer,
          global_step=current_step,
          images=image_dict,
          audios=audio_dict,
          audio_sampling_rate=hps.data.sampling_rate
        )

      val_bar.update(1)

    loss_val_mel = loss_val_mel / len(eval_loader)
    loss_val_yin = loss_val_yin / len(eval_loader)

    scalar_dict = {
      "loss/val/mel": loss_val_mel,
      "loss/val/yin": loss_val_yin,
    }

    utils.summarize(
      writer=writer,
      global_step=current_step,
      scalars=scalar_dict
    )

  generator.train()


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '-c',
    '--config',
    type=str,
    default="./configs/default.yaml",
    help='Path to configuration file'
  )
  parser.add_argument(
    '-m',
    '--model',
    type=str,
    required=True,
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
