import math
import os

import torch
from phaseaug.phaseaug import PhaseAug
from torch.cuda.amp import autocast
from torch.nn import functional as F
from tqdm import tqdm

import commons
import utils
from evaluate import evaluate
from losses import discriminator_loss, kl_loss, feature_loss, generator_loss
from mel_processing import spec_to_mel_torch, mel_spectrogram_torch


def train_and_evaluate(rank, epoch, hps, nets, optims, scaler, loaders, writer):
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
        evaluate(hps, global_step, net_g, eval_loader, writer)

      if global_step % hps.train.save_interval == 0:
        utils.save_checkpoint(
          net_g, optim_g, net_d, optim_d,
          hps, epoch,
          global_step
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
