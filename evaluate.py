import torch
from torch.cuda.amp import autocast
from torch.nn import functional as F
from tqdm import tqdm

import commons
import utils
from mel_processing import spec_to_mel_torch, mel_spectrogram_torch


def evaluate(hps, current_step, generator, eval_loader, writer):
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

      x = x.cuda(0, non_blocking=True)
      x_lengths = x_lengths.cuda(0, non_blocking=True)

      spec = spec.cuda(0, non_blocking=True)
      spec_lengths = spec_lengths.cuda(0, non_blocking=True)

      ying = ying.cuda(0, non_blocking=True)
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
