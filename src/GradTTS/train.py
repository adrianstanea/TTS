import logging
import os
import os.path

import hydra
import numpy as np
from omegaconf import OmegaConf
import torch
from conf.hydra_config import AdamConfig, GradTTSConfig
from hydra.core.config_store import ConfigStore
from hydra.utils import get_original_cwd, to_absolute_path

from model.gradTTS import GradTTS
from text.symbols import symbols
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import create_symlink, plot_tensor, save_plot

from data import TextMelBatchCollate, TextMelDataset

logger = logging.getLogger("train.py")
logger.setLevel(logging.DEBUG)

cs = ConfigStore.instance()
cs.store(name="config", node=GradTTSConfig)


@hydra.main(config_path="conf", config_name="config")
def hydra_main(cfg: GradTTSConfig):
    # Fix hydra working directory issue
    os.chdir(get_original_cwd())
    logger.debug(f"Running from: {os.getcwd()}")
    cfg.train.log_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir + '/' + cfg.train.log_dir
    logger.info(f"logging data into: {cfg.train.log_dir}")
    # =============================================================================
    logger.info("Creating a symlink to dataset...")
    source_dir_name = "DUMMY"
    target_path = os.path.join(os.path.expanduser('~'), 'datasets', 'LJSpeech')
    create_symlink(source_dir_name, target_path)
    # =============================================================================
    # Hyperparameters that need to be set at runtime
    nsymbols = len(symbols) + 1 if cfg.data.add_blank else len(symbols)
    # =============================================================================
    logger.info("Current configuration: ")
    logger.info(cfg)
    logger.info("Initializing random seed...")
    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)
    # =============================================================================
    logger.info("Initializing tensorboard...")
    writer = SummaryWriter(log_dir=to_absolute_path(cfg.train.log_dir))

    logger.info("Initializing data loaders...")
    logger.info(
        f"filelist_path: {to_absolute_path(cfg.data.train_filelist_path)}")
    logger.info(f"cmudict_path: {to_absolute_path(cfg.data.cmudict_path)}")

    train_dataset = TextMelDataset(
        filelist_path=to_absolute_path(cfg.data.train_filelist_path),
        cmudict_path=to_absolute_path(cfg.data.cmudict_path),
        random_seed=cfg.train.seed,
        add_blank=cfg.data.add_blank,
        n_fft=cfg.data.n_fft,
        n_mels=cfg.data.n_feats,
        sample_rate=cfg.data.sample_rate,
        hop_length=cfg.data.hop_length,
        win_length=cfg.data.win_length,
        f_min=cfg.data.f_min,
        f_max=cfg.data.f_max,
        preprocess=True
    )
    batch_callate = TextMelBatchCollate()
    loader = DataLoader(dataset=train_dataset,
                        batch_size=cfg.train.batch_size,
                        collate_fn=batch_callate,
                        drop_last=True,
                        num_workers=4,
                        shuffle=False,
                        )
    test_dataset = TextMelDataset(
        filelist_path=to_absolute_path(cfg.data.valid_filelist_path),
        cmudict_path=to_absolute_path(cfg.data.cmudict_path),
        random_seed=cfg.train.seed,
        add_blank=cfg.data.add_blank,
        n_fft=cfg.data.n_fft,
        n_mels=cfg.data.n_feats,
        sample_rate=cfg.data.sample_rate,
        hop_length=cfg.data.hop_length,
        win_length=cfg.data.win_length,
        f_min=cfg.data.f_min,
        f_max=cfg.data.f_max,
        preprocess=True
    )
    logger.info("Initializing model...")
    model = GradTTS(n_vocab=nsymbols,
                    n_spks=1,
                    spk_emb_dim=None,
                    n_enc_channels=cfg.encoder.n_enc_channels,
                    filter_channels=cfg.encoder.filter_channels,
                    filter_channels_dp=cfg.encoder.filter_channels_dp,
                    n_heads=cfg.encoder.n_heads,
                    n_enc_layers=cfg.encoder.n_enc_layers,
                    enc_kernel=cfg.encoder.enc_kernel,
                    enc_dropout=cfg.encoder.enc_dropout,
                    window_size=cfg.encoder.window_size,
                    n_feats=cfg.data.n_feats,
                    dec_dim=cfg.decoder.dec_dim,
                    beta_min=cfg.decoder.beta_min,
                    beta_max=cfg.decoder.beta_max,
                    pe_scale=cfg.decoder.pe_scale).cuda()
    logger.info(
        f"Number of encoder + duration predictor parameters: {model.encoder.nparams/1e6:.2f}m")
    logger.info(
        f"Number of decoder parameters: {model.decoder.nparams/1e6:.2f}m")
    logger.info(f"Total parameters: {model.nparams/1e6:.2f}m")

    logger.info("Initializing optimizer...")
    if OmegaConf.get_type(cfg.optimizer) is AdamConfig:
        optimizer = torch.optim.Adam(params=model.parameters(),
                                     lr=cfg.optimizer.learning_rate)
    else:
        raise NotImplementedError("Only Adam optimizer is supported for now")

    logger.info("Logging test batch...")
    test_batch = test_dataset.sample_test_batch(size=cfg.train.test_size)
    for i, item in enumerate(test_batch):
        mel = item['y']
        writer.add_image(f"mel_{i}/ground_truth", plot_tensor(mel.squeeze()),
                         global_step=0, dataformats="HWC")
        save_plot(mel.squeeze(), f'{cfg.train.log_dir}/original_{i}.png')

    logger.info("Training...")
    iteration = 0
    for epoch in range(1, cfg.train.n_epochs + 1):
        model.train()
        dur_losses, prior_losses, diff_losses = [], [], []
        with tqdm(loader, total=len(train_dataset)//cfg.train.batch_size) as progress_bar:
            for batch_idx, batch in enumerate(progress_bar):
                model.zero_grad()
                x, x_lengths = batch['x'].cuda(), batch['x_lengths'].cuda()
                y, y_lengths = batch['y'].cuda(), batch['y_lengths'].cuda()
                dur_loss, prior_loss, diff_loss = model.compute_loss(x, x_lengths,
                                                                     y, y_lengths,
                                                                     out_size=cfg.train.out_size)
                loss = sum([dur_loss, prior_loss, diff_loss])
                loss.backward()

                enc_grad_norm = torch.nn.utils.clip_grad_norm_(model.encoder.parameters(),
                                                               max_norm=1)
                dec_grad_norm = torch.nn.utils.clip_grad_norm_(model.decoder.parameters(),
                                                               max_norm=1)
                optimizer.step()

                writer.add_scalar('training/duration_loss', dur_loss.item(),
                                  global_step=iteration)
                writer.add_scalar('training/prior_loss', prior_loss.item(),
                                  global_step=iteration)
                writer.add_scalar('training/diffusion_loss', diff_loss.item(),
                                  global_step=iteration)
                writer.add_scalar('training/encoder_grad_norm', enc_grad_norm,
                                  global_step=iteration)
                writer.add_scalar('training/decoder_grad_norm', dec_grad_norm,
                                  global_step=iteration)

                dur_losses.append(dur_loss.item())
                prior_losses.append(prior_loss.item())
                diff_losses.append(diff_loss.item())

                if batch_idx % 5 == 0:
                    msg = f'Epoch: {epoch}, iteration: {iteration} | dur_loss: {dur_loss.item()}, prior_loss: {prior_loss.item()}, diff_loss: {diff_loss.item()}'
                    progress_bar.set_description(msg)

                iteration += 1

        log_msg = 'Epoch %d: duration loss = %.3f ' % (
            epoch, np.mean(dur_losses))
        log_msg += '| prior loss = %.3f ' % np.mean(prior_losses)
        log_msg += '| diffusion loss = %.3f\n' % np.mean(diff_losses)
        with open(f'{cfg.train.log_dir}/train.log', 'a') as f:
            f.write(log_msg)

        if epoch % cfg.train.save_every > 0:
            continue

        model.eval()
        logger.info('Synthesis...')
        with torch.no_grad():
            for i, item in enumerate(test_batch):
                x = item['x'].to(torch.long).unsqueeze(0).cuda()
                x_lengths = torch.LongTensor([x.shape[-1]]).cuda()
                y_enc, y_dec, attn = model(x, x_lengths, n_timesteps=50)
                writer.add_image(f'image_{i}/generated_enc',
                                 plot_tensor(y_enc.squeeze().cpu()),
                                 global_step=iteration, dataformats='HWC')
                writer.add_image(f'image_{i}/generated_dec',
                                 plot_tensor(y_dec.squeeze().cpu()),
                                 global_step=iteration, dataformats='HWC')
                writer.add_image(f'image_{i}/alignment',
                                 plot_tensor(attn.squeeze().cpu()),
                                 global_step=iteration, dataformats='HWC')
                save_plot(y_enc.squeeze().cpu(),
                          f'{cfg.train.log_dir}/generated_enc_{i}.png')
                save_plot(y_dec.squeeze().cpu(),
                          f'{cfg.train.log_dir}/generated_dec_{i}.png')
                save_plot(attn.squeeze().cpu(),
                          f'{cfg.train.log_dir}/alignment_{i}.png')

        logger.info(f"Saving model checkpoint at epoch {epoch}...")
        ckpt = model.state_dict()
        torch.save(ckpt, f=f"{cfg.train.log_dir}/grad_{epoch}.pt")


if __name__ == "__main__":
    hydra_main()
