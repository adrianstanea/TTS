from dataclasses import dataclass, field
from typing import Any, List

from model.utils import fix_len_compatibility


@dataclass
class DataConfig:
    train_filelist_path: str = 'resources/filelists/ljspeech/train.txt'
    valid_filelist_path: str = 'resources/filelists/ljspeech/valid.txt'
    test_filelist_path: str = 'resources/filelists/ljspeech/test.txt'
    cmudict_path: str = 'resources/cmu_dictionary'
    add_blank: bool = True
    n_feats: int = 80
    n_spks: int = 1  # 247 for Libri-TTS filelist and 1 for LJSpeech
    spk_emb_dim: int = 64
    n_fft: int = 1024
    sample_rate: int = 22050
    hop_length: int = 256
    win_length: int = 1024
    f_min: int = 0
    f_max: int = 8000


@dataclass
class EncoderConfig:
    n_enc_channels: int = 192
    filter_channels: int = 768
    filter_channels_dp: int = 256
    n_enc_layers: int = 6
    enc_kernel: int = 3
    enc_dropout: float = 0.1
    n_heads: int = 2
    window_size: int = 4


@dataclass
class DecoderConfig:
    dec_dim: int = 64
    beta_min: float = 0.05
    beta_max: float = 20.0
    pe_scale: int = 1000  # 1 for `grad-tts-old.pt` checkpoint


@dataclass
class TrainConfig:
    log_dir: str = 'logs/new_exp'
    test_size: int = 4
    n_epochs: int = 10
    batch_size: int = 16
    seed: int = 37
    save_every: int = 1
    out_size: int = fix_len_compatibility(2*22050//256)


@dataclass
class AdamConfig:
    learning_rate: float = 1e-4


@dataclass
class GradTTSConfig:
    data: DataConfig = DataConfig()
    optimizer: AdamConfig = AdamConfig()
    encoder: EncoderConfig = EncoderConfig()
    decoder: DecoderConfig = DecoderConfig()
    train: TrainConfig = TrainConfig()
