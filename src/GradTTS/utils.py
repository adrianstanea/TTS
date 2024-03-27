# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import logging
import os
import glob
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import torch

from hydra.utils import get_original_cwd


def intersperse(lst, item):
    # Adds blank symbol
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result


def parse_filelist(filelist_path, split_char="|"):
    with open(filelist_path, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split_char) for line in f]
    return filepaths_and_text


def latest_checkpoint_path(dir_path, regex="grad_*.pt"):
    f_list = glob.glob(os.path.join(dir_path, regex))
    f_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
    x = f_list[-1]
    return x


def load_checkpoint(logdir, model, num=None):
    if num is None:
        model_path = latest_checkpoint_path(logdir, regex="grad_*.pt")
    else:
        model_path = os.path.join(logdir, f"grad_{num}.pt")
    print(f'Loading checkpoint {model_path}...')
    model_dict = torch.load(model_path, map_location=lambda loc, storage: loc)
    model.load_state_dict(model_dict, strict=False)
    return model


def save_figure_to_numpy(fig):
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data


def plot_tensor(tensor):
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(tensor, aspect="auto", origin="lower", interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data


def save_plot(tensor, savepath):
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(tensor, aspect="auto", origin="lower", interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.canvas.draw()
    plt.savefig(savepath)
    plt.close()
    return

def create_symlink(source_dir_name: str, target_path: str):
    """
    Creates a symbolic link from the source path to the target path.

    Args:
        source_dir_name (str): The name of the directory in the local instance.
        target_path (str): The path in the host machine where a dataset is located.
    """
    source_path = Path(get_original_cwd()) / source_dir_name

    logging.debug(f"Creating symlink from {source_path} to {target_path}")

    # Check if the symlink already exists
    if not os.path.islink(source_path):
        if os.path.exists(source_path):
            logging.info(f"File or directory {source_path} already exists and is not a symlink.")
        else:
            os.symlink(target_path, str(source_path))
    else:
        logging.info(f"Symlink {source_path} already exists.")
