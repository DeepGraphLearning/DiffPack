import os
import sys
import time
import random
import logging

import torch
import yaml
import easydict
import jinja2

from torch import distributed as dist

from torchdrug.utils import comm


def meshgrid(dict):
    if len(dict) == 0:
        yield {}
        return

    key = next(iter(dict))
    values = dict[key]
    sub_dict = dict.copy()
    sub_dict.pop(key)

    if not isinstance(values, list):
        values = [values]
    for value in values:
        for result in meshgrid(sub_dict):
            result[key] = value
            yield result


def load_config(cfg_file):
    with open(cfg_file, "r") as fin:
        raw_text = fin.read()
    cfg = yaml.load(raw_text, Loader=yaml.CLoader)
    cfg = easydict.EasyDict(cfg)

    return cfg


def get_root_logger(file=True):
    logger = logging.getLogger("")
    logger.setLevel(logging.INFO)
    format = logging.Formatter("%(asctime)-10s %(message)s", "%H:%M:%S")

    if file:
        handler = logging.FileHandler("log.txt")
        handler.setFormatter(format)
        logger.addHandler(handler)

    return logger


def create_working_directory(cfg):
    file_name = "%s_working_dir.tmp" % os.environ["SLURM_JOB_ID"]
    world_size = comm.get_world_size()
    if world_size > 1 and not dist.is_initialized():
        comm.init_process_group("nccl", init_method="env://")

    output_dir = os.path.join(os.path.expanduser(cfg.output_dir),
                              cfg.task["class"],
                              cfg.task.model["class"] + "_" + time.strftime("%Y-%m-%d-%H-%M-%S"))

    # synchronize working directory
    if comm.get_rank() == 0:
        with open(file_name, "w") as fout:
            fout.write(output_dir)
        os.makedirs(output_dir)
    comm.synchronize()
    if comm.get_rank() != 0:
        with open(file_name, "r") as fin:
            output_dir = fin.read()
    comm.synchronize()
    if comm.get_rank() == 0:
        os.remove(file_name)

    os.chdir(output_dir)
    return output_dir

def rot_matmul(
    a: torch.Tensor,
    b: torch.Tensor
) -> torch.Tensor:
    """
        Performs matrix multiplication of two rotation matrix tensors. Written
        out by hand to avoid AMP downcasting.

        Args:
            a: [*, 3, 3] left multiplicand
            b: [*, 3, 3] right multiplicand
        Returns:
            The product ab
    """
    def row_mul(i):
        return torch.stack(
            [
                a[..., i, 0] * b[..., 0, 0]
                + a[..., i, 1] * b[..., 1, 0]
                + a[..., i, 2] * b[..., 2, 0],
                a[..., i, 0] * b[..., 0, 1]
                + a[..., i, 1] * b[..., 1, 1]
                + a[..., i, 2] * b[..., 2, 1],
                a[..., i, 0] * b[..., 0, 2]
                + a[..., i, 1] * b[..., 1, 2]
                + a[..., i, 2] * b[..., 2, 2],
            ],
            dim=-1,
        )

    return torch.stack(
        [
            row_mul(0),
            row_mul(1),
            row_mul(2),
        ],
        dim=-2
    )


def rot_vec_mul(
    r: torch.Tensor,
    t: torch.Tensor
) -> torch.Tensor:
    """
        Applies a rotation to a vector. Written out by hand to avoid transfer
        to avoid AMP downcasting.

        Args:
            r: [*, 3, 3] rotation matrices
            t: [*, 3] coordinate tensors
        Returns:
            [*, 3] rotated coordinates
    """
    x, y, z = torch.unbind(t, dim=-1)
    return torch.stack(
        [
            r[..., 0, 0] * x + r[..., 0, 1] * y + r[..., 0, 2] * z,
            r[..., 1, 0] * x + r[..., 1, 1] * y + r[..., 1, 2] * z,
            r[..., 2, 0] * x + r[..., 2, 1] * y + r[..., 2, 2] * z,
        ],
        dim=-1,
    )


def convert_model_ckpt(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        for i in range(4):
            if k.startswith(f"model_list.{i}.sigma_embed_layer"):
                new_k = k.replace(f"model_list.{i}.sigma_embed_layer", f"sigma_embedding_list.{i}")
                new_state_dict[new_k] = v
            elif k.startswith(f"model_list.{i}.gearnet"):
                new_k = k.replace(f"model_list.{i}.gearnet", f"model_list.{i}")
                new_state_dict[new_k] = v
            elif k.startswith(f"model_list.{i}.torsion_mlp"):
                new_k = k.replace(f"model_list.{i}.torsion_mlp", f"torsion_mlp_list.{i}")
                new_state_dict[new_k] = v
            elif k.startswith(f"confidence_model."):
                new_k = k
                new_state_dict[new_k] = v
            elif k.startswith(f"mlp."):
                new_k = k
                new_state_dict[new_k] = v
    return new_state_dict
