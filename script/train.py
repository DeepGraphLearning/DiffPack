import os
import sys
import math
import pprint
import argparse
import numpy as np

import torch

from torchvision import datasets
from torch_geometric.data import dataset
import torchdrug
from torch import nn
from torchdrug.patch import patch
from torchdrug import core, datasets, tasks, models, layers
from torchdrug.utils import comm

patch(nn, "Module", nn._Module)

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from diffpack import util, dataset, task
from diffpack.engine import DiffusionEngine


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="yaml configuration file",
                        default="config/inference.yaml")
    parser.add_argument("--seed", help="random seed", type=int, default=0)
    args = parser.parse_known_args()[0]

    return args


def set_seed(seed):
    torch.manual_seed(seed + comm.get_rank())
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return


def build_solver(cfg, logger):
    datasets = []
    for split in ["train", "valid", "test"]:
        if "%s_set" % split in cfg:
            datasets.append(core.Configurable.load_config_dict(cfg["%s_set" % split]))
            if comm.get_rank() == 0:
                logger.warning(datasets[-1])
                logger.warning("#%s: %d" % (split, len(datasets[-1])))
        else:
            datasets.append(None)
    train_set, valid_set, test_set = datasets
    task = core.Configurable.load_config_dict(cfg.task)

    # build solver
    cfg.optimizer.params = [p for p in task.parameters() if p.requires_grad]
    optimizer = core.Configurable.load_config_dict(cfg.optimizer)
    if "scheduler" not in cfg:
        scheduler = None
    else:
        cfg.scheduler.optimizer = optimizer
        scheduler = core.Configurable.load_config_dict(cfg.scheduler)

    # build solver
    solver = DiffusionEngine(task, train_set, valid_set, test_set, optimizer, scheduler, **cfg.engine)
    if "checkpoint" in cfg:
        solver.load(cfg.checkpoint, load_optimizer=cfg.get("load_optimizer", False))

    if "model_checkpoint" in cfg:
        model_checkpoint = os.path.expanduser(cfg.model_checkpoint)
        model_dict = torch.load(model_checkpoint, map_location=torch.device('cpu'))["model"]
        missing_keys, unexpected_keys = task.load_state_dict(model_dict, strict=False)

    # Calculate the parameter number of the model
    if comm.get_rank() == 0:
        logger.warning("#parameter: %d" % sum(p.numel() for p in task.parameters() if p.requires_grad))

    return solver


def train(cfg, solver:DiffusionEngine):
    if not cfg.train.num_epoch > 0:
        return solver

    step = math.ceil(cfg.train.num_epoch / 20)

    for i in range(0, cfg.train.num_epoch, step):
        kwargs = cfg.train.copy()
        kwargs["num_epoch"] = min(step, cfg.train.num_epoch - i)
        solver.train(**kwargs)

        # solver.save("model_epoch_%d.pth" % solver.epoch)
    
    return solver


if __name__ == "__main__":
    args = parse_args()
    args.config = os.path.realpath(args.config)
    cfg = util.load_config(args.config)

    set_seed(args.seed)
    output_dir = util.create_working_directory(cfg)
    logger = util.get_root_logger()
    if comm.get_rank() == 0:
        logger.warning("Config file: %s" % args.config)
        logger.warning(pprint.pformat(cfg))
        logger.warning("Output dir: %s" % output_dir)

    solver = build_solver(cfg, logger)
    solver = train(cfg, solver)
