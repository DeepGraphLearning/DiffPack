import os
import sys
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
    parser.add_argument("-o", "--output_dir", help="output directory", default="output")
    parser.add_argument("-f", "--pdb_files", help="list of pdb files", nargs='*', default=[])
    args = parser.parse_known_args()[0]
    args.output_dir = os.path.expanduser(args.output_dir)
    args.output_dir = os.path.realpath(args.output_dir)

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
    task = core.Configurable.load_config_dict(cfg.task)

    # build solver
    solver = DiffusionEngine(task, None, None, None, None, None, **cfg.engine)
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


if __name__ == "__main__":
    args = parse_args()
    args.config = os.path.realpath(args.config)
    cfg = util.load_config(args.config)
    cfg.test_set.pdb_files = args.pdb_files

    set_seed(args.seed)
    logger = util.get_root_logger()
    if comm.get_rank() == 0:
        logger.warning("Config file: %s" % args.config)
        logger.warning(pprint.pformat(cfg))
        logger.warning("Output dir: %s" % args.output_dir)

    solver = build_solver(cfg, logger)
    test_set = core.Configurable.load_config_dict(cfg.test_set)
    solver.generate(test_set=test_set, path=args.output_dir)
