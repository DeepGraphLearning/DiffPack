import os
import pickle
import logging

import torch
from torchdrug import core, data, utils
from torchdrug.utils import comm
from torch import nn
from torch.utils import data as torch_data
logger = logging.getLogger(__name__)

class DiffusionEngine(core.Engine):
    @torch.no_grad()
    def generate(self, test_set, path):
        if comm.get_rank() == 0:
            logger.warning(f"Test on {test_set}")
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        logger.warning(path)
        dataloader = data.DataLoader(test_set, self.batch_size, shuffle=False)
        model = self.model

        model.eval()
        id = 0
        data_dict = {}
        for batch in dataloader:
            if self.device.type == "cuda":
                batch = utils.cuda(batch, device=self.device)
            true_proteins = batch["graph"].clone()
            pred_proteins = self.model.generate(batch)["graph"]
            evaluation_metric = self.model.get_metric(pred_proteins, true_proteins, {})
            print(f"atom_rmsd_per_residue: {evaluation_metric['atom_rmsd_per_residue'].mean():<20}"
                  f"chi_0_mae_deg: {evaluation_metric['chi_0_ae_deg'].mean():<20}"
                  f"chi_1_mae_deg: {evaluation_metric['chi_1_ae_deg'].mean():<20}"
                  f"chi_2_mae_deg: {evaluation_metric['chi_2_ae_deg'].mean():<20}"
                  f"chi_3_mae_deg: {evaluation_metric['chi_3_ae_deg'].mean():<20}")
            for p in pred_proteins.unpack():
                pdb_file = os.path.basename(test_set.pdb_files[id])
                protein = p.cpu()
                protein.to_pdb(os.path.join(path, pdb_file))
                data_dict[pdb_file] = p.cpu()
                id += 1
