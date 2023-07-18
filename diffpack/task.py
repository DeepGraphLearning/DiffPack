from copy import deepcopy
from typing import Optional, Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch_scatter import scatter_mean
from torchdrug import core, tasks, layers
from torchdrug.core import Registry as R
from tqdm import tqdm

from diffpack import rotamer, layer
from diffpack.rotamer import atom_name_vocab, bb_atom_name, res_sym_atom_posn, _rmsd_per_residue, _get_symm_atoms
from diffpack.schedule import SO2VESchedule


@R.register("tasks.TorsionalDiffusion")
class TorsionalDiffusion(tasks.Task, core.Configurable):
    """
    NewTorsionalDiffusion is a class for simulating the torsional diffusion of a protein model.

    It inherits from the tasks.Task and core.Configurable classes and uses these to setup and control the diffusion
    simulation.

    Attributes:
        eps (float): A small number to avoid division by zero errors.
        _option_members (set): A set containing the names of class attributes.
        model (nn.Module): The neural network model to be used.
        schedule_1pi_periodic (SO2VESchedule): The schedule for 1pi periodic tasks.
        schedule_2pi_periodic (SO2VESchedule): The schedule for 2pi periodic tasks.
        num_mlp_layer (int): The number of layers in the model.
        graph_construction_model (Optional[Any]): The model used for graph construction.
        verbose (int): Verbosity level.
        train_chi_id (Optional[Any]): Chi angle for training ranging from 0 to 3. If not specified, random chi angles are trained.
    """
    NUM_CHI_ANGLES = 4
    eps = 1e-10
    _option_members = {"task", "criterion", "metric"}

    def __init__(self, sigma_embedding: nn.Module,
                 model: nn.Module,
                 torsion_mlp_hidden_dims: list,
                 schedule_1pi_periodic: SO2VESchedule,
                 schedule_2pi_periodic: SO2VESchedule,
                 graph_construction_model: Optional[Any] = None,
                 verbose: int = 0,
                 train_chi_id: Optional[Any] = None, ):
        super(TorsionalDiffusion, self).__init__()
        self.torsion_mlp_hidden_dims = torsion_mlp_hidden_dims
        self.model_list = nn.ModuleList([deepcopy(model) for _ in range(self.NUM_CHI_ANGLES)])
        self.sigma_embedding_list = nn.ModuleList([deepcopy(sigma_embedding) for _ in range(self.NUM_CHI_ANGLES)])
        self.torsion_mlp_list = nn.ModuleList([layers.MLP(self.model_list[i].output_dim, torsion_mlp_hidden_dims
                                                          + [4,]) for i in range(self.NUM_CHI_ANGLES)])
        self.schedule_2pi_periodic = schedule_2pi_periodic
        self.schedule_1pi_periodic = schedule_1pi_periodic
        self.graph_construction_model = graph_construction_model
        self.verbose = verbose
        self.train_chi_id = train_chi_id

    def forward(self, batch):
        all_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        metric = {}

        # Sample from the schedule
        protein = batch['graph']
        t = self.schedule_1pi_periodic.sample_train_t(shape=(protein.batch_size,)).to(self.device)

        # Add noise to protein
        train_chi_id = np.random.randint(self.NUM_CHI_ANGLES) if self.train_chi_id is None else self.train_chi_id
        batch = self.add_noise(batch, t, chi_id=train_chi_id)

        # Predict and take loss
        pred = self.predict(batch, all_loss, metric)
        target = self.target(batch)

        metric = self.evaluate(pred, target)
        all_loss += metric["diffusion loss"]

        return all_loss, metric

    def add_noise(self, batch, t, chi_id=None):
        """Add noise to protein and update protein

        Args:
            batch (dict): batch
            t (torch.Tensor): [num_graph] random number in [0, 1]

        Returns:
            batch (dict): dict with the following attributes:
                Protein:
                    chi_1pi_periodic_mask (torch.Tensor): [num_residue, 4] bool
                    chi_2pi_periodic_mask (torch.Tensor): [num_residue, 4] bool
                    chi_mask (torch.Tensor): [num_residue, 4] bool
                chi_id (int): chi angle to be trained
                sigma (torch.Tensor): [num_graph] sigma
                score (torch.Tensor): [num_residue, 4] score
        """
        protein = batch['graph']
        if chi_id is not None:
            protein = rotamer.remove_by_chi(protein, chi_id)
        chis = rotamer.get_chis(protein, protein.node_position)  # [num_residue, 4]

        # Add noise to chis
        chis, score_1pi = self.schedule_1pi_periodic.add_noise(chis, t, protein.chi_1pi_periodic_mask)
        chis, score_2pi = self.schedule_2pi_periodic.add_noise(chis, t, protein.chi_2pi_periodic_mask)
        score = torch.where(protein.chi_1pi_periodic_mask, score_1pi, score_2pi)
        protein = rotamer.set_chis(protein, chis)  # TODO：maybe have bug

        batch['protein'] = protein
        batch['chi_id'] = chi_id
        batch['sigma'] = self.schedule_1pi_periodic.t_to_sigma(t)
        batch['score'] = score
        return batch

    def predict(self, batch, all_loss=None, metric=None):
        protein = batch['graph']
        chi_id = batch['chi_id']
        sigma = batch['sigma']                  # [num_graph]
        if self.graph_construction_model:
            protein = self.graph_construction_model(protein)

        # Model forward
        node_sigma = sigma[protein.atom2graph]  # [num_node]
        node_feature = self.sigma_embedding_list[chi_id](protein.node_feature.float(), node_sigma)
        node_feature = self.model_list[chi_id](protein, node_feature, all_loss=all_loss, metric=metric)["node_feature"]
        residue_feature = scatter_mean(node_feature, protein.atom2residue, dim=0, dim_size=protein.num_residue)
        pred = self.torsion_mlp_list[chi_id](residue_feature)

        # Scaled by norm
        torsion_sigma = sigma[protein.residue2graph].unsqueeze(-1).expand(-1, self.NUM_CHI_ANGLES)  # [num_residue, 4]
        score_norm_1pi = torch.tensor(self.schedule_1pi_periodic.score_norm(torsion_sigma), device=self.device)
        score_norm_2pi = torch.tensor(self.schedule_2pi_periodic.score_norm(torsion_sigma), device=self.device)
        score_norm = torch.where(protein.chi_1pi_periodic_mask, score_norm_1pi, score_norm_2pi)
        pred_score = pred * score_norm.sqrt()

        # Mask out non-related chis
        pred_score = pred_score * protein.chi_mask.to(pred_score.dtype)

        return pred_score, score_norm

    def target(self, batch):
        protein = batch["graph"]
        target_score = batch['score']  # Move to protein attribute
        target_score = target_score * protein.chi_mask
        return target_score

    def evaluate(self, pred, target):
        metric = {}
        pred_score, score_norm = pred
        target_score = target

        metric["diffusion loss"] = ((target_score - pred_score) ** 2 / (score_norm + self.eps)).mean()
        metric["diffusion base loss"] = (pred_score ** 2 / (score_norm + self.eps)).mean()

        return metric

    @torch.no_grad()
    def generate(self, batch, randomize=True):
        protein = batch['graph']
        if randomize:
            protein = rotamer.randomize(protein)

        schedule = self.schedule_1pi_periodic.reverse_t_schedule.to(self.device)
        for chi_id in tqdm(range(self.NUM_CHI_ANGLES), desc="Autoregressive generation"):
            for j in range(len(schedule) - 1):
                t = schedule[j]
                dt = schedule[j] - schedule[j + 1] if j + 1 < len(schedule) else 1
                chis = rotamer.get_chis(protein, protein.node_position)  # [num_residue, 4]

                # Predict score
                sigma = self.schedule_1pi_periodic.t_to_sigma(t).repeat(protein.batch_size)
                chi_protein = rotamer.remove_by_chi(protein, chi_id)
                pred_score, _ = self.predict({
                    "graph": chi_protein,
                    "sigma": sigma,
                    "chi_id": chi_id
                })

                # Step backward
                chis = self.schedule_1pi_periodic.step(chis, pred_score, t, dt, chi_protein.chi_1pi_periodic_mask)
                chis = self.schedule_2pi_periodic.step(chis, pred_score, t, dt, chi_protein.chi_2pi_periodic_mask)
                protein = rotamer.set_chis(protein, chis)
        return batch

    def get_metric(self, pred_protein, true_protein, metric):
        # assert pred_pos.shape == true_pos.shape
        pred_pos = pred_protein.node_position
        true_pos = true_protein.node_position
        protein = true_protein

        pred_pos_per_residue = torch.zeros(protein.num_residue, len(atom_name_vocab), 3, device=protein.device)
        true_pos_per_residue = torch.zeros(protein.num_residue, len(atom_name_vocab), 3, device=protein.device)
        pred_pos_per_residue[protein.atom2residue, protein.atom_name] = pred_pos
        true_pos_per_residue[protein.atom2residue, protein.atom_name] = true_pos
        symm_true_pos_per_residue = _get_symm_atoms(true_pos_per_residue, protein.residue_type)

        # Symmetric alignment
        rmsd_per_residue = _rmsd_per_residue(pred_pos_per_residue, true_pos_per_residue, protein.sidechain37_mask)
        sym_rmsd_per_residue = _rmsd_per_residue(pred_pos_per_residue, symm_true_pos_per_residue,
                                                 protein.sidechain37_mask)
        sym_replace_mask = rmsd_per_residue > sym_rmsd_per_residue
        rmsd_per_residue[sym_replace_mask] = sym_rmsd_per_residue[sym_replace_mask]
        true_pos_per_residue[sym_replace_mask] = symm_true_pos_per_residue[sym_replace_mask]
        true_pos = true_pos_per_residue[protein.atom2residue, protein.atom_name]
        metric["atom_rmsd_per_residue"] = rmsd_per_residue

        pred_chi = rotamer.get_chis(protein, pred_pos)
        true_chi = rotamer.get_chis(protein, true_pos)
        chi_diff = (pred_chi - true_chi).abs()
        chi_ae = torch.minimum(chi_diff, 2 * np.pi - chi_diff)
        chi_ae_periodic = torch.minimum(chi_ae, np.pi - chi_ae)
        chi_ae[protein.chi_1pi_periodic_mask] = chi_ae_periodic[protein.chi_1pi_periodic_mask]
        metric["chi_ae_rad"] = chi_ae[protein.chi_mask]  # [num_residue, 4]
        metric["chi_ae_deg"] = chi_ae[protein.chi_mask] * 180 / np.pi  # [num_residue, 4]
        for i in range(self.NUM_CHI_ANGLES):
            metric[f"chi_{i}_ae_rad"] = chi_ae[:, i][protein.chi_mask[:, i]]
            metric[f"chi_{i}_ae_deg"] = chi_ae[:, i][protein.chi_mask[:, i]] * 180 / np.pi

        return metric


@R.register("tasks.ConfidencePrediction")
class ConfidencePrediction(TorsionalDiffusion):
    eps = 1e-10
    _option_members = {"task", "criterion", "metric"}

    def __init__(self, sigma_embedding: nn.Module,
                 model: nn.Module,
                 confidence_model: nn.Module,
                 torsion_mlp_hidden_dims: list,
                 schedule_1pi_periodic: SO2VESchedule,
                 schedule_2pi_periodic: SO2VESchedule,
                 num_sample: int = 5,
                 num_mlp_layer: int = 1,
                 graph_construction_model: Optional[Any] = None,
                 verbose: int = 0,
                 train_chi_id: Optional[Any] = None):
        super().__init__(sigma_embedding,
                         model,
                         torsion_mlp_hidden_dims,
                         schedule_1pi_periodic,
                         schedule_2pi_periodic,
                         graph_construction_model,
                         verbose,
                         train_chi_id)
        self.confidence_model = confidence_model
        self.num_sample = num_sample
        self.mlp = layers.MLP(self.confidence_model.output_dim,
                              [self.confidence_model.output_dim] * num_mlp_layer + [1])

    def predict_rmsd(self, batch, all_loss=None, metric=None):
        protein = batch['graph']
        if self.graph_construction_model:
            protein = self.graph_construction_model(protein)
        atom_feature = self.confidence_model(protein, protein.node_feature.float())["node_feature"]
        residue_feature = scatter_mean(atom_feature, protein.atom2residue, dim=0,
                                       dim_size=protein.num_residue)  # [num_residue, feature_dim]
        pred = self.mlp(residue_feature).squeeze(-1)  # [num_residue]
        return pred

    @torch.no_grad()
    def generate(self, batch, randomize=True):
        protein = batch['graph']
        if randomize:
            protein = rotamer.randomize(protein)

        best_protein = protein.clone()
        best_rmsd = torch.zeros(protein.num_residue, device=self.device) + 1e6
        for _ in tqdm(range(self.num_sample), desc="Confidence sampling"):
            batch = super().generate(batch, randomize=True)  # TODO: do we need to randomize?
            protein = batch['graph']
            rmsd = self.predict_rmsd(batch)
            residue_update_mask = rmsd < best_rmsd  # [num_residue]
            atom_update_mask = residue_update_mask[protein.atom2residue]  # [num_atom]
            best_protein.node_position[atom_update_mask] = protein.node_position[atom_update_mask]
            best_rmsd[residue_update_mask] = rmsd[residue_update_mask]

        best_batch = {
            "graph": best_protein,
            "rmsd": best_rmsd
        }
        return best_batch
