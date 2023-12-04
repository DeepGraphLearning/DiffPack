from abc import abstractmethod

import numpy as np
import os
import torch
import torch.nn as nn
from torch_scatter import scatter_add
from torchdrug import core

from torchdrug.core import Registry as R
import tqdm


def p(x, sigma, N=10, PI=np.pi):
    p_ = 0
    for i in tqdm.trange(-N, N + 1):
        p_ += np.exp(-(x + 2 * PI * i) ** 2 / 2 / sigma ** 2)
    return p_


def grad(x, sigma, N=10, PI=np.pi):
    p_ = 0
    for i in tqdm.trange(-N, N + 1):
        p_ += (x + 2 * PI * i) / sigma ** 2 * np.exp(-(x + 2 * PI * i) ** 2 / 2 / sigma ** 2)
    return p_


def sample(sigma, PI=np.pi):
    out = sigma * np.random.randn(*sigma.shape)
    out = (out + PI) % (2 * PI) - PI
    return out


class SO2Schedule(nn.Module, core.Configurable):
    X_MIN, X_N = 1e-5, 5000
    SIGMA_MIN, SIGMA_MAX, SIGMA_N = 3e-3, 2, 5000

    def __init__(self, PI, cache_folder):
        super().__init__()
        self.PI = PI
        self.cache_folder = os.path.expanduser(cache_folder) if cache_folder is not None \
            else os.path.join(os.path.dirname(__file__), "cache")
        self.x = 10 ** np.linspace(np.log10(self.X_MIN), 0,
                                   self.X_N + 1) * PI
        self.sigma = 10 ** np.linspace(np.log10(self.SIGMA_MIN), np.log10(self.SIGMA_MAX),
                                       self.SIGMA_N + 1) * PI

        os.makedirs(self.cache_folder, exist_ok=True)
        self.p_table_path = os.path.join(self.cache_folder, f'Periodic.{PI:.3f}.p.npy')
        self.score_table_path = os.path.join(self.cache_folder, f'Periodic.{PI:.3f}.score.npy')
        if os.path.exists(self.p_table_path):
            self.p_ = np.load(self.p_table_path)
            self.score_ = np.load(self.score_table_path)
        else:
            self.p_ = p(self.x, self.sigma[:, None], N=100, PI=PI)
            self.score_ = grad(self.x, self.sigma[:, None], N=100, PI=PI) / self.p_
            np.save(self.p_table_path, self.p_)
            np.save(self.score_table_path, self.score_)

        # Precompute the normalization constant
        score_norm_table = self.score(
            sample(self.sigma[None].repeat(10000, 0).flatten(), PI=PI),
            (self.sigma[None].repeat(10000, 0).flatten()),
        ).reshape(10000, -1)

        self.score_norm_ = (score_norm_table ** 2).mean(0)

    def score(self, x, sigma):
        x = (x + self.PI) % (2 * self.PI) - self.PI  # range from -pi to pi
        sign = np.sign(x)
        x = np.log(np.abs(x) / self.PI + 1e-10)
        x = (x - np.log(self.X_MIN)) / (0 - np.log(self.X_MIN)) * self.X_N
        x = np.round(np.clip(x, 0, self.X_N)).astype(int)
        sigma = np.log(sigma / self.PI)
        sigma = (sigma - np.log(self.SIGMA_MIN)) / (np.log(self.SIGMA_MAX) - np.log(self.SIGMA_MIN)) * self.SIGMA_N
        sigma = np.round(np.clip(sigma, 0, self.SIGMA_N)).astype(int)
        return -sign * self.score_[sigma, x]


    def p(self, x, sigma):
        x = (x + self.PI) % (2 * self.PI) - self.PI
        x = np.log(np.abs(x) / self.PI + 1e-10)
        x = (x - np.log(self.X_MIN)) / (0 - np.log(self.X_MIN)) * self.X_N
        x = np.round(np.clip(x, 0, self.X_N)).astype(int)
        sigma = np.log(sigma / self.PI)
        sigma = (sigma - np.log(self.SIGMA_MIN)) / (np.log(self.SIGMA_MAX) - np.log(self.SIGMA_MIN)) * self.SIGMA_N
        sigma = np.round(np.clip(sigma, 0, self.SIGMA_N)).astype(int)
        return self.p_[sigma, x]

    def score_norm(self, sigma):
        if type(sigma) == torch.Tensor:
            sigma = sigma.cpu().numpy()
        sigma = np.log(sigma / self.PI)
        sigma = (sigma - np.log(self.SIGMA_MIN)) / (np.log(self.SIGMA_MAX) - np.log(self.SIGMA_MIN)) * self.SIGMA_N
        sigma = np.round(np.clip(sigma, 0, self.SIGMA_N)).astype(int)
        return self.score_norm_[sigma]

    # def score_norm_torch(self, sigma):
    #     sigma = torch.log(sigma / self.PI)
    #     sigma = (sigma - torch.log(torch.tensor(self.SIGMA_MIN))) / (torch.log(torch.tensor(self.SIGMA_MAX)) - torch.log(torch.tensor(self.SIGMA_MIN))) * self.SIGMA_N
    #     sigma = torch.round(torch.clip(sigma, 0, self.SIGMA_N)).long()
    #     return torch.tensor(self.score_norm_[sigma.cpu().numpy()], device=sigma.device)

    @abstractmethod
    def add_noise(self, x, t, x_mask=None):
        """Add noise to the input tensor (torsion angles).

        Args:
            x (Tensor): input tensor
            t (Tensor): timesteps tensor
            x_mask (Tensor): input mask tensor

        Returns:
            Tensor: noisy input tensor
        """
        pass

    @abstractmethod
    def step(self, x, x_score, t, dt, x_mask=None):
        """Update the input tensor (torsion angles).

        Args:
            x (Tensor): input tensor
            x_score (Tensor): input tensor's score
            t (Tensor): timesteps tensor
            dt (Tensor): timestep size tensor
            x_mask (Tensor): input mask tensor

        Returns:
            Tensor: updated input tensor
        """
        pass

    @abstractmethod
    def sample_train_t(self, shape):
        """Sample timesteps for training.

        Args:
            shape (tuple): shape of the output tensor

        Returns:
            Tensor: sampled timesteps
        """
        pass

    @property
    @abstractmethod
    def reverse_t_schedule(self):
        """Reverse timesteps schedule."""
        pass



@R.register('SO2VESchedule')
class SO2VESchedule(SO2Schedule, core.Configurable):
    def __init__(self, pi_periodic=False, cache_folder=None, sigma_min=0.01 * np.pi, sigma_max=np.pi, annealed_temp=3, mode="sde"):
        """
        Args:
            sigma_min (float): minimum standard deviation
            sigma_max (float): maximum standard deviation
        """
        PI = 1/2 * np.pi if pi_periodic else np.pi  # TODO: remove ambiguity
        super().__init__(PI, cache_folder)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_min_log = np.log(sigma_min)
        self.sigma_max_log = np.log(sigma_max)
        self.annealed_temp = annealed_temp
        self.mode = mode

    def t_to_sigma(self, t):
        """Transfer timesteps to standard deviation.

        Args:
            t (Tensor): timesteps ranging from 0 to 1

        Returns:
            Tensor: standard deviation ranging from sigma_min to sigma_max
        """
        return torch.exp(self.sigma_min_log + (self.sigma_max_log - self.sigma_min_log) * t)

    @torch.no_grad()
    def add_noise(self, x, t, x_mask=None):
        """Add noise to the input tensor (torsion angles).

        Args:
            x (Tensor): Torsion angles of shape :math:`(num_res, 4)`
            x_mask (Tensor): Mask of shape :math:`(num_res, 4)`
            t (Tensor): Timesteps of shape :math:`(num_res)`
        """
        sigmas = self.t_to_sigma(t).unsqueeze(-1)
        noise = torch.randn_like(x) * sigmas
        score = torch.tensor(
            self.score(noise.cpu().numpy(), sigmas.cpu().numpy()), device=x.device, dtype=x.dtype
        )

        if x_mask is not None:
            noise *= x_mask
            score *= x_mask

        x = x + noise
        return x, score

    @torch.no_grad()
    def step(self, x, x_score, t, dt, x_mask=None):
        """Denoise step for the input tensor (torsion angles).

        Args:
            x (Tensor): Torsion angles of shape :math:`(num_res, 4)`
            x_score (Tensor): Score of shape :math:`(num_res, 4)`
            t (Tensor): Timesteps of shape :math:`(num_res)`
            dt (float): Step size of shape :math:`(num_res)`
            x_mask (Tensor): Mask of shape :math:`(num_res, 4)`

        Returns:
            Tensor: Denoised torsion angles of shape :math:`(num_res, 4)`
        """
        sigma = self.t_to_sigma(t)  # (num_res)
        g = sigma * np.sqrt(2 * np.log(self.sigma_max / self.sigma_min))  # (num_res)

        # Temperature Coefficient
        alpha = 1 - (sigma / np.exp(self.sigma_max_log)) ** 2 if self.annealed_temp else None
        annealed_weight = self.annealed_temp / (alpha + (1 - alpha) * self.annealed_temp) if self.annealed_temp else 1

        # Noise
        x_prev = x.clone()
        if self.mode == "ode":
            drift = (0.5 * g ** 2 * dt * (x_score * annealed_weight))
            x_prev += drift
        elif self.mode == "sde":
            noise = torch.normal(mean=0, std=1, size=x_score.shape, device=x_score.device)
            drift = g ** 2 * dt * (x_score * annealed_weight)
            diffusion = g * torch.sqrt(dt) * noise
            x_prev += (drift + diffusion)
        else:
            raise NotImplementedError

        if x_mask is not None:
            x_prev[~x_mask] = x[~x_mask]

        return x_prev

    @torch.no_grad()
    def step_correct(self, x, x_score, x_batch, x_mask=None, snr=0.16):
        """Correct step for the input tensor (torsion angles).

        Args:
            x (Tensor): Torsion angles of shape :math:`(num_res, 4)`
            x_score (Tensor): Score of shape :math:`(num_res, 4)`
            x_batch (Tensor): Batch of shape :math:`(num_res)`
            x_mask (Tensor): Mask of shape :math:`(num_res, 4)`
            snr (float): Signal to noise ratio

        Returns:
            Tensor: Corrected torsion angles of shape :math:`(num_res, 4)`
        """
        x_batch = x_batch.reshape(-1, 4)

        # Calculate Score Norm
        x_score_2 = x_score ** 2
        score_norm = torch.sqrt(scatter_add(x_score_2[x_mask], x_batch[x_mask], dim=0)).mean()

        # Calculate Noise Norm
        noise = torch.randn_like(x_score)
        noise_2 = noise ** 2
        noise_norm = torch.sqrt(scatter_add(noise_2[x_mask], x_batch[x_mask], dim=0)).mean()

        # Step Size
        step_size = (snr * noise_norm / score_norm) ** 2 * 2

        # Correct Step
        x_prev = x.clone()
        x_prev += step_size * x_score
        x_prev += ((step_size * 2) ** 0.5) * noise

        if x_mask is not None:
            x_prev[~x_mask] = x[~x_mask]

        return x_prev

    def sample_train_t(self, shape):
        """Sample timesteps from uniform distribution.

        Args:
            shape (tuple): shape of the timesteps

        Returns:
            Tensor: timesteps of shape :math:`(shape)`
        """
        return torch.rand(shape)

    @property
    def reverse_t_schedule(self):
        return torch.linspace(1, 0, 11)






