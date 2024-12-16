import torch
import numpy as np
from tqdm import tqdm


class DDIMSolver:
    """eta == 0., as one would expect :)"""
    def __init__(
        self,
        n_train_timesteps=1000,
        n_inference_steps=50, 
        beta_start=1e-3, 
        beta_end=1e-2,
        eta=0.,
        device=torch.device("cuda"),
    ):
        self.device=device
        self.n_steps = n_inference_steps
        self.n_timesteps = n_train_timesteps
        self.eta = eta
        self.betas = torch.linspace( # linear scheduling
            beta_start, beta_end, n_train_timesteps, dtype=torch.float32, device=device
        )
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0) # a_t


    @torch.no_grad()
    def step(self, x_t, model_output, t):
        # compute alphas, betas
        prev_t = t - self.n_timesteps // self.n_steps
        alpha_prod_t = self.alphas_cumprod[t][:, None]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t][:, None] if torch.all(prev_t > 0 ) else 1.0
        beta_prod_t = 1 - alpha_prod_t

        # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_original_sample = (x_t - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        pred_epsilon = model_output
                
        # compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_sample_direction = (1 - alpha_prod_t_prev) ** (0.5) * pred_epsilon

        # compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction

        return prev_sample

    @torch.no_grad()
    def sample_loop(self, score_model, n=1, shape=(2,), noise=None):
        if noise is None:
            size = [n, *shape]
            noise = torch.randn(*size, dtype=torch.float32, device=self.device, requires_grad=False)
        sample = noise
        bs = sample.shape[0]
        timesteps = (
            np.linspace(0, self.n_timesteps - 1, self.n_steps + 1).round()[::-1][:-1].copy().astype(np.int64)
        )
        timesteps = torch.from_numpy(timesteps).to(self.device)
        for step_id in tqdm(range(self.n_steps)):
            t = timesteps[step_id].expand(bs)
            score_pred = score_model(sample, t)
            sample = self.step(sample, score_pred, t)
        return sample 
        
    def inverse_step(self, x_t, model_output, t):
               # compute alphas, betas
        prev_t = t + self.n_timesteps // self.n_steps
        alpha_prod_t = self.alphas_cumprod[t][:, None]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t][:, None] if torch.all(prev_t < self.n_timesteps) else alpha_prod_t[-1]
        beta_prod_t = 1 - alpha_prod_t

        # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_original_sample = (x_t - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        pred_epsilon = model_output
                
        # compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_sample_direction = (1 - alpha_prod_t_prev) ** (0.5) * pred_epsilon

        # compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction

        return prev_sample

    def inverse_loop(self, score_model, x_0):
        latent = x_0
        bs = latent.shape[0]
        timesteps = (
            np.linspace(0, self.n_timesteps - 1, self.n_steps + 1).round()[:-1].copy().astype(np.int64)
        )
        timesteps = torch.from_numpy(timesteps).to(self.device)
        for step_id in tqdm(range(self.n_steps)):
            t = timesteps[step_id].expand(bs)
            score_pred = score_model(latent, t)
            latent = self.inverse_step(latent, score_pred, t)
        return latent 


