import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class SinusoidalEmbedding(nn.Module):
    def __init__(self, size: int, scale: float = 1.0):
        super().__init__()
        self.size = size
        self.scale = scale

    def forward(self, x: torch.Tensor):
        x = x * self.scale
        half_size = self.size // 2
        emb = torch.log(torch.Tensor([10000.0])).to(x.device) / (half_size - 1)
        emb = torch.exp(-emb * torch.arange(half_size).to(x.device))
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=-1)
        return emb

    def __len__(self):
        return self.size


class ResBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()

        self.ff = nn.Linear(dim, dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor):
        return x + self.act(self.ff(x))


class ScoreModel2D(nn.Module):
    def __init__(self, hidden_dim=128, num_hidden=3, embedding_dim=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        if embedding_dim is None:
            embedding_dim = hidden_dim
        self.embedding_dim = embedding_dim

        self.t_embed = SinusoidalEmbedding(hidden_dim)
        # taken from https://bmild.github.io/fourfeat - we encode x,y positions into freq space
        self.pos_embed_x = SinusoidalEmbedding(hidden_dim, 10.)
        self.pos_embed_y = SinusoidalEmbedding(hidden_dim, 10.)

        layers = [nn.Linear(embedding_dim*3, hidden_dim), nn.GELU()]
        for _ in range(num_hidden):
            layers.append(ResBlock(hidden_dim))
        layers.append(nn.Linear(hidden_dim, 2))
        self.mlp = nn.Sequential(*layers)
    

    def forward(self, x, t):
        e1 = self.t_embed(t)
        e2 = self.pos_embed_x(x[:, 0])
        e3 = self.pos_embed_y(x[:, 1])
        return self.mlp(torch.cat((e1, e2, e3), dim=-1))


class Scheduler:
    """only stuff for training, following ddpm https://arxiv.org/pdf/2006.11239"""
    def __init__(
        self, 
        n_timesteps=1000, 
        beta_start=1e-3, 
        beta_end=1e-2,
        device=torch.device("cuda")
    ):
        self.device=device
        self.n_timesteps = n_timesteps
        self.betas = torch.linspace( # linear scheduling
            beta_start, beta_end, n_timesteps, dtype=torch.float32, device=device
        )
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0) # a_t
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.) # a_t-1

        # stuff for sampling x_t from forward process
        self.a_cp_sqrt = self.alphas_cumprod ** 0.5
        self.one_minus_a_cp_sqrt = (1 - self.alphas_cumprod)**0.5

        # stuff for posterior
        self.a_cp_prev_sqrt = self.alphas_cumprod_prev**0.5
        

    def q_posterior_mean(self, x_0, x_t, t):
        """eq. 7, needed for sampling, but we don't do it here"""
        a = self.a_cp_prev_sqrt[t] * self.betas[t] / (1 - self.alphas_cumprod[t])
        b = torch.sqrt(self.alphas[t]) * (1 - self.alphas_cumprod_prev[t]) / (1 - self.alphas_cumprod[t])
        return a[:,None]*x_0 + b[:,None]*x_t 

    
    def q_posterior_var(self, x_0, x_t, t):
        if t == 0: return torch.zeros_like(x_0, device=x_0)
        return self.betas[t] * (1 - self.alphas_cumprod_prev[t]) / (1 - self.alphas_cumprod[t])

    @torch.no_grad()
    def step(self, x_t, model_output, t): # reverse-sde step, alg. 2
        pred_x_0 = torch.sqrt(1 / self.a_cp_sqrt[t])[:, None] * x_t \
            - (torch.sqrt(self.alphas[t])*(1 - self.alphas_cumprod_prev[t]) / (1. - self.alphas_cumprod[t]))[:, None] * model_output

        pred_x_prev = self.q_posterior_mean(pred_x_0, x_t, t)

        if torch.all(t > 0.): # add stochastic term
            sigma_t_sq = self.betas[t] * (1. - self.alphas_cumprod_prev[t]) / (1. - self.alphas_cumprod[t])
            sigma_t_sq = sigma_t_sq.clip(1e-20)
            pred_x_prev += torch.randn_like(pred_x_prev) * (sigma_t_sq**0.5)[:, None]

        return pred_x_prev 

    @torch.no_grad()
    def sample_loop(self, score_model, n, noise=None, cache_every=50): # full num_steps sde sampling
        timesteps = list(range(self.n_timesteps))[::-1] # t-steps from T to 0
        if noise is None:
            noise = torch.randn(n, 2, device=self.device, requires_grad=False)
        sample = noise
        samples=[]
        for i, t in enumerate(tqdm(timesteps)):
            t = torch.tensor([t], dtype=torch.long, device=self.device).expand(n)
            score_pred = score_model(sample, t)
            sample = self.step(sample, score_pred, t)
            if (i + 1) % cache_every == 0:
                samples.append(sample)
        return sample, samples


    def fwd_sample(self, x_0: torch.Tensor, t: int, noise: torch.Tensor = None):
        """eq. 4"""
        if noise is None:
            noise = torch.randn_like(x_0, device=x_0.device)
        return self.a_cp_sqrt[t][:,None] * x_0 + (1. - self.a_cp_sqrt[t])[:, None] * noise