import argparse
import random
import os
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from diff import ScoreModel2D, Scheduler
from data import PointsDataset
from ema import EMA


SEED=42
random.seed(SEED)
np.random.seed(SEED)
torch.random.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_diffusion(
    ds_size=30_000,
    ds_type="M",
    t_steps=1000,
    hidden_dim=256,
    num_hidden=3,
    batch_size=1000,
    epochs=1000,
    save_every=10
):
    # 1. set up dataloader, model, optimizer, ema, paths
    dataset = PointsDataset(ds_size, ds_type, DEVICE)
    loader = dataset.get_dataloader(batch_size, shuffle=True)

    scheduler = Scheduler(n_timesteps=t_steps, device=DEVICE)
    model = ScoreModel2D(hidden_dim=hidden_dim, num_hidden=num_hidden).to(DEVICE)
    ema = EMA(
        model,
        beta = 0.9999,
        update_after_step = 5*(ds_size // batch_size),
        update_every = ds_size // batch_size,          
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.)
    loss_fn = F.mse_loss

    # 2. training loop (save every iter, remember loss)
    best_loss = torch.inf
    best_loss_id = -1
    loss_hist = []
    tmp_ckpts = []
    for e in tqdm(range(epochs)):
        e_loss = 0.
        for x, _ in loader: # alg 1. from ddpm paper
            optimizer.zero_grad()
            eps_targ = torch.randn_like(x, device=DEVICE)
            t = torch.randint(0, scheduler.n_timesteps, (batch_size, ), device=DEVICE).long()
            x_t = scheduler.fwd_sample(x, t, eps_targ)
            eps_pred = model(x_t, t)
            loss = loss_fn(eps_pred, eps_targ)
            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), 1.)
            optimizer.step()
            ema.update()
            e_loss += loss.cpu().item()
        e_loss /= len(loader)

        if (e + 1) % save_every == 0:
            torch.save(model.state_dict(), f"models/{e+1}_SM_{ds_type}.pth")
            tmp_ckpts.append(f"models/{e+1}_SM_{ds_type}.pth")
            if e_loss < best_loss:
                best_loss = e_loss
                best_loss_id = e + 1
        loss_hist.append(e_loss)

    # 3. print log _best model, save it again, save loss_hist
    print(f"best loss was at {best_loss_id} epoch")
    torch.save(
        torch.load(f"models/{best_loss_id}_SM_{ds_type}.pth"),
        f"best_ckpts/{ds_type}_SM_{best_loss_id}.pth"
    )
    torch.save(
        ema.ema_model.state_dict(),
        f"best_ckpts/{ds_type}_SM_ema.pth"
    )
    with open(f"best_ckpts/{ds_type}_SM_loss_hist.pkl", "wb") as f:
        pickle.dump(loss_hist, f)
    
    # clean tmp checkpoints
    for f in tmp_ckpts:
        os.remove(f)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-ds", "--dstype", type=str, default="M")

    args = parser.parse_args()

    train_diffusion(ds_type=args.dstype)