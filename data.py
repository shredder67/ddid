from typing import Callable

import torch
import numpy as np
from sklearn.datasets import make_moons, make_checkerboard
from torch.utils.data import Dataset, DataLoader


def normalize(points):
    mean = points.mean()
    std = points.std()
    return (points - mean) / std


def sample_moons(n):
    return make_moons(n_samples=(n, n), shuffle=False, noise=0.05)


def sample_checkerboard(n):
    x1 = np.random.rand(n) * 4 - 2
    x1 = np.sort(x1)[::-1]
    x2 = np.random.rand(n) - np.random.randint(0, 2, n) * 2
    x2 = x2 + (np.floor(x1) % 2)
    return np.stack([x1, x2], axis=1), np.array([0]*len(x1))


def sample_conc_rings(n, radiuses):
    circ_data = []
    r = radiuses
    for i in range(3):
        theta = np.random.rand(n) * 2 * np.pi
        theta = np.sort(theta, 0).reshape(-1)[::-1]
        r_eps1, r_eps2 = (np.random.rand(n) - 0.5) * 0.5, (np.random.rand(n) - 0.5) * 0.5
        circ_data.append(np.vstack([np.cos(theta)*(r[i] + r_eps1), np.sin(theta)*(r[i] + r_eps2)])) # outer ring (x, y)
    
    # theta = np.random.rand(n // len(radiuses)) * 2 * np.pi
    # theta = np.sort(theta, 1).reshape(-1)[::-1]
    # r_eps1, r_eps2 = (np.random.rand(n) - 0.5) * 0.5, (np.random.rand(n) - 0.5) * 0.5
    # circ_data.append(np.vstack([(np.cos(ls)*(r[1] + r_eps1), np.sin(ls)*(r[1] + r_eps2))])) # mid ring
    # r_eps1, r_eps2 = (np.random.rand(n) - 0.5) * 0.5, (np.random.rand(n) - 0.5) * 0.5
    # circ_data.append(np.vstack([(np.cos(ls)*(r[2] + r_eps1), np.sin(ls)*(r[2] + r_eps2))])) # inner ring

    data = np.hstack(circ_data).T
    ids = np.hstack([
        np.zeros(n),
        np.ones(n),
        np.ones(n)*2
    ])
    assert data.shape == (n*3, 2)
    assert ids.shape == (n*3,)
    return data, ids


def sample_conc_squares(n, radiuses):
        # Generate numbers for the four edges
        K = 3
        radii = radiuses
        if isinstance(radii, list):
            radii = np.array(radii)
        assert n % (4 * K) == 0
        rand = lambda c: 2 * np.sort(np.random.rand(c) - 0.5)  # Increasing in [-1, 1]

        px = list()
        py = list()
        indices = list()

        for i in range(K):
            C = n // (K * 4)
            px0 = rand(C)  # Up
            py0 = np.ones(C)
            px1 = np.ones(C)  # Right
            py1 = rand(C)[::-1]
            px2 = rand(C)[::-1]  # Down
            py2 = -np.ones(C)
            px3 = -np.ones(C)  # Left
            py3 = rand(C)
            indices.append(np.ones(C * 4) * i)
            px += [px0, px1, px2, px3]
            py += [py0, py1, py2, py3]
        px = np.concatenate(px)
        py = np.concatenate(py)
        indices = np.concatenate(indices).astype(int)

        # Then, assign the points randomly to the squares
        radii_eps = (np.random.rand(n) - .5) * 0.5
        radii = radii[indices] + radii_eps

        px, py = px * radii, py * radii
        points = np.stack([px, py], axis=1)

        return points, indices


def sample_parallel_rings(n, r, centers):
    circ_data = []
    ls = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r_eps1, r_eps2 = (np.random.rand(n) - 0.5) * 0.5, (np.random.rand(n) - 0.5) * 0.5
    circ_data.append(np.vstack([np.cos(ls)*(r[0] + r_eps1) + centers[0],
        np.sin(ls)*(r[0] + r_eps2)]))
    r_eps1, r_eps2 = (np.random.rand(n) - 0.5) * 0.5, (np.random.rand(n) - 0.5) * 0.5
    circ_data.append(np.vstack([(np.cos(ls)*(r[1] + r_eps1) + centers[1],
     np.sin(ls)*(r[1] + r_eps2))]))
    r_eps1, r_eps2 = (np.random.rand(n) - 0.5) * 0.5, (np.random.rand(n) - 0.5) * 0.5
    circ_data.append(np.vstack([(np.cos(ls)*(r[2] + r_eps1) + centers[2],
     np.sin(ls)*(r[2] + r_eps2))]))

    data = np.hstack(circ_data).T
    ids = np.hstack([
        np.zeros(n),
        np.ones(n),
        np.ones(n)*2
    ])
    assert data.shape == (n*3, 2)
    assert ids.shape == (n*3,)
    return data, ids


def sample_parallel_squares(n, r, centers):
        # Generate numbers for the four edges
        K = 3
        radii = r
        if isinstance(radii, list):
            radii = np.array(radii)
        if isinstance(centers, list):
            centers = np.array(centers)
        assert n % (4 * K) == 0
        rand = lambda c: 2 * np.sort(np.random.rand(c) - 0.5)  # Increasing in [-1, 1]

        px = list()
        py = list()
        indices = list()

        for i in range(K):
            C = n // (K * 4)
            px0 = rand(C)  # Up
            py0 = np.ones(C)
            px1 = np.ones(C)  # Right
            py1 = rand(C)[::-1]
            px2 = rand(C)[::-1]  # Down
            py2 = -np.ones(C)
            px3 = -np.ones(C)  # Left
            py3 = rand(C)
            indices.append(np.ones(C * 4) * i)
            px += [px0, px1, px2, px3]
            py += [py0, py1, py2, py3]
        px = np.concatenate(px)
        py = np.concatenate(py)
        indices = np.concatenate(indices).astype(int)

        # Then, assign the points randomly to the squares
        radii_eps = (np.random.rand(n) - .5) * 0.5
        radii = radii[indices] + radii_eps

        px, py = px * radii, py * radii
        points = np.stack([px, py], axis=1)

        points[:, 0] += centers[indices]
        return points, indices


TYPE_TO_SAMPLER = {
    "M": sample_moons,
    "CB": sample_checkerboard,
    "CR": sample_conc_rings,
    "CS": sample_conc_squares,
    "PR": sample_parallel_rings,
    "PS": sample_parallel_squares
}

class PointsDataset(Dataset):
    def __init__(self, n, ds_type: str, device):
        if ds_type not in TYPE_TO_SAMPLER.keys():
            raise ValueError(f"Unrecognized dataset type: {ds_type}, has to be one of {list(TYPE_TO_SAMPLER.keys())}")

        self.n = n
        self.device = device
        
        r_diff = [1, 2, 3]
        r_same = [1, 1, 1]
        centers = [-2.3, 0., 2.3]
        d = 0.2
        
        if ds_type in ["M", "CB"]:
            points, ids = TYPE_TO_SAMPLER[ds_type](n)
        if ds_type in ["CR", "CS"]:
            points, ids = TYPE_TO_SAMPLER[ds_type](n, r_diff)
        if ds_type in ["PR", "PS"]:
            points, ids = TYPE_TO_SAMPLER[ds_type](n, r_same, centers)

        points = normalize(points)
        self.points = torch.from_numpy(points).to(dtype=torch.float32).to(device)
        self.ids = ids # used for coloring

    def __len__(self):
        return self.n


    def __getitem__(self, idx):
        x = self.points[idx]
        y = self.ids[idx]
        return x, y 


    def get_dataloader(self, bs, shuffle):
        return DataLoader(self, batch_size=bs, shuffle=shuffle)