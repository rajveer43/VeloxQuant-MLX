"""Test the paper's Observations 3.1/3.2 on REAL trained weights."""

import sys

import mlx_lm
import numpy as np

from veloxquant_mlx.quantizers.qfilters_calibration import (
    collect_query_activations,
    compute_qfilters,
)

mid = sys.argv[1]
m, t = mlx_lm.load(mid)
texts = [
    "The capital of France is Paris, a city known for its art, cuisine and long history. " * 40,
    "Machine learning models require large amounts of data and compute to train effectively. " * 40,
    "In 1969, humans first walked on the surface of the moon during the Apollo 11 mission. " * 40,
    "Photosynthesis converts light energy into chemical energy stored in glucose molecules. " * 40,
]
acts = collect_query_activations(m, t, texts, max_length=512, max_samples_per_head=3000)
print(f"\n=== {mid} : {len(acts)} layers, Q shape {tuple(acts[0].shape)} ===")

# Obs 3.1: E<Q_i, u^h> > 0 for the top direction.
# Obs 3.2: projections on all OTHER SVD components have ~zero mean.
r1, r2, energy = [], [], []
for a in acts:
    q = np.array(a, dtype=np.float64)  # [H, N, D]
    f = np.array(compute_qfilters(a), dtype=np.float64)  # [H, D]
    for h in range(q.shape[0]):
        X = q[h]
        proj1 = X @ f[h]
        r1.append(proj1.mean())
        # full SVD basis to check the OTHER components
        _, S, Vt = np.linalg.svd(X, full_matrices=False)
        means = np.abs(X @ Vt.T).mean(axis=0)  # |E<Q,v_m>| per component
        r2.append(means[0] / (means[1:].mean() + 1e-12))
        energy.append((S[0] ** 2) / (S**2).sum())

r1, r2, energy = np.array(r1), np.array(r2), np.array(energy)
print(f"Obs 3.1  E<Q,u> > 0 : {100 * np.mean(r1 > 0):.1f}% of heads   (median {np.median(r1):.3f})")
print(f"Obs 3.2  |E<Q,v1>| / mean|E<Q,vm>|, m>=2 : median {np.median(r2):.1f}x")
print(f"Top-component energy share : median {100 * np.median(energy):.1f}%")
