"""SoftImpute and batch Sinkhorn imputation: our own implementations, written from the papers.

Both take a float array with NaN gaps (the standardised model input) and return (filled array, FitInfo).
Visible cells are never changed.

- SoftImpute (Mazumder, Hastie & Tibshirani, JMLR 2010, Algorithm 1): fill the gaps, take a soft-thresholded SVD
  (every singular value shrunk by lambda, the small ones to zero), put the low-rank values back into the gaps, and
  repeat until the filled values settle. lambda sets how many dimensions survive, so it is tuned by hide-and-check.
- Batch Sinkhorn imputation (Muzellec, Josse, Boyer & Cuturi, ICML 2020, Algorithm 1): the missing values are free
  parameters. Each step draws two random batches of rows and moves the missing values in them to reduce the Sinkhorn
  divergence between the batches, so that any two batches of the filled table look like samples of one
  distribution. The paper's fixed settings are used, with no per-dataset tuning.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

# Relative lambdas tried by tuning: lambda = fraction x the largest singular value of the zero-filled input.
SOFT_GRID = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5)
SOFT_TOL = 1e-5           # stop when the filled values change by less than this (relative squared change)
SOFT_MAX_ITER = 500

# Muzellec et al. (2020), Section 4: batch 128, RMSprop with step 0.01, start at the column mean plus N(0, 0.1) noise,
# entropic regularisation = 5% of the median distance between rows of the mean-imputed table.
SINKHORN_BATCH = 128
SINKHORN_LR = 1e-2
SINKHORN_NOISE = 0.1
SINKHORN_EPS_SHARE = 0.05
# Iteration budget, as the number of times each row is drawn into a batch (iterations x 2 x batch / rows), so that
# tables of any size get the same amount of optimisation. Chosen once by hide-and-check on the quick tables: from
# about 500 visits every table was within 0.01 of its best check score (README, "Choosing the budget").
SINKHORN_VISITS = 500


@dataclass
class FitInfo:
    seconds: float
    history: list = field(default_factory=list)      # the training curve, one dict per recorded step
    details: dict = field(default_factory=dict)


def _check(X) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or not np.isnan(X).any():
        raise ValueError('Expected a 2-D table with at least one missing value')
    if (~np.isnan(X)).sum(axis=0).min() < 2:
        raise ValueError('Every column needs at least two visible values')
    return X


# ---- SoftImpute ----

def largest_singular_value(X: np.ndarray) -> float:
    return float(np.linalg.svd(np.where(np.isnan(X), 0.0, X), compute_uv=False)[0])


def soft_impute(X, lam: float, start: np.ndarray | None = None, tol: float = SOFT_TOL,
                max_iter: int = SOFT_MAX_ITER) -> tuple[np.ndarray, FitInfo]:
    """SoftImpute with an absolute lambda. `start` warm-starts from another fill (used along the tuning path)."""
    X = _check(X)
    began = time.perf_counter()
    gaps = np.isnan(X)
    Z = np.where(gaps, 0.0 if start is None else start, X)     # columns are standardised, so 0 is the visible mean
    history, rank = [], 0
    for iteration in range(1, max_iter + 1):
        U, s, Vt = np.linalg.svd(Z, full_matrices=False)
        shrunk = np.maximum(s - lam, 0.0)
        rank = int((shrunk > 0).sum())
        new = np.where(gaps, (U[:, :rank] * shrunk[:rank]) @ Vt[:rank], X)
        # Change of the filled values relative to their size (visible cells never change, so they are left out;
        # including them would make every change look tiny and stop the loop too early).
        change = float(np.sum((new[gaps] - Z[gaps]) ** 2) / max(np.sum(Z[gaps] ** 2), 1e-12))
        Z = new
        history.append({'iteration': iteration, 'change': change, 'rank': rank})
        if change < tol:
            break
    return Z, FitInfo(time.perf_counter() - began, history,
                      {'lambda': lam, 'rank': rank, 'iterations': len(history), 'converged': history[-1]['change'] < tol})


def soft_impute_path(X, fractions=SOFT_GRID, **kwargs) -> list[tuple[float, np.ndarray, FitInfo]]:
    """Fit every relative lambda, largest first, each warm-started from the previous fit (the paper's Algorithm 1)."""
    X = _check(X)
    top = largest_singular_value(X)
    fits, start = [], None
    for fraction in sorted(fractions, reverse=True):
        filled, info = soft_impute(X, fraction * top, start=start, **kwargs)
        info.details['fraction'] = fraction
        fits.append((fraction, filled, info))
        start = filled
    return fits


def impute_softimpute(X_missing, seed: int, fraction: float, grid=SOFT_GRID) -> tuple[np.ndarray, FitInfo]:
    """Fill the table with lambda = fraction x its largest singular value.

    It follows the same warm-started path as tuning (every grid value above `fraction`, largest first, then
    `fraction`), so the final fit is computed exactly as the tuned candidate was. The seed is unused (SoftImpute is
    deterministic); it is accepted so both methods have the same signature.
    """
    X = _check(X_missing)
    began = time.perf_counter()
    fits = soft_impute_path(X, sorted({*(f for f in grid if f > fraction), fraction}))
    _, filled, info = fits[-1]
    info.details['path'] = [{'fraction': f, 'rank': i.details['rank'], 'iterations': i.details['iterations']}
                            for f, _, i in fits]
    info.seconds = time.perf_counter() - began
    return filled, info


# ---- Batch Sinkhorn imputation ----

def _entropic_ot(x, y, eps: float, scaling: float = 0.9, extra: int = 10):
    """Entropic optimal transport between two equally weighted point clouds, cost |x - y|^2 / 2.

    Log-domain Sinkhorn iterations with epsilon-scaling (start at the largest cost, shrink by `scaling` per step
    down to eps, then `extra` steps at eps). The potentials are found without gradients and one last update with
    gradients gives the gradient of the optimal value (envelope theorem), as in the geomloss library used by the
    authors' code (Feydy et al., AISTATS 2019).
    """
    import torch
    C = 0.5 * torch.cdist(x, y) ** 2
    log_a, log_b = -np.log(len(x)), -np.log(len(y))
    with torch.no_grad():
        f = torch.zeros(len(x), dtype=x.dtype)
        g = torch.zeros(len(y), dtype=x.dtype)
        level, remaining = max(float(C.max()), eps), extra
        while remaining > 0:
            f = -level * torch.logsumexp((g[None, :] - C) / level + log_b, dim=1)
            g = -level * torch.logsumexp((f[:, None] - C) / level + log_a, dim=0)
            if level <= eps:
                remaining -= 1
            level = max(level * scaling, eps)
    f = -eps * torch.logsumexp((g[None, :] - C) / eps + log_b, dim=1)
    g = -eps * torch.logsumexp((f.detach()[:, None] - C) / eps + log_a, dim=0)
    return f.mean() + g.mean()


def sinkhorn_divergence(x, y, eps: float):
    """Debiased Sinkhorn divergence S(x, y) = OT(x, y) - (OT(x, x) + OT(y, y)) / 2 (the paper's Eq. 3)."""
    return _entropic_ot(x, y, eps) - 0.5 * (_entropic_ot(x, x, eps) + _entropic_ot(y, y, eps))


def sinkhorn_iterations(n_rows: int, visits: float = SINKHORN_VISITS, batch: int = SINKHORN_BATCH) -> int:
    return int(np.ceil(visits * n_rows / (2 * batch)))


def impute_sinkhorn(X_missing, seed: int, visits: float = SINKHORN_VISITS, batch: int = SINKHORN_BATCH,
                    lr: float = SINKHORN_LR, noise: float = SINKHORN_NOISE, eps_share: float = SINKHORN_EPS_SHARE,
                    record_every: int | None = None, monitor=None) -> tuple[np.ndarray, FitInfo]:
    """Batch Sinkhorn imputation. `monitor(iteration, filled)` is called every `record_every` steps if given
    (used only to choose the iteration budget, never in a benchmark run)."""
    import torch
    from .seeds import stream, torch_generator
    X = _check(X_missing)
    began = time.perf_counter()
    n = len(X)
    batch = min(batch, 2 * (n // 2))
    n_iter = sinkhorn_iterations(n, visits, batch)
    record_every = record_every or max(1, n_iter // 100)
    rng = stream(seed, 'sinkhorn', 'batches')
    Xt = torch.tensor(X, dtype=torch.float64)
    gaps = torch.isnan(Xt)
    rows, cols = torch.nonzero(gaps, as_tuple=True)
    means = torch.nanmean(Xt, dim=0)
    start = means[cols] + noise * torch.randn(len(cols), generator=torch_generator(seed, 'sinkhorn', 'start'),
                                              dtype=torch.float64)
    imputed = start.clone().requires_grad_()
    mean_filled = torch.where(gaps, means.expand_as(Xt), Xt)
    sample = mean_filled[torch.as_tensor(stream(seed, 'sinkhorn', 'epsilon').choice(n, min(n, 500), replace=False))]
    distances = 0.5 * torch.cdist(sample, sample) ** 2
    eps = float(eps_share * torch.quantile(distances[distances > 0], 0.5))
    optimiser = torch.optim.RMSprop([imputed], lr=lr)
    base = torch.where(gaps, torch.zeros_like(Xt), Xt)
    position = torch.zeros(Xt.shape, dtype=torch.long)       # index of each missing cell in `imputed`
    position[rows, cols] = torch.arange(len(rows))

    def batch_rows(index):
        """The filled rows of one batch, built from those rows only (not the whole table) to keep steps cheap."""
        return torch.where(gaps[index], imputed[position[index]], base[index])

    history, window = [], []
    for iteration in range(1, n_iter + 1):
        first = torch.as_tensor(rng.choice(n, batch, replace=False))
        second = torch.as_tensor(rng.choice(n, batch, replace=False))
        loss = sinkhorn_divergence(batch_rows(first), batch_rows(second), eps)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        window.append(loss.item())
        if iteration % record_every == 0 or iteration == n_iter:
            history.append({'iteration': iteration, 'visits_per_row': iteration * 2 * batch / n,
                            'loss': float(np.mean(window))})     # mean batch loss since the last record
            window = []
            if monitor is not None:
                out = X.copy()
                out[gaps.numpy()] = imputed.detach().numpy()
                monitor(iteration, out)
    out = X.copy()
    out[gaps.numpy()] = imputed.detach().numpy()
    return out, FitInfo(time.perf_counter() - began, history,
                        {'epsilon': eps, 'iterations': n_iter, 'batch': batch, 'visits_per_row': visits})
