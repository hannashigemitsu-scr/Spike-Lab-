# %% [markdown]
# # Grid cell RNN: path integration in a 16 x 16 conceptual space
#
# Based on Cueva & Wei (2018). Self-contained: run the cells top to bottom.
#
# | Setting | Value |
# |---|---|
# | Arena | 16 x 16, coordinates -8..+8 |
# | Trajectory | 100 velocity inputs; input *t* moves the agent from step *t* to *t+1* |
# | Input at t = 0 | start position (x0, y0), scaled to [-1, 1] |
# | Input every step | speed, cos(heading), sin(heading) |
# | Output every step | (x, y) position after that step |
# | Walls | near a wall, a heading pointing into it is reflected away, so the agent never leaves the arena |
# | Stillness | 90% of speed inputs are 0 |
# | RNN | vanilla leaky tanh RNN, 1 layer, 512 units, leak alpha = 0.1 |
# | Loss | MSE + L1 firing-rate penalty + L2 weight penalty (W_in, W_rec, W_out) |
#
# **Why the position is given only at t = 0:** if the network saw its true position every step, it
# could just add the velocity to it. Nothing would need to be integrated or remembered, and
# grid-like codes would have no reason to form. Giving only the start position forces the network
# to path-integrate, as in Cueva & Wei.

# %%
import copy
import math
import random
import time
from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter, rotate
from scipy.signal import fftconvolve
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

# %% [markdown]
# ## A. Settings
# Comments say whether a value comes from the paper, the project spec, or is our own choice.

# %%
def set_seed(seed: int) -> None:
    """Seed python, numpy and torch so a run can be repeated exactly."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class SpaceConfig:
    """The 'conceptual space': a bounded 2D square whose axes are factors A and B.
    (In a physical-space version A and B would be x and y; the math is identical.)"""
    arena_size: float = 16.0        # side length, coordinates -8..+8 (project spec)
    max_speed: float = 1.5          # max step size. Paper: 0.2 in a ~2.2-wide arena (~9% of width);
                                    # 1.5 keeps that ratio at width 16 (our scaling)
    heading_noise_std: float = 0.3  # radians of heading wobble per step; small = long straight runs
                                    # ("modified Brownian motion" in the paper). Value is our choice
    zero_speed_frac: float = 0.9    # fraction of steps with speed 0. Paper: "e.g. 90%", crucial for
                                    # grid-like units (project spec: 0.9)
    wall_margin: float = 1.5        # within this distance of a wall, a heading pointing INTO the wall
                                    # is mirrored away. Must be >= max_speed to guarantee the agent
                                    # stays inside (our choice)
    start_mode: str = "random"      # "random": start anywhere, start position given at t = 0 only
                                    # "center": always start at (0, 0), so the start input is all 0
    input_mode: str = "speed_cossin"  # see _make_inputs


@dataclass
class RNNConfig:
    n_hidden: int = 512             # paper: 100; project spec: 512
    alpha: float = 0.1              # Euler step / time constant (paper: step = tau/10)
    noise_std: float = 0.05         # recurrent noise. Paper: constant, exact value not important
    expansion_dim: int = 0          # 0 = off (Cueva & Wei). >0 adds a FIXED random ReLU layer in
                                    # front of the RNN (Fascianelli). Off for the replication
    rec_init: str = "orthogonal"    # paper: orthogonal (square/triangle) or Gaussian with
                                    # variance 1.5^2/N (hexagon)
    rec_init_gain: float = 1.0


@dataclass
class GridTrainConfig:
    seq_len: int = 100              # velocity inputs per trajectory (project spec; paper: 500)
    batch_size: int = 128           # paper: 500 with Hessian-free. Smaller for CPU
    n_steps: int = 5000             # optimizer steps (our choice)
    lr: float = 1e-3                # Adam. The paper used Hessian-free optimization, which PyTorch
                                    # doesn't have. If grid cells don't appear, the optimizer and the
                                    # regularization strength are the first suspects
    grad_clip: float = 1.0
    l2_weight: float = 1e-4         # L2 weight penalty: mean squared weight, summed over l2_targets
    l2_targets: tuple = ("W_in", "W_rec", "W_out")
                                    # paper eq. 4 penalizes W_in and W_out; project spec adds W_rec
    rate_penalty: str = "l1"        # "l1": mean |rate| (project spec); "l2": mean rate^2 (paper)
    fr_ratio: float = 0.1           # firing-rate penalty is scaled to ~fr_ratio x position error.
                                    # Paper keeps it between error/10 and error/3. THE MOST IMPORTANT
                                    # KNOB FOR GRID EMERGENCE (paper sec 3.2)
    log_every: int = 100


@dataclass
class GridAnalysisConfig:
    n_eval_traj: int = 1000         # trajectories used for rate maps (random starts cover the arena)
    eval_seq_len: int = 100         # same as training, so we don't ask for longer integration
    eval_zero_frac: Optional[float] = None
                                    # None = same fraction of still steps as training (safest).
                                    # Lower values explore more but change the input statistics
    n_bins: int = 20                # rate-map resolution (n_bins x n_bins)
    smooth_sigma: float = 1.0       # Gaussian smoothing of rate maps, in bins

# %% [markdown]
# ## B. Trajectory generator
# Random start, 100 velocity inputs, 90% zero speeds, headings reflected away from walls.
# Network input per step = `[x0, y0 (t = 0 only, else 0), speed, cos(heading), sin(heading)]`.

# %%
# Number of per-step features for each input_mode (plus 2 for the start position).
N_STEP_INPUTS = {"speed_angle": 2, "speed_cossin": 3, "cartesian": 2}
N_INPUTS = {k: 2 + v for k, v in N_STEP_INPUTS.items()}


def _make_inputs(speed, heading, step, mode):
    """Turn raw (speed, heading, displacement) into network input features.

    speed_angle  : speed, heading/pi. Jumps at +-pi (3.1 and -3.1 rad are almost the same
                   direction but become +0.99 and -0.99, which look very different).
    speed_cossin : speed, cos(heading), sin(heading). Puts the angle on a circle, no jump.
    cartesian    : displacement in A, displacement in B.
    """
    if mode == "speed_angle":
        feats = [speed, heading / np.pi]
    elif mode == "speed_cossin":
        feats = [speed, np.cos(heading), np.sin(heading)]
    elif mode == "cartesian":
        feats = [step[..., 0], step[..., 1]]   # "..." = all leading dims (trial, time)
    else:
        raise ValueError(f"unknown input_mode {mode}")
    return np.stack(feats, axis=-1)


def _avoid_walls(pos, heading, half, margin):
    """Mirror the heading away from any wall the agent is close to.

    If the agent is within `margin` of a wall AND heading toward it, that component of the
    heading is flipped. With margin >= max_speed this alone keeps every step inside the arena.

    Input : pos (batch, 2), heading (batch,), half = arena_size / 2
    Output: new heading (batch,), turned (batch,) bool
    """
    c, s = np.cos(heading), np.sin(heading)
    flip_x = ((pos[:, 0] > half - margin) & (c > 0)) | ((pos[:, 0] < -half + margin) & (c < 0))
    flip_y = ((pos[:, 1] > half - margin) & (s > 0)) | ((pos[:, 1] < -half + margin) & (s < 0))
    c = np.where(flip_x, -c, c)
    s = np.where(flip_y, -s, s)
    return np.arctan2(s, c), flip_x | flip_y


def generate_trajectories(batch, T, cfg: SpaceConfig, rng: np.random.Generator, device="cpu"):
    half = cfg.arena_size / 2.0

    pos = np.zeros((batch, T + 1, 2))
    if cfg.start_mode == "random":
        pos[:, 0] = rng.uniform(-half, half, size=(batch, 2))
    elif cfg.start_mode != "center":
        raise ValueError(f"unknown start_mode {cfg.start_mode}")
    speed_all = np.zeros((batch, T))
    heading_all = np.zeros((batch, T))
    step_all = np.zeros((batch, T, 2))
    boundary_all = np.zeros((batch, T), dtype=bool)

    heading = rng.uniform(-np.pi, np.pi, size=batch)   # random initial direction

    def propose(h, s, p):
        step = np.stack([s * np.cos(h), s * np.sin(h)], axis=-1)
        return step, p + step

    for t in range(T):
        # 1) heading does a small random walk -> mostly straight runs
        heading = heading + rng.normal(0.0, cfg.heading_noise_std, size=batch)

        # 2) speed for this step; some steps are forced to "stand still"
        speed = rng.uniform(0.0, cfg.max_speed, size=batch)
        still = rng.random(batch) < cfg.zero_speed_frac
        speed = np.where(still, 0.0, speed)

        # 3) near a wall, turn away from it so the velocity keeps us inside
        heading, hit = _avoid_walls(pos[:, t], heading, half, cfg.wall_margin)
        heading = np.angle(np.exp(1j * heading))       # wrap to (-pi, pi]

        # 4) safety net (only matters if wall_margin < max_speed): resample the heading until
        #    the step fits; after 50 tries, freeze whatever is still stuck.
        step, new_pos = propose(heading, speed, pos[:, t])
        outside = np.any(np.abs(new_pos) > half, axis=-1)
        hit |= outside
        for _ in range(50):
            if not outside.any():
                break
            heading = np.where(outside, rng.uniform(-np.pi, np.pi, size=batch), heading)
            step, new_pos = propose(heading, speed, pos[:, t])
            outside = np.any(np.abs(new_pos) > half, axis=-1)
        if outside.any():
            speed = np.where(outside, 0.0, speed)
            step, new_pos = propose(heading, speed, pos[:, t])

        pos[:, t + 1] = new_pos
        speed_all[:, t] = speed
        heading_all[:, t] = heading
        step_all[:, t] = step
        boundary_all[:, t] = hit

    step_feats = _make_inputs(speed_all, heading_all, step_all, cfg.input_mode)
    start_feats = np.zeros((batch, T, 2))
    start_feats[:, 0] = pos[:, 0] / half                # start position, t = 0 only
    inputs = np.concatenate([start_feats, step_feats], axis=-1)
    to_t = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)
    return {
        "pos": pos,                 # numpy (batch, T+1, 2), used for analysis
        "inputs": to_t(inputs),     # (batch, T, n_in)
        "target": to_t(pos[:, 1:]), # (batch, T, 2)
        "boundary": boundary_all,   # (batch, T) True where the heading was turned at a wall
    }

# %% [markdown]
# ## C. The leaky RNN (Cueva & Wei eq. 1-2)

# %%
class PathIntegratorRNN(nn.Module):
    """
    Each time step (Cueva & Wei eq. 1-2, discretized with Euler):

        u_t   = tanh(x_{t-1})                         # firing rates
        drive = W_rec u_t + W_in I_t + b + noise      # what each unit receives
        x_t   = (1 - alpha) * x_{t-1} + alpha * drive # leaky update
        y_t   = W_out tanh(x_t)                       # position readout (no bias)

    Plain language: each unit holds a "memory" x that slowly leaks away (alpha = 0.1: it keeps
    90% of its old state and takes in 10% of its new drive). Its firing rate is tanh of that
    memory, and the position readout is a weighted sum of firing rates.

    The initial state is zero and the readout has no bias. The start position arrives as input
    at t = 0, so the network must store it in its own activity and update it from speed/heading.

    Optional (expansion_dim > 0, Fascianelli): the raw input first passes through a FIXED,
    untrained random ReLU layer. Off for the Cueva & Wei replication.

    inputs : (batch, T, n_in)
    noise  : add recurrent noise (True in training, False for clean rate maps)
    returns: outputs (batch, T, 2) predicted positions, rates (batch, T, n_hidden)
    """

    def __init__(self, n_in, n_hidden=512, n_out=2, alpha=0.1, noise_std=0.05,
                 expansion_dim=0, rec_init="orthogonal", rec_init_gain=1.0):
        super().__init__()
        self.n_hidden, self.alpha, self.noise_std = n_hidden, alpha, noise_std

        self.expansion_dim = expansion_dim
        if expansion_dim > 0:
            self.expand = nn.Linear(n_in, expansion_dim, bias=False)
            nn.init.normal_(self.expand.weight, std=1.0 / math.sqrt(n_in))
            self.expand.weight.requires_grad_(False)       # frozen
            n_in = expansion_dim

        self.W_in = nn.Linear(n_in, n_hidden, bias=False)
        self.W_rec = nn.Linear(n_hidden, n_hidden, bias=False)
        self.b = nn.Parameter(torch.zeros(n_hidden))
        self.W_out = nn.Linear(n_hidden, n_out, bias=False)

        # initialization, following the paper
        nn.init.normal_(self.W_in.weight, std=1.0 / math.sqrt(n_in))      # variance 1/N_in
        if rec_init == "orthogonal":
            nn.init.orthogonal_(self.W_rec.weight, gain=rec_init_gain)
        else:                                                              # Gaussian
            nn.init.normal_(self.W_rec.weight, std=rec_init_gain / math.sqrt(n_hidden))
        nn.init.zeros_(self.W_out.weight)                                  # paper: zeros

    def forward(self, inputs, noise=True):
        B, T, _ = inputs.shape
        if self.expansion_dim > 0:
            inputs = F.relu(self.expand(inputs))

        x = inputs.new_zeros(B, self.n_hidden)
        outs, rates = [], []
        for t in range(T):
            u = torch.tanh(x)
            drive = self.W_rec(u) + self.W_in(inputs[:, t]) + self.b
            if noise and self.noise_std > 0:
                drive = drive + self.noise_std * torch.randn_like(drive)
            x = (1 - self.alpha) * x + self.alpha * drive
            u_new = torch.tanh(x)
            outs.append(self.W_out(u_new))
            rates.append(u_new)
        return torch.stack(outs, dim=1), torch.stack(rates, dim=1)

# %% [markdown]
# ## D. Analysis tools
# Rate maps, spatial autocorrelation, gridness scores, shuffle null, and how linear the
# population code is.
#
# TODO: add shattering dimensionality and cross-condition generalization performance (CCGP).

# %%
def compute_rate_maps(rates, pos, arena_size, n_bins=20, smooth_sigma=1.0):
    """Average each unit's firing rate in every spatial bin.

    rates (N_traj, T, H), pos (N_traj, T, 2) true position at the same steps.
    Returns maps (H, n_bins, n_bins), axes [A bin, B bin]; NaN where never visited.
    """
    H = rates.shape[-1]
    half = arena_size / 2.0
    p = pos.reshape(-1, 2)
    r = rates.reshape(-1, H)
    idx = np.floor((p + half) / arena_size * n_bins).astype(int)
    idx = np.clip(idx, 0, n_bins - 1)
    lin = idx[:, 0] * n_bins + idx[:, 1]

    counts = np.bincount(lin, minlength=n_bins * n_bins).astype(float)
    visited = counts > 0
    maps = np.full((H, n_bins * n_bins), np.nan)
    c_s = gaussian_filter(counts.reshape(n_bins, n_bins), smooth_sigma)
    for h in range(H):
        sums = np.bincount(lin, weights=r[:, h], minlength=n_bins * n_bins)
        s_s = gaussian_filter(sums.reshape(n_bins, n_bins), smooth_sigma)
        m = np.where(c_s > 1e-9, s_s / np.maximum(c_s, 1e-9), np.nan).ravel()
        m[~visited] = np.nan
        maps[h] = m
    return maps.reshape(H, n_bins, n_bins)


def _xcorr(a, b):
    """R_ab(s) = sum_i a[i] * b[i+s] for all 2D shifts s, via FFT."""
    return fftconvolve(b, a[::-1, ::-1], mode="full")


def spatial_autocorr(rm, min_overlap=20):
    """Pearson autocorrelation of a rate map with itself at every 2D shift, using only bins
    that are visited in BOTH shifted copies. Output (2n-1, 2n-1), zero shift at the center."""
    M = np.isfinite(rm).astype(float)
    m = np.where(M > 0, rm, 0.0)
    n = _xcorr(M, M)
    sx, sy = _xcorr(m, M), _xcorr(M, m)
    sxy = _xcorr(m, m)
    sxx, syy = _xcorr(m * m, M), _xcorr(M, m * m)
    num = n * sxy - sx * sy
    den = np.sqrt(np.maximum((n * sxx - sx ** 2) * (n * syy - sy ** 2), 0))
    return np.where((n >= min_overlap) & (den > 1e-12), num / np.maximum(den, 1e-12), 0.0)


def gridness_scores(ac, max_frac=0.8):
    """Sargolini-style gridness. A hexagonal lattice's autocorrelation matches itself when
    rotated by 60 and 120 degrees but NOT by 30, 90, 150:
        hex    = min(r60, r120) - max(r30, r90, r150)
    A square lattice (expected in a square arena per Cueva & Wei) repeats every 90 degrees:
        square = r90 - max(r45, r135)

    Caveats:
      * Cueva & Wei's units show only a few periods, so these scores are noisy.
      * The square score is UNCALIBRATED (averages ~0.3 on smoothed random noise). Never use a
        fixed threshold; compare against shuffle_null_gridness.
    Returns dict(hex=..., square=...), NaN if the map is too weak to score.
    """
    c = (ac.shape[0] - 1) // 2
    yy, xx = np.indices(ac.shape)
    rr = np.hypot(yy - c, xx - c)

    # radial mean profile -> where the central peak ends (first radius where it is <= 0)
    prof = np.array([ac[(rr >= r - 0.5) & (rr < r + 0.5)].mean() for r in range(1, c)])
    below = np.flatnonzero(prof <= 0)
    if len(below) == 0:
        return {"hex": np.nan, "square": np.nan}
    r_in = int(below[0]) + 1

    angles = [30, 45, 60, 90, 120, 135, 150]
    rot = {a: rotate(ac, a, reshape=False, order=1, mode="constant", cval=0.0) for a in angles}

    best_hex, best_sq = -np.inf, -np.inf
    for r_out in range(r_in + 3, int(max_frac * c) + 1):
        mask = (rr > r_in) & (rr <= r_out)
        if mask.sum() < 10:
            continue

        def corr(a):
            v = np.corrcoef(ac[mask], rot[a][mask])[0, 1]
            return v if np.isfinite(v) else np.nan

        r = {a: corr(a) for a in angles}
        if any(np.isnan(v) for v in r.values()):
            continue
        best_hex = max(best_hex, min(r[60], r[120]) - max(r[30], r[90], r[150]))
        best_sq = max(best_sq, r[90] - max(r[45], r[135]))
    if not np.isfinite(best_hex):
        return {"hex": np.nan, "square": np.nan}
    return {"hex": float(best_hex), "square": float(best_sq)}


def score_all_units(maps):
    """gridness_scores for every unit. Returns (H, 2): columns hex, square.
    Silent/constant or barely-visited units get NaN."""
    H = maps.shape[0]
    out = np.full((H, 2), np.nan)
    for h in range(H):
        if np.isfinite(maps[h]).sum() < 50 or np.nanstd(maps[h]) < 1e-6:
            continue
        s = gridness_scores(spatial_autocorr(maps[h]))
        out[h] = (s["hex"], s["square"])
    return out


def shuffle_null_gridness(rates, pos, arena_size, n_bins, smooth_sigma, rng, n_shuffles=20):
    """Null distribution for gridness. Each shuffle circularly shifts every trajectory's
    activity in time by a random offset relative to its position (breaks the link between
    activity and place, keeps each unit's activity statistics), then recomputes maps and scores.

    Returns (n_shuffles, H, 2), last axis = hex, square. Pool over units and shuffles and take
    e.g. the 95th percentile per column as the threshold.
    """
    N, T, H = rates.shape
    out = np.full((n_shuffles, H, 2), np.nan)
    for k in range(n_shuffles):
        offs = rng.integers(T // 4, 3 * T // 4, size=N)
        shuf = np.stack([np.roll(rates[i], offs[i], axis=0) for i in range(N)])
        maps = compute_rate_maps(shuf, pos, arena_size, n_bins, smooth_sigma)
        out[k] = score_all_units(maps)
    return out


def participation_ratio(reps):
    """PR = (sum eig)^2 / sum(eig^2) of the covariance of reps. Roughly the number of
    dimensions the population uses."""
    c = np.cov(reps - reps.mean(0), rowvar=False)
    return float(np.trace(c) ** 2 / (c ** 2).sum())


def _fit_eval_clf(Xtr, ytr, Xte, yte):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000).fit(sc.transform(Xtr), ytr)
    return clf.score(sc.transform(Xte), yte)


def _fit_eval_reg(Xtr, Ytr, Xte, Yte, alpha=1.0):
    sc = StandardScaler().fit(Xtr)
    reg = Ridge(alpha=alpha).fit(sc.transform(Xtr), Ytr)
    pred = reg.predict(sc.transform(Xte))
    # r^2 pooled over latent dimensions; 0 = predict the mean, can be negative
    return 1.0 - ((Yte - pred) ** 2).sum() / ((Yte - Yte.mean(0)) ** 2).sum()


def axis_split_generalization(reps, latents):
    """A literal 2-factor CCGP for points on a grid of positions (e.g. rate maps flattened
    to (bins^2, units) with latents = (A, B)).

    For each split axis s in {A, B}:
      classifier: learn "is the other factor positive?" using only points with factor s < 0,
                  test on points with factor s > 0 (and swap). Does the decoder for A transfer
                  across levels of B?
      regression: learn to read out BOTH coordinates on one side, test on the other.
    Returns averaged clf accuracy (chance 0.5) and reg r^2 (chance 0).
    """
    accs, r2s = [], []
    for s in (0, 1):
        other = 1 - s
        y = (latents[:, other] > 0).astype(int)
        neg, pos = latents[:, s] < 0, latents[:, s] > 0
        for tr, te in ((neg, pos), (pos, neg)):
            accs.append(_fit_eval_clf(reps[tr], y[tr], reps[te], y[te]))
            r2s.append(_fit_eval_reg(reps[tr], latents[tr], reps[te], latents[te]))
    return {"clf_acc": float(np.mean(accs)), "reg_r2": float(np.mean(r2s))}

# %% [markdown]
# ## E. Loss, training loop, evaluation
# Loss = MSE + L1 firing-rate penalty + L2 weight penalty.

# %%
def compute_loss(outputs, target, rates, model, tcfg):
    """
    outputs, target : (B, T, 2) predicted / true position
    rates           : (B, T, H) firing rates
    Returns (total_loss, dict of components for logging).

    Loss = MSE(position)
         + fr_weight * firing-rate penalty  (L1: mean |rate| over batch, time and units;
                                             or L2: mean rate^2)
         + l2_weight * weight penalty       (mean squared weight of each matrix in l2_targets, summed)

    fr_weight is set adaptively so that penalty ~= fr_ratio * position error (an approximation
    of the paper's "between error/10 and error/3" schedule).
    """
    mse = F.mse_loss(outputs, target)
    if tcfg.rate_penalty == "l1":
        r_fr = rates.abs().mean()
    elif tcfg.rate_penalty == "l2":
        r_fr = (rates ** 2).mean()
    else:
        raise ValueError(f"unknown rate_penalty {tcfg.rate_penalty}")
    r_l2 = sum((getattr(model, name).weight ** 2).mean() for name in tcfg.l2_targets)
    fr_weight = tcfg.fr_ratio * mse.detach() / (r_fr.detach() + 1e-8)
    total = mse + fr_weight * r_fr + tcfg.l2_weight * r_l2
    return total, {"mse": mse.item(), "fr": r_fr.item(), "l2": r_l2.item()}


def train(model, scfg, tcfg, rng, device="cpu"):
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=tcfg.lr)
    history = []
    t0 = time.time()
    for step in range(1, tcfg.n_steps + 1):
        # fresh random trajectories every step -> effectively infinite data
        batch = generate_trajectories(tcfg.batch_size, tcfg.seq_len, scfg, rng, device)
        outputs, rates = model(batch["inputs"], noise=True)
        loss, parts = compute_loss(outputs, batch["target"], rates, model, tcfg)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        opt.step()

        history.append(parts)
        if step % tcfg.log_every == 0 or step == 1:
            print(f"step {step:6d}  mse {parts['mse']:.5f}  rate penalty {parts['fr']:.3f}  "
                  f"({time.time() - t0:.0f}s)")
    return history


@torch.no_grad()
def collect_activity(model, scfg, acfg, rng, device="cpu"):
    """Run fresh trajectories with noise OFF; return rates, positions (numpy) and MSE."""
    eval_cfg = copy.copy(scfg)
    if acfg.eval_zero_frac is not None:
        eval_cfg.zero_speed_frac = acfg.eval_zero_frac
    batch = generate_trajectories(acfg.n_eval_traj, acfg.eval_seq_len, eval_cfg, rng, device)
    outputs, rates = model(batch["inputs"], noise=False)
    final_err = F.mse_loss(outputs, batch["target"]).item()
    return rates.cpu().numpy(), batch["pos"][:, 1:], final_err


def analyze_grid(rates, pos, scfg, acfg, rng, n_shuffles=20):
    """Rate maps -> gridness per unit -> compare to a shuffle null -> summary, plus how linear
    the population code is over (A, B). Returns dict(maps, scores, thresholds, geometry)."""
    maps = compute_rate_maps(rates, pos, scfg.arena_size, acfg.n_bins, acfg.smooth_sigma)
    H = maps.shape[0]
    scores = score_all_units(maps)                  # (H, 2): hex, square

    # A unit only counts as periodic if it beats the 95th percentile of the pooled null scores.
    null = shuffle_null_gridness(rates, pos, scfg.arena_size, acfg.n_bins,
                                 acfg.smooth_sigma, rng, n_shuffles)
    thresh = np.nanpercentile(null.reshape(-1, 2), 95, axis=0)    # [hex, square]

    # Population linearity: treat each spatial bin as a "condition" (rep = mean firing vector,
    # latent = (A, B)).
    nb = acfg.n_bins
    centers = (np.arange(nb) + 0.5) / nb * scfg.arena_size - scfg.arena_size / 2
    A, B = np.meshgrid(centers, centers, indexing="ij")
    latents = np.stack([A.ravel(), B.ravel()], axis=1)
    reps = maps.reshape(H, -1).T                    # (nb*nb, H)
    ok = np.all(np.isfinite(reps), axis=1)
    geom = axis_split_generalization(reps[ok], latents[ok])
    geom["participation_ratio"] = participation_ratio(reps[ok])
    geom["n_bins_used"] = int(ok.sum())

    return {"maps": maps, "scores": scores, "thresholds": thresh, "geometry": geom}

# %% [markdown]
# ---
# # Running it

# %%
QUICK = True        # False for the full run (5000 steps, ~20 min on CPU)
SEED = 0
set_seed(SEED)
rng = np.random.default_rng(SEED)

scfg, rcfg, tcfg, acfg = SpaceConfig(), RNNConfig(), GridTrainConfig(), GridAnalysisConfig()
if QUICK:
    tcfg.n_steps, tcfg.log_every = 500, 100
    acfg.n_eval_traj = 400
print("torch", torch.__version__, "| quick mode:", QUICK)

# %% [markdown]
# ## 1. The task
# Each trajectory starts at a random point. The network sees that point once (t = 0), then 100
# velocity inputs, and must report its position after every step.

# %%
batch = generate_trajectories(batch=500, T=tcfg.seq_len, cfg=scfg, rng=rng)
inp, pos = batch["inputs"].numpy(), batch["pos"]
half = scfg.arena_size / 2

print("inputs :", inp.shape, " columns = [x0, y0, speed, cos(heading), sin(heading)]")
print("target :", tuple(batch["target"].shape), " position after each step")
print("fraction of zero-speed inputs :", (inp[:, :, 2] == 0).mean().round(3))
print(f"largest |coordinate| reached  : {np.abs(pos).max():.3f} (arena edge = {half:g})")
print("fraction of steps turned at a wall:", batch["boundary"].mean().round(3))

fig, ax = plt.subplots(figsize=(4.5, 4.5))
for i in range(8):
    ax.plot(pos[i, :, 0], pos[i, :, 1], lw=1)
    ax.plot(*pos[i, 0], "k.", ms=6)
ax.set_xlim(-half, half); ax.set_ylim(-half, half); ax.set_aspect("equal")
ax.set_title("example trajectories (dots = start)"); plt.show()

# %% [markdown]
# ## 2. The model
# Leaky RNN (Cueva & Wei eq. 1-2, Euler step):
# `x_t = 0.9 x_{t-1} + 0.1 (W_rec tanh(x_{t-1}) + W_in I_t + b + noise)`, rates `u_t = tanh(x_t)`,
# output `y_t = W_out u_t`.

# %%
model = PathIntegratorRNN(N_INPUTS[scfg.input_mode], rcfg.n_hidden, 2, rcfg.alpha,
                          rcfg.noise_std, rcfg.expansion_dim, rcfg.rec_init, rcfg.rec_init_gain)
print(model)
print("trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

# %% [markdown]
# ## 3. Training
# Fresh trajectories every step. The firing-rate penalty weight is adjusted on the fly so the
# penalty stays at about `fr_ratio` x the MSE. Set `tcfg.rate_penalty = "l2"` for the paper's
# squared-rate form.

# %%
history = train(model, scfg, tcfg, rng)

plt.figure(figsize=(5, 3))
plt.semilogy([h["mse"] for h in history]); plt.xlabel("step"); plt.ylabel("position MSE")
plt.title("training curve"); plt.show()

# Reference points: training only means something relative to these.
b = generate_trajectories(1000, tcfg.seq_len, scfg, rng)
P = b["pos"]
print("MSE of always guessing the centre         :", round(float((P[:, 1:] ** 2).mean()), 3))
print("MSE of always guessing the start position :", round(float(((P[:, 1:] - P[:, :1]) ** 2).mean()), 3))

# %% [markdown]
# ## 4. Does it path-integrate?
# Noise off. Predicted vs true path for a few trajectories, and error as a function of time step.

# %%
with torch.no_grad():
    b = generate_trajectories(500, tcfg.seq_len, scfg, rng)
    out, _ = model(b["inputs"], noise=False)
out, tgt = out.numpy(), b["target"].numpy()

fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
for ax, i in zip(axes[:3], range(3)):
    ax.plot(tgt[i, :, 0], tgt[i, :, 1], "k-", lw=1.5, label="true")
    ax.plot(out[i, :, 0], out[i, :, 1], "r--", lw=1, label="RNN")
    ax.set_xlim(-half, half); ax.set_ylim(-half, half); ax.set_aspect("equal")
axes[0].legend(fontsize=7)
axes[3].plot(((out - tgt) ** 2).sum(-1).mean(0))
axes[3].set_xlabel("time step"); axes[3].set_ylabel("squared error"); axes[3].set_title("error over time")
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 5. Rate maps and gridness
# Average each unit's firing rate in every spatial bin, then score how periodic each map is.
# Scores are compared with a **shuffle null** (activity shifted in time relative to position).
# Units whose map is one smooth ramp get no score (NaN), because their autocorrelation never
# drops to zero.

# %%
rates_np, pos_np, final_err = collect_activity(model, scfg, acfg, rng)
print("noise-free MSE at evaluation:", round(final_err, 4))

res = analyze_grid(rates_np, pos_np, scfg, acfg, rng, n_shuffles=10 if QUICK else 20)
sc, th = res["scores"], res["thresholds"]
n_ok = int(np.isfinite(sc[:, 0]).sum())
print("units with a valid score:", n_ok, "of", sc.shape[0])
print("shuffle-null 95th percentile  [hex, square]:", np.round(th, 2))
if n_ok:
    frac_hex = float(np.mean(sc[np.isfinite(sc[:, 0]), 0] > th[0]))
    frac_sq = float(np.mean(sc[np.isfinite(sc[:, 1]), 1] > th[1]))
    print(f"fraction above null  hex: {frac_hex:.3f}  square: {frac_sq:.3f}  (chance ~0.05)")
print("population geometry:", res["geometry"])

# %%
# 16 most periodic units, then 16 random units for comparison
best = np.nan_to_num(sc, nan=-9).max(axis=1)
groups = {"most periodic units": np.argsort(-best)[:16],
          "random units": rng.choice(sc.shape[0], 16, replace=False)}
for title, units in groups.items():
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, h in zip(axes.ravel(), units):
        ax.imshow(res["maps"][h].T, origin="lower", cmap="viridis")
        ax.set_title(f"u{h} hex {sc[h, 0]:.2f} sq {sc[h, 1]:.2f}", fontsize=7)
    fig.suptitle(title); plt.tight_layout(); plt.show()

# %%
# Save the trained network
torch.save({"state_dict": model.state_dict(), "space": vars(scfg), "rnn": vars(rcfg),
            "train": vars(tcfg)}, "grid_rnn.pt")
