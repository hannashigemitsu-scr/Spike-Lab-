"""Task 1 (non-navigation): color, delay, shape, delay, go  ->  the (x, y) position that (color, shape) names.

Hypothesis: no grid cells. One self-contained file: settings, task, network, training, analysis.
Needs: numpy, scipy, scikit-learn, matplotlib, torch.

    python task1_color_shape_position.py train   --out runs/task1
    python task1_color_shape_position.py analyze --run runs/task1
    python task1_color_shape_position.py train   --out runs/quick --set train.n_steps=500      # quick check

`--set section.key=value` changes any setting in the Settings block below for one run
(for example `--set rnn.n_hidden=512 seed=3`; list values are comma separated:
`--set train.rate_penalty=l1,l2`). `train` writes <out>/model.pt (weights + all settings + seed)
and history.json; `analyze` writes figures and summary.json to <run>/analysis_*/.

THE TASK
  Each (color, shape) pair names one cell of a 16 x 16 grid laid over the 16 x 16 arena
  (coordinates -8..+8): the color index picks the x coordinate and the shape index picks y. The
  "goal" is the center of that cell. The network sees the color, then the shape, each for 10 steps with a random 10-20 step delay after each, then a go cue; during the go cue it must output the goal (x, y).
  The output is scored only while the go cue is on.

  Input channels per step (35 numbers, the same layout in tasks 1 and 2 so one network shape
  fits both): [ color one-hot (16) | shape one-hot (16) | current x, y scaled to [-1, 1] (2) | go (1) ]. Task 1 never turns the x, y channels on.

NETWORK, LOSS, OPTIMIZER (identical across tasks)
  Leaky tanh RNN after Cueva & Wei (2018), 100 units. Loss = position error + L2 firing-rate
  penalty (their eq. 5) + L2 weight penalty on W_in and W_out (their eq. 4). Adam at default
  settings with gradient-norm clipping at 1 (Fascianelli et al. 2024).
"""
import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                      # write figures to files; no window needed
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter, rotate
from scipy.signal import fftconvolve
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler


# =====================================================================================
# 1. SETTINGS
# =====================================================================================
@dataclass
class SpaceConfig:
    arena_size: float = 16.0        # side length, coordinates -8..+8 (project spec)


@dataclass
class RNNConfig:
    n_hidden: int = 100             # Cueva & Wei: 100. Scale up (e.g. 512) only if needed
    alpha: float = 0.1              # Euler step / time constant (paper: step = tau/10)
    noise_std: float = 0.05         # recurrent noise (paper: constant, exact value not important)
    expansion_dim: int = 0          # 0 = off. >0 adds a FIXED random ReLU layer (Fascianelli)
    rec_init: str = "orthogonal"    # paper: orthogonal, or Gaussian with variance 1.5^2/N
    rec_init_gain: float = 1.0


@dataclass
class TrainConfig:
    batch_size: int = 128
    n_steps: int = 5000
    lr: float = 1e-3                # Adam, default settings, as in Fascianelli et al. 2024
    grad_clip: float = 1.0          # gradient-norm clip
    l2_weight: float = 1e-4         # weight penalty: mean squared weight, summed over l2_targets
    l2_targets: tuple = ("W_in", "W_out")   # Cueva & Wei eq. 4: input and output weights, NOT W_rec
    rate_penalty: tuple = ("l2",)   # Cueva & Wei eq. 5: squared rates. Also allowed: ("l1",), ("l1", "l2")
    fr_ratio: float = 0.1           # rate penalty is scaled to ~fr_ratio x position error
                                    # (paper keeps it between error/10 and error/3)
    log_every: int = 100


@dataclass
class CueConfig:
    n_colors: int = 16              # color index -> x coordinate
    n_shapes: int = 16              # shape index -> y coordinate
    stim_dur: int = 10              # steps each cue is shown (leak 0.1 -> time constant ~10)
    delay_min: int = 10             # every delay is drawn from [delay_min, delay_max],
    delay_max: int = 20             # freshly for every delay of every trial
    resp_dur: int = 10              # steps the go cue is on; output scored only here


@dataclass
class AnalysisConfig:
    n_bins: int = 16                # rate-map resolution; 16 = one bin per (color, shape) cell
    smooth_sigma: float = 1.0       # Gaussian smoothing of rate maps, in bins (same default as task 3; 0 = none)
    n_rep: int = 8                  # trials per cell (random delays), averaged into one map
    n_null_nets: int = 5            # untrained networks used as the gridness comparison


@dataclass
class Config:
    seed: int = 0
    space: SpaceConfig = field(default_factory=SpaceConfig)
    rnn: RNNConfig = field(default_factory=RNNConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    cue: CueConfig = field(default_factory=CueConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)


def config_from_dict(d):
    """Inverse of asdict(Config): rebuild the settings saved next to a trained model."""
    parts = {"space": SpaceConfig, "rnn": RNNConfig, "train": TrainConfig, "cue": CueConfig,
             "analysis": AnalysisConfig}
    out = {}
    for key, cls in parts.items():
        vals = dict(d.get(key, {}))
        for f in fields(cls):
            if f.name in vals and isinstance(getattr(cls(), f.name), tuple):
                vals[f.name] = tuple(vals[f.name])
        out[key] = cls(**vals)
    return Config(seed=int(d.get("seed", 0)), **out)


def apply_overrides(cfg, items):
    """'train.n_steps=500' / 'rnn.n_hidden=512' / 'seed=3' -> change cfg in place."""
    for item in items:
        key, _, raw = item.partition("=")
        if key == "seed":
            cfg.seed = int(raw)
            continue
        section, _, name = key.partition(".")
        obj = getattr(cfg, section)
        old = getattr(obj, name)                     # AttributeError = unknown setting
        if isinstance(old, tuple):
            new = tuple(x.strip() for x in raw.split(","))
        elif isinstance(old, bool):
            new = raw.lower() in ("1", "true", "yes")
        else:
            new = type(old)(raw)
        setattr(obj, name, new)
    return cfg


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch so a run can be repeated exactly."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# =====================================================================================
# 2. THE TASK
# =====================================================================================
def n_inputs(cue):
    return cue.n_colors + cue.n_shapes + 2 + 1       # color | shape | current x, y | go


def goal_position(color, shape, cue, arena_size):
    """Center of the arena cell named by (color, shape). Index arrays -> (N, 2)."""
    half = arena_size / 2.0
    x = (np.asarray(color) + 0.5) * arena_size / cue.n_colors - half
    y = (np.asarray(shape) + 0.5) * arena_size / cue.n_shapes - half
    return np.stack([x, y], axis=-1)


def trial_length(cue):
    """Longest possible trial: color, delay, shape, delay, go, with every delay at delay_max."""
    return 2 * cue.stim_dur + 2 * cue.delay_max + cue.resp_dur


def generate_trials(n, cfg, rng, device="cpu", colors=None, shapes=None):
    """Make n trials. Pass colors / shapes (index arrays) to fix them instead of drawing at random.
    Returns a dict:
        inputs (n, T, 35), target (n, T, 2), mask (n, T) = 1 during the response epoch,
        goal (n, 2), color, shape, onsets (n, 3) = start step of color, shape, go (numpy).
    Shorter trials are padded with zeros at the end up to T, the longest possible trial."""
    cue = cfg.cue
    T = trial_length(cue)
    colors = rng.integers(0, cue.n_colors, size=n) if colors is None else np.asarray(colors)
    shapes = rng.integers(0, cue.n_shapes, size=n) if shapes is None else np.asarray(shapes)
    goal = goal_position(colors, shapes, cue, cfg.space.arena_size)

    # start time of every event: color, shape, go. Each cue lasts stim_dur, then a random delay.
    delays = rng.integers(cue.delay_min, cue.delay_max + 1, size=(n, 2))
    onsets = np.zeros((n, 3), dtype=int)
    for k in range(1, 3):
        onsets[:, k] = onsets[:, k - 1] + cue.stim_dur + delays[:, k - 1]

    shape0, go = cue.n_colors, cue.n_colors + cue.n_shapes + 2
    inputs = np.zeros((n, T, n_inputs(cue)), dtype=np.float32)
    target = np.zeros((n, T, 2), dtype=np.float32)
    mask = np.zeros((n, T), dtype=np.float32)
    sd, rd = cue.stim_dur, cue.resp_dur
    for i in range(n):
        t = onsets[i]
        inputs[i, t[0]:t[0] + sd, colors[i]] = 1.0
        inputs[i, t[1]:t[1] + sd, shape0 + shapes[i]] = 1.0
        inputs[i, t[2]:t[2] + rd, go] = 1.0
        mask[i, t[2]:t[2] + rd] = 1.0
        target[i, t[2]:t[2] + rd] = goal[i]

    to_t = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)
    return {"inputs": to_t(inputs), "target": to_t(target), "mask": to_t(mask),
            "goal": goal, "color": colors, "shape": shapes, "onsets": onsets}


def baseline_errors(cfg, rng, n=4000):
    """Errors (per coordinate, response epoch) of simple strategies. A trained network has to
    beat all of them."""
    goal = generate_trials(n, cfg, rng)["goal"]
    return {"guess_zero": float((goal ** 2).mean()),
            "knows_only_color": float((goal[:, 1] ** 2).mean() / 2),
            "knows_only_shape": float((goal[:, 0] ** 2).mean() / 2)}

# =====================================================================================
# 3. THE NETWORK (Cueva & Wei eq. 1-2)
# =====================================================================================
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


def build_model(cfg):
    r = cfg.rnn
    return PathIntegratorRNN(n_inputs(cfg.cue), r.n_hidden, 2, r.alpha, r.noise_std,
                             r.expansion_dim, r.rec_init, r.rec_init_gain)

# =====================================================================================
# 4. LOSS AND TRAINING
# =====================================================================================
def compute_loss(outputs, target, rates, model, tcfg, mask=None):
    """
    outputs, target : (B, T, 2) predicted / true position
    rates           : (B, T, H) firing rates
    mask            : optional (B, T), 1 where the output is scored (tasks with a response epoch);
                      None scores every step
    Returns (total_loss, dict of components for logging).

    Loss = MSE(position)
         + firing-rate penalties            (L1: mean |rate| over batch, time and units;
                                             L2: mean rate^2; any combination)
         + l2_weight * weight penalty       (mean squared weight of each matrix in l2_targets, summed)

    fr_weight is set adaptively so that penalty ~= fr_ratio * position error (an approximation
    of the paper's "between error/10 and error/3" schedule).
    """
    if mask is None:
        mse = F.mse_loss(outputs, target)
    else:
        sq = ((outputs - target) ** 2).mean(-1)                  # (B, T)
        mse = (sq * mask).sum() / mask.sum().clamp(min=1.0)
    kinds = (tcfg.rate_penalty,) if isinstance(tcfg.rate_penalty, str) else tuple(tcfg.rate_penalty)
    terms = {"l1": rates.abs().mean(), "l2": (rates ** 2).mean()}
    if not kinds or any(k not in terms for k in kinds):
        raise ValueError(f"rate_penalty must be made of 'l1'/'l2', got {tcfg.rate_penalty}")
    r_l2 = sum((getattr(model, name).weight ** 2).mean() for name in tcfg.l2_targets)
    # Each rate penalty is rescaled on the fly to (fr_ratio / number of penalties) x position error.
    fr_total = sum((tcfg.fr_ratio / len(kinds)) * mse.detach() / (terms[k].detach() + 1e-8) * terms[k]
                   for k in kinds)
    total = mse + fr_total + tcfg.l2_weight * r_l2
    return total, {"mse": mse.item(), "fr": fr_total.item(), "l2": r_l2.item()}


def train(model, cfg, rng):
    tcfg = cfg.train
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=tcfg.lr)
    history = []
    t0 = time.time()
    for step in range(1, tcfg.n_steps + 1):
        # fresh random trials every step -> effectively infinite data
        batch = generate_trials(tcfg.batch_size, cfg, rng)
        outputs, rates = model(batch["inputs"], noise=True)
        loss, parts = compute_loss(outputs, batch["target"], rates, model, tcfg, batch["mask"])

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        opt.step()

        history.append(parts)
        if step % tcfg.log_every == 0 or step == 1:
            print(f"step {step:6d}  mse {parts['mse']:.5f}  rate penalty {parts['fr']:.3f}  "
                  f"({time.time() - t0:.0f}s)")
    return history

# =====================================================================================
# 5. ANALYSIS: rate maps, gridness, population geometry
# The "rate map" of a unit is its mean firing rate during the response epoch as a function of
# the goal cell.
# Gridness is compared with UNTRAINED networks of the same size, because the response-epoch
# activity barely changes in time, so the time-shift null used for path integration cannot work.
# =====================================================================================
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


@torch.no_grad()
def collect_activity(model, cfg, rng):
    """Run every (color, shape) cell n_rep times with noise OFF. Returns response-epoch averages:
    rates (N, H), pred (N, 2), target (N, 2), pos (N, 2) = goal, and the response-epoch mse."""
    cue = cfg.cue
    cc, ss = np.meshgrid(np.arange(cue.n_colors), np.arange(cue.n_shapes), indexing="ij")
    colors = np.repeat(cc.ravel(), cfg.analysis.n_rep)
    shapes = np.repeat(ss.ravel(), cfg.analysis.n_rep)
    b = generate_trials(len(colors), cfg, rng, colors=colors, shapes=shapes)
    out, rates = model(b["inputs"], noise=False)
    m = b["mask"]
    w = m / m.sum(1, keepdim=True)
    mean_over_resp = lambda x: (x * w[..., None]).sum(1).numpy()
    sq = ((out - b["target"]) ** 2).mean(-1)
    return {"rates": mean_over_resp(rates), "pred": mean_over_resp(out),
            "target": mean_over_resp(b["target"]), "pos": b["goal"],
            "mse": float((sq * m).sum() / m.sum())}


def _maps(rates, pos, cfg):
    a = cfg.analysis
    return compute_rate_maps(rates[:, None, :], pos[:, None, :], cfg.space.arena_size,
                             a.n_bins, a.smooth_sigma)


def analyze_network(model, cfg, rng):
    """Rate maps -> gridness per unit -> compare with untrained networks -> population geometry."""
    a = cfg.analysis
    act = collect_activity(model, cfg, rng)
    maps = _maps(act["rates"], act["pos"], cfg)
    scores = score_all_units(maps)                              # (H, 2): hex, square

    null = []
    for k in range(a.n_null_nets):                              # untrained nets, different seeds
        set_seed(cfg.seed + 1000 + k)
        a0 = collect_activity(build_model(cfg).eval(), cfg, rng)
        null.append(score_all_units(_maps(a0["rates"], a0["pos"], cfg)))
    null = np.concatenate(null) if null else np.full((1, 2), np.nan)
    thresh = np.nanpercentile(null, 95, axis=0) if np.isfinite(null).any() else np.full(2, np.nan)

    # population geometry: each spatial bin is a "condition", latent = (A, B)
    nb, H = a.n_bins, maps.shape[0]
    centers = (np.arange(nb) + 0.5) / nb * cfg.space.arena_size - cfg.space.arena_size / 2
    A, B = np.meshgrid(centers, centers, indexing="ij")
    latents = np.stack([A.ravel(), B.ravel()], axis=1)
    reps = maps.reshape(H, -1).T
    ok = np.all(np.isfinite(reps), axis=1)
    geom = axis_split_generalization(reps[ok], latents[ok])
    geom["participation_ratio"] = participation_ratio(reps[ok])
    geom["n_bins_used"] = int(ok.sum())
    return {"maps": maps, "scores": scores, "thresholds": thresh, "null_scores": null,
            "geometry": geom, "activity": act}


# =====================================================================================
# 6. COMMAND LINE
# =====================================================================================
def clean(x):
    """NaN -> None so summary.json is valid JSON."""
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [clean(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return None if not np.isfinite(x) else float(x)
    return x


def run_train(args):
    cfg = apply_overrides(Config(), args.set)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    set_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    model = build_model(cfg).to(args.device)
    print(f"{sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters")

    base = baseline_errors(cfg, rng)
    print("reference errors (a trained network should beat all of these):")
    for name, value in base.items():
        print(f"  {name:<22}: {value:.3f}")

    history = train(model, cfg, rng)
    torch.save({"state_dict": model.state_dict(), "config": asdict(cfg), "baselines": base},
               out / "model.pt")
    (out / "history.json").write_text(json.dumps(history))
    print("saved to", out)


def run_analyze(args):
    run = Path(args.run)
    ckpt = torch.load(run / "model.pt", map_location="cpu")
    cfg = config_from_dict(ckpt["config"])
    cfg.analysis.n_bins = cfg.cue.n_colors                 # one bin per cell
    if args.n_null_nets is not None:
        cfg.analysis.n_null_nets = args.n_null_nets

    set_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed + 1)              # different stream from training
    model = build_model(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    by = "goal"
    out_dir = run / f"analysis_{by}"
    out_dir.mkdir(exist_ok=True)
    res = analyze_network(model, cfg, rng)
    act, sc, th = res["activity"], res["scores"], res["thresholds"]
    half = cfg.space.arena_size / 2

    # 1) did it learn the task?
    print(f"noise-free response-epoch MSE = {act['mse']:.3f}")
    base = ckpt.get("baselines", {})
    for k, v in base.items():
        print(f"   reference {k:<20}: {v:.3f}")
    learned = {"mse": act["mse"], "baselines": base}
    snap = lambda v: np.clip(np.floor((v + half) / cfg.space.arena_size * cfg.cue.n_colors), 0,
                             cfg.cue.n_colors - 1)
    hit = snap(act["pred"]) == snap(act["target"])
    learned["frac_color_correct"] = float(hit[:, 0].mean())
    learned["frac_shape_correct"] = float(hit[:, 1].mean())
    learned["frac_cell_correct"] = float(hit.all(axis=1).mean())
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.8))
    for ax, j, name in zip(axes, (0, 1), ("x", "y")):
        ax.scatter(act["target"][:, j], act["pred"][:, j], s=4, alpha=0.3)
        lim = [act["target"][:, j].min(), act["target"][:, j].max()]
        ax.plot(lim, lim, "k--", lw=1)
        ax.set_xlabel(f"true {name} output"); ax.set_ylabel(f"network {name} output")
    fig.suptitle("task 1: response epoch"); fig.tight_layout()
    fig.savefig(out_dir / "outputs.png", dpi=150); plt.close(fig)

    # 2) grid-like units?
    ok = np.isfinite(sc[:, 0])
    summary = {"task": "color_shape_position", "maps_indexed_by": by, "learned": learned,
               "n_units": int(sc.shape[0]), "n_valid_scores": int(ok.sum()),
               "untrained_null_95th_hex_square": th,
               "frac_above_null_hex": float(np.mean(sc[ok, 0] > th[0])) if ok.any() else None,
               "frac_above_null_square": float(np.mean(sc[ok, 1] > th[1])) if ok.any() else None,
               "geometry": res["geometry"]}
    print(json.dumps(clean(summary), indent=2))
    (out_dir / "summary.json").write_text(json.dumps(clean(summary), indent=2))
    np.save(out_dir / "rate_maps.npy", res["maps"])
    np.save(out_dir / "gridness_scores.npy", sc)

    best = np.nan_to_num(sc, nan=-9).max(axis=1)
    groups = {"most_periodic": np.argsort(-best)[:16],
              "random": rng.choice(sc.shape[0], min(16, sc.shape[0]), replace=False)}
    for gname, units in groups.items():
        fig, axes = plt.subplots(4, 4, figsize=(8, 8))
        for ax in axes.ravel():
            ax.axis("off")
        for ax, h in zip(axes.ravel(), units):
            ax.imshow(res["maps"][h].T, origin="lower", cmap="viridis")
            ax.set_title(f"u{h} hex {sc[h, 0]:.2f} sq {sc[h, 1]:.2f}", fontsize=7)
        fig.suptitle(f"{gname.replace('_', ' ')} units (maps by {by})")
        fig.tight_layout(); fig.savefig(out_dir / f"rate_maps_{gname}.png", dpi=150); plt.close(fig)
    print("saved to", out_dir)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("train", help="train the network")
    p.add_argument("--out", required=True, help="folder for this run (created if missing)")
    p.add_argument("--set", nargs="*", default=[], metavar="section.key=value",
                   help="override settings, e.g. train.n_steps=500 rnn.n_hidden=512 seed=3")
    p.add_argument("--device", default="cpu")
    p.set_defaults(func=run_train)

    p = sub.add_parser("analyze", help="did it learn the task? are there grid-like units?")
    p.add_argument("--run", required=True, help="folder written by train")
    p.add_argument("--n-null-nets", type=int, default=None, help="untrained networks for the comparison")
    p.set_defaults(func=run_analyze)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
