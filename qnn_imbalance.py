"""
Class-wise qubit angle analysis for a Quantum Neural Network on CIFAR-10 (2 classes).

Usage:
  python qnn_imbalance.py pairs                 # 1) find a class pair that is actually learnable
  python qnn_imbalance.py quick  --a 0 --b 1    # 2) fast sanity run
  python qnn_imbalance.py exp2   --a 0 --b 1    # imbalance sweep (plain vs weighted loss)
  python qnn_imbalance.py exp3   --a 0 --b 1    # sequential / forgetting (+ head-refit probe)
  python qnn_imbalance.py exp4   --a 0 --b 1    # 67/23 degree constraint, plain vs class-balanced angle loss (add --weighted)
  python qnn_imbalance.py exp5   --a 0 --b 1    # sequential imbalance, any number of phases (see --phases)
  python qnn_imbalance.py exp6   --a 0 --b 1    # how much does each qubit rotate under imbalance, does weighting fix it
                                                #   (v2: also saves SIGNED noise-floor shifts + a 'beyond noise' heatmap with * = significant)
  python qnn_imbalance.py exp7   --a 0 --b 1    # NEW: exact per-qubit, per-class contribution to the decision (+ the bias/prior term)
  (do not paste '# comments' into zsh; run commands without them)

v3 changes (exp6 / exp7 only; exp2-5 draw their data exactly as before):
  * exp6/exp7 data draws: the balanced reference uses per-class indices [0, 300) ONLY; the noise-floor controls and every
    imbalanced config draw from [300, 1000) with their own seeds. Before, imbalanced configs used the same seed as the
    reference and were NESTED in it ((500,200) and (200,500) contained the whole reference set), which shrank their shifts
    and made the 'weighted below the floor' result an artefact. The (200,200) column is still the reference data itself,
    so it stays exactly 0 as a sanity check.
  * error bars are standard errors over seeds (was std); the plain/weighted/floor comparison prints 95% t-intervals
  * --no_bias : drop the head's bias (Linear(4,2,bias=False)). The prior can then no longer hide in a classical bias term,
                so any imbalance has to be absorbed by the circuit -> this is the "can the qubits solve imbalance" test.
  * exp7 also prints / plots how much of log(nB/nA) the bias absorbs (plain minus weighted bias of the same config, paired)

v4 changes:
  * --freeze_head : classical head fixed to  logit_A = -sum(z), logit_B = +sum(z)  (no bias, not trained). Only the circuit
                    learns, so any imbalance effect on recall/angles has to go THROUGH THE QUBITS. Check the printed balanced
                    recall first: if the frozen-head model does not learn the pair (recall far below the trained-head ~0.96
                    on 0v1), the frozen-head imbalance results are not comparable.
  * exp4 --lams 0,5,20,100       : sweep of angle-loss weights (default 0,1,5). The loss is MSE on <Z>=cos(angle), which is weak
                                   near 23 deg, so large lambdas are needed to see whether class B can reach its target.
  * exp4 --configs all           : run the constraint for every imbalanced config (or "500,100;100,500" ...), one PNG each.
                                   exp4 now prints recall (+/-SE over seeds) and each class's miss = mean angle - target.

Options: --features resnet|pixels (default resnet), --epochs N, --seeds N,
         --target_a DEG --target_b DEG (angle targets for exp4/train, default 67 and 23)
         --phases "500,100;100,500;500,200"  (exp5: class A,B sample counts per phase, quotes needed in zsh)
         --out DIR  parent folder for results (default results)
Outputs: every run creates its OWN new numbered folder  results/run_1/, results/run_2/, ...  holding all PNGs and log.txt
         (everything printed, including the exact command). Nothing existing is ever overwritten.
         results/runs_index.txt gets one line per run (run number, time, experiment, command) so you can find a run again.
Examples: exp4 --target_a 67 --target_b 45   |   exp4 --target_a 23 --target_b 67 (swap)
          exp6 --epochs 150 --seeds 5   (40 epochs = only 40 Adam steps; angles are still drifting, use more)

What "angle" means here (the circuit weights are shared by both classes, so only the DATA makes classes differ):
  input angle   : RY(x_q) fed in, x_q = pi * scaled PCA feature            (class-dependent, not trained)
  output angle  : theta_q = acos(<Z_q>) after the circuit, 0 = |0>, 90 = equator, 180 = |1>   (class-dependent)
  weight rotation: trainable Rot(phi,theta,omega) per layer/qubit; net rotation alpha has
                   cos(alpha/2) = cos(theta/2) cos((phi+omega)/2)           (class-independent)
CIFAR-10 classes: 0 airplane, 1 automobile, 2 bird, 3 cat, 4 deer,
                  5 dog, 6 frog, 7 horse, 8 ship, 9 truck
"""
import argparse
import datetime
import os
import re
import sys
import numpy as np
import torch
import torch.nn.functional as TF
import torchvision
import pennylane as qml
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from scipy import stats

# ----------------------------- config -----------------------------
N_QUBITS, N_LAYERS = 4, 3
EPOCHS, LR = 40, 0.05
CONFIGS = [(200, 200), (500, 100), (100, 500), (500, 200), (200, 500)]
OBS_QUBITS = (0, 1)              # qubits that get the angle constraint
TARGET_DEG = (67.0, 23.0)        # class A -> 67 deg, class B -> 23 deg
PAIRS = [(0, 1), (0, 8), (1, 6), (3, 5)]
PER_CLASS = 1000                 # images per class used from train and from test
REF_POOL = (0, 300)              # exp6/exp7: per-class train indices used ONLY by the balanced reference model
OTHER_POOL = (300, 1000)         # exp6/exp7: controls and imbalanced configs draw from here -> never share a sample with the reference
HEAD_BIAS = True                 # set from --no_bias in main; False = linear head without bias
FREEZE_HEAD = False              # set from --freeze_head in main; True = fixed symmetric head, only the circuit is trained
RES = 128                        # ResNet input size (upsampled from 32x32)
CACHE = "cache"
OUT = "results"                  # every run gets its own NEW folder run_<n> in here (see run_dir_for), nothing is overwritten
RUN_DIR = "."                    # set by main to the folder of the current run


def run_dir_for(name, epochs, seeds, out=OUT):
    """Create and return the next numbered folder <out>/run_1, run_2, ... (never reuses one, also safe if two runs start together).
    One line per run is appended to <out>/runs_index.txt. Older folders with other names are left alone and ignored."""
    os.makedirs(out, exist_ok=True)
    nums = []
    for f in os.listdir(out):
        m = re.fullmatch(r"run_(\d+)", f)
        if m:
            nums.append(int(m.group(1)))
    k = max(nums, default=0) + 1
    while True:
        d = f"{out}/run_{k}"
        try:
            os.makedirs(d)                                   # fails if another run just took this number -> try the next one
            break
        except FileExistsError:
            k += 1
    with open(f"{out}/runs_index.txt", "a") as f:
        f.write(f"run_{k} | {datetime.datetime.now():%Y-%m-%d %H:%M} | {name} | python {os.path.basename(sys.argv[0])} "
                f"{' '.join(sys.argv[1:])}\n")
    return d


class Tee:
    """Write everything printed to the terminal AND to the run's log file."""
    def __init__(self, *files):
        self.files = files

    def write(self, s):
        for f in self.files:
            f.write(s)

    def flush(self):
        for f in self.files:
            f.flush()

    def __getattr__(self, name):                 # isatty(), encoding, ... come from the real terminal stream
        return getattr(self.files[0], name)


torch.set_default_dtype(torch.float32)
dev = qml.device("default.qubit", wires=N_QUBITS)


@qml.qnode(dev, interface="torch", diff_method="backprop")
def circuit(x, w):
    # x: [N, n_qubits] -> parameter broadcasting, whole batch in one call
    qml.AngleEmbedding(x, wires=range(N_QUBITS), rotation="Y")
    qml.StronglyEntanglingLayers(w, wires=range(N_QUBITS))
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]


class QNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(0.1 * torch.randn(N_LAYERS, N_QUBITS, 3))
        self.register_buffer("w0", self.w.detach().clone())   # init weights, to measure how far training rotates them
        self.head = torch.nn.Linear(N_QUBITS, 2, bias=HEAD_BIAS)
        if FREEZE_HEAD:                                       # fixed head: margin logit_B - logit_A = 2 * sum_q <Z_q>; no bias
            with torch.no_grad():                             # (the default init above has already consumed its random numbers)
                self.head.weight.copy_(torch.tensor([[-1.0] * N_QUBITS, [1.0] * N_QUBITS]))
                if self.head.bias is not None:
                    self.head.bias.zero_()
            for p_ in self.head.parameters():
                p_.requires_grad_(False)

    def forward(self, x, ablate=None):
        z = torch.stack(circuit(x, self.w), dim=1).float()   # [N, n_qubits]
        zz = z.clone()
        if ablate is not None:
            zz[:, ablate] = 0.0                              # remove one qubit's contribution
        return self.head(zz), z


# ----------------------------- data -----------------------------
S = {}   # global state filled by setup()
_cifar = {}
_resnet = {}


def _load_cifar():
    if not _cifar:
        print("Loading CIFAR-10 ...")
        tr = torchvision.datasets.CIFAR10("./data", train=True, download=True)
        te = torchvision.datasets.CIFAR10("./data", train=False, download=True)
        _cifar.update(Xtr=tr.data, ytr=np.array(tr.targets), Xte=te.data, yte=np.array(te.targets))
    return _cifar


def _extract(imgs):
    """ResNet18 (ImageNet-pretrained) 512-d features for uint8 images [N,32,32,3]."""
    dv = "mps" if torch.backends.mps.is_available() else "cpu"
    if "m" not in _resnet:
        m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        m.fc = torch.nn.Identity()
        _resnet["m"] = m.eval().to(dv)
    m = _resnet["m"]
    mean = torch.tensor([0.485, 0.456, 0.406], device=dv).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dv).view(1, 3, 1, 1)
    out = []
    with torch.no_grad():
        for i in range(0, len(imgs), 128):
            b = torch.tensor(imgs[i:i + 128], device=dv).permute(0, 3, 1, 2).float() / 255.0
            b = TF.interpolate(b, size=RES, mode="bilinear", align_corners=False)
            out.append(m((b - mean) / std).cpu().numpy())
    return np.concatenate(out)


def class_feats(c, features):
    """(train_feats, test_feats) for CIFAR class c, cached on disk."""
    os.makedirs(CACHE, exist_ok=True)
    path = f"{CACHE}/feat_{features}_{c}.npz"
    if os.path.exists(path):
        d = np.load(path)
        return d["tr"], d["te"]
    d = _load_cifar()
    itr = np.where(d["ytr"] == c)[0][:PER_CLASS]
    ite = np.where(d["yte"] == c)[0][:PER_CLASS]
    if features == "pixels":
        tr = d["Xtr"][itr].reshape(len(itr), -1) / 255.0
        te = d["Xte"][ite].reshape(len(ite), -1) / 255.0
    else:
        print(f"Extracting ResNet18 features for class {c} ...")
        tr, te = _extract(d["Xtr"][itr]), _extract(d["Xte"][ite])
    np.savez(path, tr=tr, te=te)
    return tr, te


def setup(a, b, features="resnet"):
    """Fit PCA + scaling on the train data of the two chosen classes."""
    trA, teA = class_feats(a, features)
    trB, teB = class_feats(b, features)
    pca = PCA(N_QUBITS).fit(np.r_[trA, trB])
    p = pca.transform(np.r_[trA, trB])
    S.update(a=a, b=b, pca=pca, lo=np.percentile(p, 2, 0), hi=np.percentile(p, 98, 0),
             train=(trA, trB), test=(teA, teB), tag=f"{a}v{b}_{features}")
    print(f"Setup done: classes {a} vs {b}, features={features}, tag={S['tag']}")


def _scale(raw):
    f = (S["pca"].transform(raw) - S["lo"]) / (S["hi"] - S["lo"])
    return np.pi * np.clip(f, 0, 1)                     # angles in [0, pi]


def make(nA, nB, seed=0, split="train", raw=False, pool=None):
    """pool=(lo, hi): draw only from per-class indices [lo, hi). pool=None draws from everything (old behaviour, same random stream)."""
    FA, FB = S[split]
    r = np.random.RandomState(seed)
    loA, hiA = (0, len(FA)) if pool is None else (pool[0], min(pool[1], len(FA)))
    loB, hiB = (0, len(FB)) if pool is None else (pool[0], min(pool[1], len(FB)))
    ia = loA + r.choice(hiA - loA, nA, replace=False)
    ib = loB + r.choice(hiB - loB, nB, replace=False)
    X = np.r_[FA[ia], FB[ib]]
    lab = np.r_[np.zeros(nA), np.ones(nB)]
    if raw:
        return X, lab.astype(int)
    return torch.tensor(_scale(X), dtype=torch.float32), torch.tensor(lab).long()


# ----------------------------- helpers -----------------------------
def train(model, x, lab, epochs=EPOCHS, lam=0.0, weighted=False, balanced_angle=False, opt=None):
    opt = opt or torch.optim.Adam(model.parameters(), lr=LR)
    tz = torch.cos(torch.deg2rad(torch.tensor(TARGET_DEG)))        # target <Z> per class
    w = None
    if weighted:                                                   # inverse-frequency class weights
        cnt = torch.bincount(lab, minlength=2).float().clamp(min=1)
        w = cnt.sum() / (2 * cnt)
    for _ in range(epochs):
        out, z = model(x)
        loss = TF.cross_entropy(out, lab, weight=w)
        if lam > 0:
            err = ((z[:, list(OBS_QUBITS)] - tz[lab][:, None]) ** 2).mean(1)   # [N]
            if balanced_angle:                                     # each class counts equally
                ang = torch.stack([err[lab == c].mean() for c in (0, 1) if (lab == c).any()]).mean()
            else:
                ang = err.mean()
            loss = loss + lam * ang
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


@torch.no_grad()
def evaluate(model, x, lab, ablate=None):
    out, z = model(x, ablate=ablate)
    pred = out.argmax(1)
    accA = (pred[lab == 0] == 0).float().mean().item()
    accB = (pred[lab == 1] == 1).float().mean().item()
    ang = torch.rad2deg(torch.acos(z.clamp(-1, 1))).numpy()       # [N, n_qubits]
    return accA, accB, ang[(lab == 0).numpy()], ang[(lab == 1).numpy()]


def test_set():
    return make(200, 200, seed=100, split="test")


def net_rot_deg(w):
    """Net rotation angle (deg, 0..180) of each qubit's Rot(phi, theta, omega) gate in one layer. w: [n_qubits, 3].
    Rot = RZ(omega) RY(theta) RZ(phi) is an SU(2) rotation by alpha with cos(alpha/2) = cos(theta/2) cos((phi+omega)/2)."""
    c = torch.cos(w[:, 1] / 2) * torch.cos((w[:, 0] + w[:, 2]) / 2)
    return torch.rad2deg(2 * torch.acos(c.abs().clamp(max=1.0)))


def mean_angles(model, x, lab):
    """Class-wise mean output angle per qubit, shape [2, n_qubits], plus the two recalls."""
    aA, aB, gA, gB = evaluate(model, x, lab)
    return np.stack([gA.mean(0), gB.mean(0)]), aA, aB


def parse_phases(s):
    """'500,100;100,500;500,200' -> [(500, 100), (100, 500), (500, 200)]"""
    return [tuple(int(v) for v in ph.split(",")) for ph in s.split(";")]



def parse_cfgs(s):
    """exp4 --configs: 'all' = the four imbalanced configs, otherwise 'A,B;A,B;...' like --phases."""
    return list(CONFIGS[1:]) if s.strip().lower() == "all" else parse_phases(s)


def parse_lams(s):
    """'0,5,20,100' -> (0.0, 5.0, 20.0, 100.0)"""
    return tuple(float(v) for v in s.split(","))


# ----------------------------- v2 helpers (exp6 signed noise control, exp7 contributions) -----------------------------
def _sem(a):
    """Standard error of the mean over axis 0 (NaN, not a warning, when there is a single seed)."""
    n = a.shape[0]
    return a.std(0, ddof=1) / np.sqrt(n) if n > 1 else np.full(a.shape[1:], np.nan)


def ci95(a):
    """Mean and 95% t-interval (lo, hi) of a 1-d sample; interval is NaN for a single seed."""
    a = np.asarray(a, float)
    n, mu = len(a), a.mean()
    if n < 2:
        return mu, np.nan, np.nan
    h = stats.t.ppf(0.975, n - 1) * a.std(ddof=1) / np.sqrt(n)
    return mu, mu - h, mu + h


def cfg_data(cfg, s):
    """Training draw for an exp6/exp7 config. Balanced (200,200) = the reference data itself (REF_POOL, seed s), so its shift
    is exactly 0. Every other config draws from OTHER_POOL with its own seed: independent of the reference and of the others.
    Plain and weighted models of the same config and seed get the same data (paired)."""
    if cfg == (200, 200):
        return make(*cfg, seed=s, pool=REF_POOL)
    return make(*cfg, seed=s + 3000 + 100 * CONFIGS.index(cfg), pool=OTHER_POOL)


def _closest(ctrl_keys, cfg):
    """The balanced control whose total sample count is closest to cfg (same rule exp6 already used for the noise floor)."""
    return min(ctrl_keys, key=lambda c: abs(sum(c) - sum(cfg)))


def beyond_noise(D_cfg, D_ctrl):
    """Signed shift of an imbalanced model BEYOND a balanced control, paired by seed (same init, same reference).
    D_cfg, D_ctrl: [seeds, 2, Q] angle shifts vs the balanced reference.
    Returns (mean [2,Q], significant [2,Q] bool). Significant = paired |t| above the two-sided 95% critical value.
    Not corrected for the 8 cells -> use it as a screen, not as proof."""
    diff = D_cfg - D_ctrl
    n = diff.shape[0]
    mu = diff.mean(0)
    if n < 2:
        return mu, np.zeros(mu.shape, bool)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = mu / _sem(diff)
    return mu, np.abs(t) > stats.t.ppf(0.975, n - 1)


def report_beyond_noise(R, ctrlD, configs):
    """Print the signed shift beyond noise for every (plain/weighted, config); return {(weighted, cfg): (mean, significant)}."""
    print("\nSIGNED shift beyond noise = (imbalanced - closest balanced control), paired by seed, deg.\n"
          "  * = paired |t| above the 95% critical value (uncorrected over 8 cells: a screen, not proof)")
    fmt = lambda row, sg: "[" + " ".join(f"{v:+6.1f}{'*' if k else ' '}" for v, k in zip(row, sg)) + "]"
    out = {}
    for weighted in (False, True):
        for cfg in configs:
            cc = _closest(ctrlD, cfg)
            mu, sig = beyond_noise(R[(weighted, cfg)]["D"], ctrlD[cc])
            out[(weighted, cfg)] = (mu, sig)
            print(f"{'weighted' if weighted else 'plain   '} {cfg} minus control{cc}:  A={fmt(mu[0], sig[0])}  B={fmt(mu[1], sig[1])}")
    return out


def plot_beyond_noise(beyond, ctrlD, configs, path):
    """Column 0 = SIGNED shift of the balanced controls (new draw: pure resampling noise, rows = the two controls).
    Columns 1.. = imbalanced minus control (row 0 plain, row 1 weighted). * = significant (see beyond_noise)."""
    keys = list(ctrlD)
    lim = max(10.0, *(np.abs(mu).max() for mu, _ in beyond.values()), *(np.abs(v.mean(0)).max() for v in ctrlD.values()))
    fig, ax = plt.subplots(2, len(configs) + 1, figsize=(3.4 * (len(configs) + 1), 5.8), sharey=True,
                           gridspec_kw={"hspace": 0.6})
    panels = [(r_, 0, ctrlD[k].mean(0), None, f"control {k[0]}/{k[1]} (new draw)\npure noise, signed") for r_, k in enumerate(keys[:2])]
    for r_, weighted in enumerate((False, True)):
        for j, cfg in enumerate(configs, start=1):
            mu, sig = beyond[(weighted, cfg)]
            panels.append((r_, j, mu, sig, f"{'weighted' if weighted else 'plain'} {cfg}\nminus control"))
    for r_, j, M, sig, ttl in panels:
        a_ = ax[r_][j]
        im = a_.imshow(M, cmap="coolwarm", vmin=-lim, vmax=lim, aspect="auto")
        for i in range(2):
            for q in range(N_QUBITS):
                star = "*" if (sig is not None and sig[i, q]) else ""
                a_.text(q, i, f"{M[i, q]:+.1f}{star}", ha="center", va="center", fontsize=8)
        a_.set_xticks(range(N_QUBITS)); a_.set_xlabel("qubit")
        a_.set_yticks([0, 1]); a_.set_yticklabels(["class A", "class B"])
        a_.set_title(ttl, fontsize=9)
    fig.colorbar(im, ax=ax.ravel().tolist(), shrink=0.8, label="signed angle shift (deg)")
    plt.savefig(path, dpi=150, bbox_inches="tight")


def contrib_from_arrays(z, out, W, b, lab):
    """Exact additive split of the logit margin  m = logit_B - logit_A  (the head is linear in <Z>):
        m(x) = sum_q (W[B,q] - W[A,q]) * <Z_q>(x) + (b[B] - b[A])
    z [N,Q] circuit outputs, out [N,2] logits, W [2,Q], b [2], lab [N] in {0,1}. Returns
      toward [2,Q]: per class, mean push of qubit q TOWARD THE TRUE CLASS (+ helps, - hurts), logit units
      bias   [2]  : the same for the bias term (the prior; under plain loss it favours the majority class)
      disc   [Q]  : class separation carried by qubit q = mean_B(c_q) - mean_A(c_q); bias cancels; a negative value = works against separation
      err         : max |sum of parts - real margin|, must be ~1e-6 (proves the split is exact)"""
    dW, db = W[1] - W[0], b[1] - b[0]
    c = z * dW[None, :]                                          # [N,Q] contribution of each qubit to the margin
    err = float(np.abs(c.sum(1) + db - (out[:, 1] - out[:, 0])).max())
    sign = 2.0 * lab - 1.0                                       # +1 class B, -1 class A
    cs = c * sign[:, None]                                       # push toward the true class
    toward = np.stack([cs[lab == k].mean(0) for k in (0, 1)])
    bias = np.array([-db, db])
    disc = c[lab == 1].mean(0) - c[lab == 0].mean(0)
    return toward, bias, disc, err


@torch.no_grad()
def qubit_contrib(model, x, lab):
    """Thin torch wrapper around contrib_from_arrays (all the math lives there so it can be tested without torch)."""
    out, z = model(x)
    b = model.head.bias.detach().numpy() if model.head.bias is not None else np.zeros(2, dtype=np.float32)
    return contrib_from_arrays(z.detach().numpy(), out.detach().numpy(), model.head.weight.detach().numpy(),
                               b, lab.detach().numpy())


def plot_exp7(R, configs, focus, prefix):
    """Fig 1: heatmaps of per-class push toward the true class (columns q0..q3 + bias), rows plain/weighted, cols configs.
    Fig 2: (left) class separation D_q per qubit for the focus config vs the balanced model; (right) bias term b_B-b_A per config."""
    names = [f"q{q}" for q in range(N_QUBITS)] + ["bias"]
    M = {k: np.concatenate([v["toward"].mean(0), v["bias"].mean(0)[:, None]], axis=1) for k, v in R.items()}   # [2, Q+1]
    lim = max(np.abs(m).max() for m in M.values())
    fig, ax = plt.subplots(2, len(configs), figsize=(3.6 * len(configs), 5.8), sharey=True, gridspec_kw={"hspace": 0.55})
    for r_, weighted in enumerate((False, True)):
        for a_, cfg in zip(ax[r_], configs):
            m = M[(weighted, cfg)]
            im = a_.imshow(m, cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
            for i in range(2):
                for j in range(N_QUBITS + 1):
                    a_.text(j, i, f"{m[i, j]:+.2f}", ha="center", va="center", fontsize=8)
            a_.set_xticks(range(N_QUBITS + 1)); a_.set_xticklabels(names, fontsize=8)
            a_.set_yticks([0, 1]); a_.set_yticklabels(["class A", "class B"])
            a_.set_title(f"{'weighted' if weighted else 'plain'} {cfg}", fontsize=10)
    fig.colorbar(im, ax=ax.ravel().tolist(), shrink=0.8, label="push toward the TRUE class (logit units, + = helps)")
    plt.savefig(f"{prefix}_exp7_contrib_heatmap.png", dpi=150, bbox_inches="tight")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    qs = np.arange(N_QUBITS)
    for k, (lab_, key, col) in enumerate((("balanced (200,200)", (False, (200, 200)), "C7"),
                                          (f"plain {focus}", (False, focus), "C3"),
                                          (f"weighted {focus}", (True, focus), "C2"))):
        d = R[key]["disc"]
        ax[0].bar(qs + (k - 1) * 0.27, d.mean(0), 0.27, yerr=_sem(d), capsize=2, color=col, label=lab_)
    ax[0].axhline(0, c="k", lw=0.8)
    ax[0].set_xticks(qs); ax[0].set_xlabel("qubit"); ax[0].set_ylabel("class separation carried, D_q (logit units)")
    ax[0].set_title(f"how much of the A-vs-B separation each qubit carries, {focus} (bars: SE)"); ax[0].legend(fontsize=8)
    xs = np.arange(len(configs))
    for k, (weighted, name, col) in enumerate(((False, "plain", "C3"), (True, "weighted", "C2"))):
        v = [R[(weighted, c)]["bias"][:, 1] for c in configs]                     # b_B - b_A per seed
        ax[1].bar(xs + (k - 0.5) * 0.38, [a.mean() for a in v], 0.38, yerr=[_sem(a) for a in v], capsize=2, color=col, label=name)
    ax[1].plot(xs, [R[(True, c)]["bias"][:, 1].mean() + np.log(c[1] / c[0]) for c in configs], "k_", ms=34, mew=2.5, zorder=5,
               label="weighted bias + log(nB/nA)")                                # what a full prior correction would look like
    ax[1].axhline(0, c="k", lw=0.8)
    ax[1].set_xticks(xs); ax[1].set_xticklabels([f"{c[0]}/{c[1]}" for c in configs])
    ax[1].set_ylabel("bias term b_B - b_A (+ favours class B)"); ax[1].set_title("the prior: how the bias absorbs imbalance (bars: SE)")
    ax[1].legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)   # below the axes: hides no bars or markers
    plt.tight_layout(); plt.savefig(f"{prefix}_exp7_summary.png", dpi=150)


# ----------------------------- experiments -----------------------------
def pairs(args):
    """Find a class pair where even a classical model can separate the classes."""
    rows = []
    for a, b in PAIRS:
        setup(a, b, args.features)
        X, y = make(200, 200, seed=0, raw=True)
        Xt, yt = make(200, 200, seed=100, split="test", raw=True)
        lr_full = LogisticRegression(max_iter=2000).fit(X, y).score(Xt, yt)
        f4, f4t = _scale(X), _scale(Xt)
        lr_4 = LogisticRegression(max_iter=2000).fit(f4, y).score(f4t, yt)
        torch.manual_seed(0)
        x, l = make(200, 200, seed=0)
        m = train(QNN(), x, l, epochs=args.epochs)
        xt, lt = test_set()
        aA, aB, _, _ = evaluate(m, xt, lt)
        rows.append((a, b, lr_full, lr_4, (aA + aB) / 2))
        print(f"  pair {a}v{b}: classical(full)={lr_full:.2f}  classical(4-d)={lr_4:.2f}  QNN={(aA + aB) / 2:.2f}")
    print("\npair | classical full | classical 4-d | QNN (balanced test acc)")
    for r in rows:
        print(f"{r[0]}v{r[1]}  | {r[2]:.2f} | {r[3]:.2f} | {r[4]:.2f}")
    print("\nPick a pair where classical 4-d is well above 0.5, then run exp2/exp3/exp4 with --a --b.")


def exp2(args, seeds=None, epochs=None):
    """Imbalance sweep: per-qubit per-class angle, class-wise recall, ablation. Plain vs weighted CE."""
    seeds = range(seeds if seeds is not None else args.seeds)
    epochs = epochs or args.epochs
    xt, lt = test_set()
    R = {}
    for weighted in (False, True):
        for cfg in CONFIGS:
            angA, angB, accs, abl = [], [], [], []
            for s in seeds:
                torch.manual_seed(s)
                x, l = make(*cfg, seed=s)
                m = train(QNN(), x, l, epochs=epochs, weighted=weighted)
                aA, aB, gA, gB = evaluate(m, xt, lt)
                angA.append(gA.mean(0)); angB.append(gB.mean(0)); accs.append((aA, aB))
                drops = []
                for q in range(N_QUBITS):                      # one forward pass per ablated qubit
                    rA, rB = evaluate(m, xt, lt, ablate=q)[:2]
                    drops.append((aA - rA, aB - rB))
                abl.append(drops)
            R[(weighted, cfg)] = dict(angA=np.array(angA), angB=np.array(angB),
                                      acc=np.array(accs), abl=np.array(abl).mean(0), abl_all=np.array(abl))
            a = R[(weighted, cfg)]["acc"].mean(0)
            print(f"{'weighted' if weighted else 'plain   '} {cfg}: recall A={a[0]:.2f} B={a[1]:.2f}")
            top = np.array(abl).sum(2).argmax(1)               # most important qubit per seed
            print(f"    top ablation qubit per seed: {top.tolist()}")

    tag = S["tag"]
    # Plot 1: angle per qubit per class (row 1 plain, row 2 weighted)
    fig, ax = plt.subplots(2, len(CONFIGS), figsize=(20, 7), sharey=True)
    for r_, weighted in enumerate((False, True)):
        for a_, cfg in zip(ax[r_], CONFIGS):
            d = R[(weighted, cfg)]
            for q in range(N_QUBITS):
                a_.errorbar(q - 0.1, d["angA"][:, q].mean(), d["angA"][:, q].std(), fmt="o", c="C0")
                a_.errorbar(q + 0.1, d["angB"][:, q].mean(), d["angB"][:, q].std(), fmt="s", c="C1")
            a_.set_title(f"{'weighted' if weighted else 'plain'}  A={cfg[0]}, B={cfg[1]}")
            a_.set_xlabel("qubit"); a_.set_xticks(range(N_QUBITS))
        ax[r_][0].set_ylabel("angle (deg), blue=A orange=B")
    plt.tight_layout(); plt.savefig(f"{RUN_DIR}/{tag}_exp2_angles.png", dpi=150)

    # Plot 2: class-wise recall, plain vs weighted
    fig, a_ = plt.subplots(figsize=(10, 4))
    xs = np.arange(len(CONFIGS))
    series = [("plain A", False, 0, "C0", ""), ("plain B", False, 1, "C1", ""),
              ("weighted A", True, 0, "C0", "//"), ("weighted B", True, 1, "C1", "//")]
    for i, (lab_, wt, k, col, h) in enumerate(series):
        vals = [R[(wt, c)]["acc"][:, k].mean() for c in CONFIGS]
        errs = [R[(wt, c)]["acc"][:, k].std() for c in CONFIGS]
        a_.bar(xs + (i - 1.5) * 0.2, vals, 0.2, yerr=errs, capsize=2, label=lab_, color=col, hatch=h, alpha=0.85)
    a_.set_xticks(xs); a_.set_xticklabels([f"{c[0]}/{c[1]}" for c in CONFIGS])
    a_.set_ylabel("recall (balanced test)"); a_.axhline(0.5, c="k", lw=0.5, ls=":"); a_.set_ylim(0, 1.05)
    a_.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.1))     # outside the axes, hides no bars
    plt.tight_layout(); plt.savefig(f"{RUN_DIR}/{tag}_exp2_recall.png", dpi=150, bbox_inches="tight")

    # Plot 3: ablation heatmaps for (500,100)
    fig, ax = plt.subplots(1, 2, figsize=(7, 4))
    for a_, weighted in zip(ax, (False, True)):
        im = a_.imshow(R[(weighted, (500, 100))]["abl"], cmap="Reds", aspect="auto", vmin=0, vmax=0.5)
        a_.set_xticks([0, 1]); a_.set_xticklabels(["A", "B"])
        a_.set_yticks(range(N_QUBITS)); a_.set_ylabel("ablated qubit")
        a_.set_title(("weighted" if weighted else "plain") + " (500,100)")
    plt.colorbar(im, ax=ax); plt.savefig(f"{RUN_DIR}/{tag}_exp2_ablation.png", dpi=150, bbox_inches="tight")
    plt.show()


def head_refit(model, xtr, ltr, xt, lt):
    """Refit ONLY a linear head on the frozen circuit outputs <Z>, using balanced data.
    If recall comes back, the circuit still separates the classes and what flipped is the head/bias."""
    with torch.no_grad():
        ztr, zt = model(xtr)[1].numpy(), model(xt)[1].numpy()
    pred = LogisticRegression(max_iter=2000).fit(ztr, ltr.numpy()).predict(zt)
    y = lt.numpy()
    return (pred[y == 0] == 0).mean(), (pred[y == 1] == 1).mean()


def exp3(args):
    """Sequential (non-IID): train on A only, then on B only. Track angles and recall.
    Dashed lines = recall after refitting only the linear head on the frozen circuit (forgetting probe)."""
    xt, lt = test_set()
    xa, la = make(300, 0, seed=0)
    xb, lb = make(0, 300, seed=0)
    xp, lp = make(200, 200, seed=7)          # balanced data used only for the head-refit probe
    torch.manual_seed(0)
    m = QNN()
    hist = []

    def snap(tag):
        aA, aB, gA, gB = evaluate(m, xt, lt)
        rA, rB = head_refit(m, xp, lp, xt, lt)
        hist.append((tag, gA.mean(0), gB.mean(0), aA, aB, rA, rB))
        print(f"{tag}: recall A={aA:.2f} B={aB:.2f} | head-refit A={rA:.2f} B={rB:.2f} | "
              f"angA={gA.mean(0).round(1)} angB={gB.mean(0).round(1)}")

    snap("init")
    train(m, xa, la, epochs=args.epochs); snap("after A")
    train(m, xb, lb, epochs=args.epochs); snap("after B")

    fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
    tags = [h[0] for h in hist]
    for q in range(N_QUBITS):
        ax[0].plot(tags, [h[1][q] for h in hist], "o-", label=f"q{q}")
        ax[1].plot(tags, [h[2][q] for h in hist], "s-", label=f"q{q}")
    ax[0].set_title("Class A mean angle"); ax[1].set_title("Class B mean angle")
    ax[2].plot(tags, [h[3] for h in hist], "o-", c="C0", label="A recall")
    ax[2].plot(tags, [h[4] for h in hist], "s-", c="C1", label="B recall")
    ax[2].plot(tags, [h[5] for h in hist], "o--", c="C0", alpha=0.6, label="A recall, head refit")
    ax[2].plot(tags, [h[6] for h in hist], "s--", c="C1", alpha=0.6, label="B recall, head refit")
    ax[2].set_title("Recall (balanced test)"); ax[2].set_ylim(-0.05, 1.05)
    for a_ in ax:
        a_.legend()
    plt.tight_layout(); plt.savefig(f"{RUN_DIR}/{S['tag']}_exp3_forgetting.png", dpi=150); plt.show()


def exp4(args):
    """Angle constraint A->67, B->23 on OBS_QUBITS for every config in --configs and every weight in --lams.
    Row 1: plain angle loss, row 2: class-balanced angle loss; one PNG per config.
    Printed per (config, loss, lam): recall (+/-SE over seeds) and, per class, the mean angle on the constrained qubits and
    miss = mean angle - target (so 'B miss +20' means class B is still 20 deg above its target)."""
    lams = args.lams
    xt, lt = test_set()
    obs = list(OBS_QUBITS)
    tgt = np.array(TARGET_DEG)
    summary = []
    for cfg in args.configs:
        fig, ax = plt.subplots(2, len(lams), figsize=(5.5 * len(lams), 8), sharey=True, squeeze=False)
        cache = {}                                                  # lam = 0: plain and balanced angle loss are the same model
        for r_, bal in enumerate((False, True)):
            for a_, lam in zip(ax[r_], lams):
                if lam == 0 and (cfg, 0.0) in cache:
                    gA, gB, rec, mA, mB = cache[(cfg, 0.0)]
                else:
                    gAs, gBs, rec, mA, mB = [], [], [], [], []
                    for s in range(args.seeds):
                        torch.manual_seed(s)
                        x, l = make(*cfg, seed=s)
                        m = train(QNN(), x, l, epochs=args.epochs, lam=lam, weighted=args.weighted, balanced_angle=bal)
                        aA, aB, gA_, gB_ = evaluate(m, xt, lt)
                        gAs.append(gA_[:, obs]); gBs.append(gB_[:, obs])
                        rec.append((aA, aB)); mA.append(gA_[:, obs].mean(0)); mB.append(gB_[:, obs].mean(0))
                    gA, gB = np.concatenate(gAs), np.concatenate(gBs)
                    rec, mA, mB = np.array(rec), np.array(mA), np.array(mB)    # [seeds, 2], [seeds, n_obs], [seeds, n_obs]
                    if lam == 0:
                        cache[(cfg, 0.0)] = (gA, gB, rec, mA, mB)
                for i, q in enumerate(OBS_QUBITS):
                    kw = dict(bins=25, alpha=0.5) if i == 0 else dict(bins=25, histtype="step", lw=2)
                    a_.hist(gA[:, i], color="C0", label=f"A q{q}", **kw)
                    a_.hist(gB[:, i], color="C1", label=f"B q{q}", **kw)
                a_.axvline(TARGET_DEG[0], c="C0", ls="--"); a_.axvline(TARGET_DEG[1], c="C1", ls="--")
                r = rec.mean(0)
                name = "balanced-angle" if bal else "plain-angle"
                a_.set_title(f"{name}, lam={lam:g}\nrecall A={r[0]:.2f} B={r[1]:.2f}")
                a_.set_xlabel("angle (deg)"); a_.legend(fontsize=8)
                missA, missB = mA.mean(0) - tgt[0], mB.mean(0) - tgt[1]
                print(f"{cfg} {name:14s} lam={lam:g}: recall A={r[0]:.2f}+/-{_sem(rec[:, 0]):.2f} B={r[1]:.2f}+/-{_sem(rec[:, 1]):.2f} | "
                      f"mean angle A={mA.mean(0).round(1)} B={mB.mean(0).round(1)} | "
                      f"miss A={missA.round(1)} B={missB.round(1)} (target A {tgt[0]:g}, B {tgt[1]:g})")
                summary.append((cfg, name, lam, r[0], r[1], missA.mean(), missB.mean()))
        plt.suptitle(f"Config A/B = {cfg}{' (weighted CE)' if args.weighted else ''}{', frozen head' if FREEZE_HEAD else ''}; "
                     f"dashed = target (A:{TARGET_DEG[0]:g}, B:{TARGET_DEG[1]:g})")
        plt.tight_layout()
        plt.savefig(f"{RUN_DIR}/{S['tag']}_exp4_constraint_{TARGET_DEG[0]:g}_{TARGET_DEG[1]:g}_cfg{cfg[0]}x{cfg[1]}{'_w' if args.weighted else ''}.png", dpi=150)
        plt.show()
        plt.close(fig)
    lam_hi = max(lams)
    print(f"\nsummary at the largest lam ({lam_hi:g}); miss = mean angle - target, averaged over the constrained qubits "
          f"(0 = target reached):")
    for cfg, name, lam, rA, rB, mA_, mB_ in summary:
        if lam == lam_hi:
            print(f"  {cfg} {name:14s}: recall A={rA:.2f} B={rB:.2f} | miss A={mA_:+.1f} B={mB_:+.1f} deg")


def exp5(args, every=5):
    """Sequential imbalance (non-IID): train on phase 1 distribution, then phase 2, then phase 3 ... (--phases).
    Tracks class-wise recall and per-qubit class-wise angle during training. Plain vs weighted CE."""
    phases = args.phases
    xt, lt = test_set()
    n_it = max(args.epochs // every, 1)
    steps = [0] + [every * (k + 1) for k in range(n_it * len(phases))]
    R = {}
    for weighted in (False, True):
        rec, aA_, aB_ = [], [], []
        for s in range(args.seeds):
            torch.manual_seed(s)
            m = QNN()
            snaps = []

            def snap():
                a, b, gA, gB = evaluate(m, xt, lt)
                snaps.append((a, b, gA.mean(0), gB.mean(0)))

            snap()
            for ph, cfg in enumerate(phases):
                x, l = make(*cfg, seed=s + 10 * ph)
                opt = torch.optim.Adam(m.parameters(), lr=LR)
                for _ in range(n_it):
                    train(m, x, l, epochs=every, weighted=weighted, opt=opt)
                    snap()
            rec.append([(a, b) for a, b, _, _ in snaps])
            aA_.append([ga for _, _, ga, _ in snaps]); aB_.append([gb for _, _, _, gb in snaps])
        R[weighted] = tuple((np.array(v).mean(0), np.array(v).std(0)) for v in (rec, aA_, aB_))
        r = R[weighted][0][0]
        msg = " | ".join(f"after phase{k + 1} {phases[k]}: A={r[n_it * (k + 1)][0]:.2f} B={r[n_it * (k + 1)][1]:.2f}"
                         for k in range(len(phases)))
        print(f"{'weighted' if weighted else 'plain   '} {msg}")

    fig, ax = plt.subplots(2, 3, figsize=(16, 7))
    st = np.array(steps)
    for r_, weighted in enumerate((False, True)):
        (rec, rec_s), (aA, aA_s), (aB, aB_s) = R[weighted]
        name = "weighted" if weighted else "plain"
        for k, (lab_, mk) in enumerate((("A recall", "o"), ("B recall", "s"))):
            ax[r_][0].plot(st, rec[:, k], mk + "-", label=lab_)
            ax[r_][0].fill_between(st, rec[:, k] - rec_s[:, k], rec[:, k] + rec_s[:, k], alpha=0.2)
        ax[r_][0].set_ylim(-0.05, 1.05); ax[r_][0].set_title(f"{name}: recall (balanced test), mean +/- std")
        for q in range(N_QUBITS):
            ax[r_][1].plot(st, aA[:, q], "o-", label=f"q{q}")
            ax[r_][1].fill_between(st, aA[:, q] - aA_s[:, q], aA[:, q] + aA_s[:, q], alpha=0.12)
            ax[r_][2].plot(st, aB[:, q], "s-", label=f"q{q}")
            ax[r_][2].fill_between(st, aB[:, q] - aB_s[:, q], aB[:, q] + aB_s[:, q], alpha=0.12)
        ax[r_][1].set_title(f"{name}: class A mean angle (deg)"); ax[r_][2].set_title(f"{name}: class B mean angle (deg)")
        for a_ in ax[r_]:
            for k in range(1, len(phases)):                   # dotted line = phase boundary
                a_.axvline(steps[n_it * k], c="k", ls=":")
            a_.set_xlabel("epoch"); a_.legend(fontsize=8)
    plt.suptitle("   |   ".join(f"phase {k + 1} {p}" for k, p in enumerate(phases)) + "   (A,B counts)")
    plt.tight_layout()
    plt.savefig(f"{RUN_DIR}/{S['tag']}_exp5_sequential_{'_'.join(f'{a}-{b}' for a, b in phases)}.png", dpi=150)
    plt.show()


def exp6(args, focus=(500, 100)):
    """How much does each qubit rotate under imbalance, and can weighting undo it?
    Reference = balanced 200/200 model with the SAME init (per seed). For every imbalance config:
      delta[class, qubit] = mean output angle (imbalanced) - mean output angle (balanced reference)
    The 200/200 column must be exactly 0 (same data, same init) -> sanity check, NOT a noise floor.
    Noise floor = balanced models (200/200 and 300/300) trained on a NEW random draw with the same init:
    any shift below that is just resampling/optimisation noise, not an imbalance effect.
    v3: the reference only sees per-class indices REF_POOL; controls and imbalanced configs only see OTHER_POOL (own seeds),
    so nothing is nested in the reference and controls/imbalanced runs are compared like for like. Error bars = SE over seeds.
    Also reports the last layer's net weight rotation (class-independent) and the input RY angle per class."""
    xt, lt = test_set()
    in_ang = np.stack([torch.rad2deg(xt[lt == c]).mean(0).numpy() for c in (0, 1)])   # [2, Q] data-side rotation

    ref = {}
    for s in range(args.seeds):
        torch.manual_seed(s)
        x, l = cfg_data((200, 200), s)                                                # REF_POOL only
        m = train(QNN(), x, l, epochs=args.epochs)
        ang, aA, aB = mean_angles(m, xt, lt)
        ref[s] = dict(ang=ang, rot=net_rot_deg(m.w[-1].detach()).numpy(), rec=(aA, aB))
    ref_rot = np.array([ref[s]["rot"] for s in ref])                                  # [seeds, Q]
    print(f"balanced reference: recall A={np.mean([ref[s]['rec'][0] for s in ref]):.2f} "
          f"B={np.mean([ref[s]['rec'][1] for s in ref]):.2f}")
    print(f"input RY angle (deg) per qubit  A={in_ang[0].round(1)}  B={in_ang[1].round(1)}")

    CTRL = [(200, 200), (300, 300)]        # balanced controls, new data draw, same init -> resampling noise floor
    ctrl, ctrlD = {}, {}
    for k_, cc in enumerate(CTRL):
        Dc = []
        for s in range(args.seeds):
            torch.manual_seed(s)
            x, l = make(*cc, seed=s + 1000 * (k_ + 1), pool=OTHER_POOL)              # own seed per control, never the reference samples
            m = train(QNN(), x, l, epochs=args.epochs)
            Dc.append(mean_angles(m, xt, lt)[0] - ref[s]["ang"])
        ctrlD[cc] = np.array(Dc)                                                      # [seeds, 2, Q] SIGNED shifts (v2)
        ctrl[cc] = np.abs(ctrlD[cc]).mean((1, 2))                                     # [seeds]
        print(f"NOISE FLOOR balanced {cc}, new draw: mean|d angle|={ctrl[cc].mean():5.1f} (SE {_sem(ctrl[cc]):.1f}) deg")
        mu_c = ctrlD[cc].mean(0)
        se_c = f"{_sem(ctrlD[cc]).mean():.1f}" if args.seeds > 1 else "n/a"
        print(f"      signed control shift (mean over seeds)  A={mu_c[0].round(1)}  B={mu_c[1].round(1)}  (SE ~{se_c} deg per cell)")

    R = {}
    for weighted in (False, True):
        for cfg in CONFIGS:
            D, REC, ROT, MOVE = [], [], [], []
            for s in range(args.seeds):
                torch.manual_seed(s)
                x, l = cfg_data(cfg, s)                                               # independent of the reference (except (200,200) = reference)
                m = train(QNN(), x, l, epochs=args.epochs, weighted=weighted)
                ang, aA, aB = mean_angles(m, xt, lt)
                D.append(ang - ref[s]["ang"]); REC.append((aA, aB))
                ROT.append(net_rot_deg(m.w[-1].detach()).numpy())
                MOVE.append(np.rad2deg((m.w - m.w0).detach().abs().mean().item()))
            R[(weighted, cfg)] = d = dict(D=np.array(D), rec=np.array(REC), rot=np.array(ROT), move=np.array(MOVE))
            shift = np.abs(d["D"]).mean((1, 2))
            gap = np.abs(d["rec"][:, 0] - d["rec"][:, 1])
            print(f"{'weighted' if weighted else 'plain   '} {cfg}: mean|d angle|={shift.mean():5.1f} (SE {_sem(shift):4.1f}) deg"
                  f" | recall A={d['rec'][:, 0].mean():.2f} B={d['rec'][:, 1].mean():.2f} gap={gap.mean():.2f}"
                  f" | weights moved {d['move'].mean():.1f} deg")
            print(f"      d angle A={d['D'].mean(0)[0].round(1)}  B={d['D'].mean(0)[1].round(1)}  (per qubit, deg)")
    print("\nplain vs weighted vs noise floor (floor = balanced control with the closest total sample count)\n"
          "  excess = (config - control) paired by seed/init, mean [95% t-interval]; an interval containing 0 = not distinguishable from noise")
    f2 = lambda t: f"{t[0]:+5.1f} [{t[1]:+5.1f}, {t[2]:+5.1f}]"
    for cfg in CONFIGS[1:]:
        sp, sw = (np.abs(R[(w_, cfg)]["D"]).mean((1, 2)) for w_ in (False, True))        # [seeds] each
        cc = min(CTRL, key=lambda c: abs(sum(c) - sum(cfg)))
        fl = ctrl[cc]
        print(f"{cfg}: plain {sp.mean():5.1f} | weighted {sw.mean():5.1f} | floor{cc} {fl.mean():5.1f} deg\n"
              f"      excess over floor: plain {f2(ci95(sp - fl))} | weighted {f2(ci95(sw - fl))}\n"
              f"      plain - weighted: {f2(ci95(sp - sw))}  (weighted lower in {int((sw < sp).sum())}/{len(sp)} seeds)")

    beyond = report_beyond_noise(R, ctrlD, CONFIGS[1:])

    tag = S["tag"]
    # Plot 1: signed angle shift vs balanced, rows = plain/weighted, cols = config, cells = [class x qubit]
    lim = max(10.0, max(np.abs(v["D"].mean(0)).max() for v in R.values()))
    fig, ax = plt.subplots(2, len(CONFIGS), figsize=(3.4 * len(CONFIGS), 5.8), sharey=True,
                           gridspec_kw={"hspace": 0.55})
    for r_, weighted in enumerate((False, True)):
        for a_, cfg in zip(ax[r_], CONFIGS):
            M = R[(weighted, cfg)]["D"].mean(0)                                       # [2, Q]
            im = a_.imshow(M, cmap="coolwarm", vmin=-lim, vmax=lim, aspect="auto")
            for i in range(2):
                for q in range(N_QUBITS):
                    a_.text(q, i, f"{M[i, q]:+.1f}", ha="center", va="center", fontsize=8)
            a_.set_xticks(range(N_QUBITS)); a_.set_xlabel("qubit")
            a_.set_yticks([0, 1]); a_.set_yticklabels(["class A", "class B"])
            a_.set_title(f"{'weighted' if weighted else 'plain'} {cfg}", fontsize=10)
    fig.colorbar(im, ax=ax.ravel().tolist(), shrink=0.8, label="angle shift vs balanced model (deg)")
    plt.savefig(f"{RUN_DIR}/{tag}_exp6_rotation_shift.png", dpi=150, bbox_inches="tight")
    plot_beyond_noise(beyond, ctrlD, CONFIGS[1:], f"{RUN_DIR}/{tag}_exp6_beyond_noise.png")

    # Plot 2: summary - rotation shift, recall gap, last-layer weight rotation, input angle
    fig, ax = plt.subplots(1, 4, figsize=(21, 4.2))
    xs = np.arange(len(CONFIGS)); lab_cfg = [f"{c[0]}/{c[1]}" for c in CONFIGS]
    for k, (weighted, name, col) in enumerate(((False, "plain", "C3"), (True, "weighted", "C2"))):
        off = (k - 0.5) * 0.38
        sh = [np.abs(R[(weighted, c)]["D"]).mean((1, 2)) for c in CONFIGS]
        gp = [np.abs(R[(weighted, c)]["rec"][:, 0] - R[(weighted, c)]["rec"][:, 1]) for c in CONFIGS]
        ax[0].bar(xs + off, [v.mean() for v in sh], 0.38, yerr=[_sem(v) for v in sh], capsize=2, color=col, label=name)
        ax[1].bar(xs + off, [v.mean() for v in gp], 0.38, yerr=[_sem(v) for v in gp], capsize=2, color=col, label=name)
    ax[0].axhline(ctrl[(200, 200)].mean(), c="k", ls="--", lw=1.2, label="noise floor 200/200 (new draw)")
    ax[0].axhline(ctrl[(300, 300)].mean(), c="gray", ls=":", lw=1.5, label="noise floor 300/300 (new draw)")
    for a_, ttl, yl in ((ax[0], "qubit rotation caused by imbalance (bars: SE)", "mean |angle shift| vs balanced (deg)"),
                        (ax[1], "recall gap between classes (bars: SE)", "|recall A - recall B|")):
        a_.set_xticks(xs); a_.set_xticklabels(lab_cfg); a_.set_title(ttl); a_.set_ylabel(yl); a_.legend()
    ax[0].set_ylim(0, ax[0].get_ylim()[1] * 1.45)                                    # headroom: legend hides no bars
    ax[0].legend(ncol=2, fontsize=7, loc="upper center")
    qs = np.arange(N_QUBITS)
    for k, (name, arr, col) in enumerate((("balanced", ref_rot, "C7"),
                                          ("plain", R[(False, focus)]["rot"], "C3"),
                                          ("weighted", R[(True, focus)]["rot"], "C2"))):
        ax[2].bar(qs + (k - 1) * 0.27, arr.mean(0), 0.27, yerr=_sem(arr), capsize=2, color=col, label=name)
    ax[2].set_xticks(qs); ax[2].set_xlabel("qubit"); ax[2].set_ylabel("net rotation of last-layer gate (deg)")
    ax[2].set_title(f"trained weight rotation, last layer, {focus} (bars: SE)"); ax[2].legend()
    ax[3].bar(qs - 0.2, in_ang[0], 0.4, color="C0", label="class A"); ax[3].bar(qs + 0.2, in_ang[1], 0.4, color="C1", label="class B")
    ax[3].set_xticks(qs); ax[3].set_xlabel("qubit"); ax[3].set_ylabel("mean input RY angle (deg)")
    ax[3].set_title("data-side rotation (test set, not trained)")
    ax[3].set_ylim(0, 135); ax[3].legend(ncol=2, loc="upper center")                  # headroom so the legend hides no bars
    plt.tight_layout(); plt.savefig(f"{RUN_DIR}/{tag}_exp6_summary.png", dpi=150); plt.show()


def exp7(args, focus=(500, 100)):
    """Exact per-qubit, per-class contribution to the classifier's decision (plain vs weighted CE, all CONFIGS).
    The head is linear in <Z>, so the margin m = logit_B - logit_A splits EXACTLY into one term per qubit plus a bias:
        m(x) = sum_q (W[B,q] - W[A,q]) * <Z_q>(x) + (b[B] - b[A])
    Per class we report how much each qubit (and the bias) pushes toward the TRUE class, averaged over the balanced test set.
    The bias is the prior: under plain loss it favours the majority class, which is how imbalance enters the decision.
    D_q = mean_B(c_q) - mean_A(c_q) is the class separation qubit q carries (the bias cancels in it)."""
    xt, lt = test_set()
    R = {}
    for weighted in (False, True):
        name = "weighted" if weighted else "plain   "
        for cfg in CONFIGS:
            T, Bz, Dd, E, rec = [], [], [], [], []
            for s in range(args.seeds):
                torch.manual_seed(s)
                x, l = cfg_data(cfg, s)                                   # balanced = REF_POOL, imbalanced = independent draws from OTHER_POOL
                m = train(QNN(), x, l, epochs=args.epochs, weighted=weighted)
                t_, b_, d_, e_ = qubit_contrib(m, xt, lt)
                aA, aB = evaluate(m, xt, lt)[:2]
                T.append(t_); Bz.append(b_); Dd.append(d_); E.append(e_); rec.append((aA, aB))
            R[(weighted, cfg)] = r = dict(toward=np.array(T), bias=np.array(Bz), disc=np.array(Dd),
                                          rec=np.array(rec), err=max(E))
            tw, bi, di, rc = r["toward"].mean(0), r["bias"].mean(0), r["disc"].mean(0), r["rec"].mean(0)
            print(f"{name} {cfg}: recall A={rc[0]:.2f} B={rc[1]:.2f}")
            print(f"      push toward TRUE class  A: q={tw[0].round(2)} | B: q={tw[1].round(2)} | bias term b_B-b_A={bi[1]:+.2f} (+ favours B)")
            print(f"      separation per qubit D={di.round(2)} | top qubit per seed {r['disc'].argmax(1).tolist()}")
    print(f"\ndecomposition check: max |sum of parts - real margin| over all models = {max(r['err'] for r in R.values()):.1e} (should be ~1e-6)")
    if HEAD_BIAS:
        print("\nhow much of the prior log-odds log(nB/nA) does the bias absorb?\n"
              "  = plain bias minus weighted bias of the SAME config (same data, same init, paired by seed); weighted ~ the prior-free model.\n"
              "  (not measured against the balanced reference: its 300-image pool makes that baseline unreliable)")
        for cfg in CONFIGS[1:]:
            sh = R[(False, cfg)]["bias"][:, 1] - R[(True, cfg)]["bias"][:, 1]
            pr = np.log(cfg[1] / cfg[0])
            mu, lo, hi = ci95(sh)
            print(f"  {cfg}: bias shift {mu:+.2f} [95% CI {lo:+.2f}, {hi:+.2f}] | log prior ratio {pr:+.2f} | fraction absorbed {mu / pr:.2f}")
    else:
        print("\n--no_bias: the head has no bias term, so the bias column/plot is identically 0; look at the qubit columns instead.")
    plot_exp7(R, CONFIGS, focus, f"{RUN_DIR}/{S['tag']}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["pairs", "quick", "exp2", "exp3", "exp4", "exp5", "exp6", "exp7"])
    p.add_argument("--a", type=int, default=0)
    p.add_argument("--b", type=int, default=1)
    p.add_argument("--features", choices=["resnet", "pixels"], default="resnet")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--target_a", type=float, default=67.0, help="target angle (deg) for class A")
    p.add_argument("--target_b", type=float, default=23.0, help="target angle (deg) for class B")
    p.add_argument("--weighted", action="store_true", help="exp4 only: use class-weighted CE (exp2, exp5, exp6 always run both)")
    p.add_argument("--phases", type=parse_phases, default=parse_phases("500,100;100,500"),
                   help='exp5 only: "A,B;A,B;..." sample counts per phase, e.g. "500,100;100,500;500,200"')
    p.add_argument("--no_bias", action="store_true", help="head = Linear(4,2,bias=False): the prior cannot hide in a bias term, so the circuit has to absorb the imbalance")
    p.add_argument("--freeze_head", action="store_true", help="head fixed to logit_A=-sum(z), logit_B=+sum(z), no bias, not trained: only the circuit can absorb imbalance")
    p.add_argument("--lams", type=parse_lams, default=(0.0, 1.0, 5.0), help='exp4 only: angle-loss weights, e.g. "0,5,20,100"')
    p.add_argument("--configs", type=parse_cfgs, default=[(500, 100)], help='exp4 only: "all" or "A,B;A,B;..." (default 500,100)')
    p.add_argument("--out", default=OUT, help="parent folder; every run creates its own NEW subfolder in it (default: results)")
    args = p.parse_args()
    TARGET_DEG = (args.target_a, args.target_b)   # module-level global read by train() and exp4()
    HEAD_BIAS = not args.no_bias                  # module-level global read by QNN()
    FREEZE_HEAD = args.freeze_head                # module-level global read by QNN()

    # every run -> its own new folder with all PNGs + log.txt; an existing folder/file is never overwritten
    ep, sd = (15, 1) if args.mode == "quick" else (args.epochs, args.seeds)
    run_name = f"pairs_{args.features}" if args.mode == "pairs" else f"{args.a}v{args.b}_{args.features}_{args.mode}"
    run_name += ("_nobias" if args.no_bias else "") + ("_frozen" if args.freeze_head else "")
    RUN_DIR = run_dir_for(run_name, ep, sd, out=args.out)
    _log = open(f"{RUN_DIR}/log.txt", "w", buffering=1)      # line-buffered: log.txt fills as the run goes
    sys.stdout, sys.stderr = Tee(sys.__stdout__, _log), Tee(sys.__stderr__, _log)
    print("command:", " ".join(sys.argv))
    print("args   :", vars(args))
    print(f"outputs and log -> {RUN_DIR}/")

    if args.mode == "pairs":
        pairs(args)
    else:
        setup(args.a, args.b, args.features)
        if args.mode == "quick":
            import time
            t = time.time()
            exp2(args, seeds=1, epochs=15)
            print(f"quick run took {time.time() - t:.1f}s")
        elif args.mode == "exp2":
            exp2(args)
        elif args.mode == "exp3":
            exp3(args)
        elif args.mode == "exp4":
            exp4(args)
        elif args.mode == "exp5":
            exp5(args)
        elif args.mode == "exp6":
            exp6(args)
        elif args.mode == "exp7":
            exp7(args)

    print(f"\nDone. Everything from this run (all PNGs + log.txt) is in {RUN_DIR}/   (index of all runs: {args.out}/runs_index.txt)")