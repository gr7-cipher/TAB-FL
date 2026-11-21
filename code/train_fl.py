"""Federated experiment runner for TAB-FL and the baselines.

The paper's grid (about a thousand 50-round runs) is executed with a
deterministic single-process simulator: every round, each of the K clients
starts from the current global weights, runs E local SGD steps on its own
shard, and reports its update; the server applies one rule from
``aggregators.py``.  The same rules are exposed as Flower strategies through
``aggregators.FlowerRobustStrategy``; ``--check-flower`` verifies that the
Flower path and the simulator produce identical global models.

Examples
--------
python train_fl.py --one --method TAB-FL --attack sign_flip --rho 0.3
python train_fl.py --grid main --workers 6
python train_fl.py --collect          # merge per-run JSON into results/*.csv
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import time
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

import aggregators as A
import attacks as atk
from data_loader import device_partition, dirichlet_partition, load_processed, root_split
from models import (CNNBiGRU, batched_forward, count_params, get_flat, layer_slices, set_flat,
                    unflatten_batch)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("TABFL_DATA", os.path.join(HERE, "..", "data", "processed"))
RUN_DIR = os.environ.get("TABFL_RUNDIR", os.path.join(HERE, "..", "results", "runs"))
DEVICE = os.environ.get("TABFL_DEVICE", "cpu")   # "mps" runs local training on the Apple GPU

DEFAULTS = dict(dataset="ciciot2023", method="TAB-FL", attack="none", rho=0.0, beta=0.3,
                seed=0, rounds=50, K=20, local_steps=10, batch=32, lr=0.005, momentum=0.9,
                mu=0.01, opt="adam", onset=0, root_per_class=30, eval_subset=8000, log_trust=False)

METHODS = ["FedAvg", "FedProx", "Krum", "Multi-Krum", "Median", "Trimmed Mean", "TAB-FL"]
ABLATIONS = ["TAB-FL", "TAB-FL w/o EMA", "TAB-FL w/o clipping", "TAB-FL w/o validation",
             "TAB-FL w/o geometry"]
ATTACKS = ["label_flip", "sign_flip", "gaussian", "backdoor"]


def run_id(cfg: dict) -> str:
    m = cfg["method"].replace(" ", "_").replace("/", "")
    return (f"{cfg['dataset']}_{m}_{cfg['attack']}_r{int(round(cfg['rho'] * 100))}"
            f"_b{cfg['beta']}_on{cfg['onset']}_s{cfg['seed']}")


def _metrics(y_true, y_pred, benign=0):
    att = y_true != benign
    dr = float(np.mean(y_pred[att] != benign)) if att.any() else float("nan")
    fpr = float(np.mean(y_pred[~att] != benign)) if (~att).any() else float("nan")
    return dict(accuracy=float(accuracy_score(y_true, y_pred)),
                macro_f1=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
                detection_rate=dr, fpr=fpr)


@torch.no_grad()
def predict(model, X, bs=4096):
    model.eval()
    out = [model(torch.from_numpy(X[i:i + bs])).argmax(1).numpy() for i in range(0, len(X), bs)]
    return np.concatenate(out)


def local_train(model, X, y, steps, batch, lr, momentum, rng, prox_mu=0.0, w_global=None,
                opt="sgd"):
    model.train()
    opt = (torch.optim.Adam(model.parameters(), lr=lr) if opt == "adam"
           else torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum))
    n = len(y)
    for _ in range(steps):
        idx = rng.integers(0, n, size=min(batch, n))
        xb = torch.from_numpy(X[idx])
        yb = torch.from_numpy(y[idx])
        loss = F.cross_entropy(model(xb), yb)
        if prox_mu > 0:
            loss = loss + 0.5 * prox_mu * (torch.nn.utils.parameters_to_vector(model.parameters())
                                           - w_global).pow(2).sum()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()


def train_clients(model, w, data, steps, batch, lr, momentum, rng, mu=0.0, clip=5.0, opt="sgd",
                  device="cpu"):
    """E local SGD-momentum steps on every client at once; returns (K, d) local weights.

    The K client models are stacked and run through ``batched_forward``; since
    their parameters never interact, the gradient of the summed loss w.r.t.
    row k equals client k's own gradient.  Optimiser and per-client gradient
    clipping match ``local_train``.
    """
    K = len(data)
    w0 = w.to(device)
    W = w0.unsqueeze(0).repeat(K, 1).clone()
    buf = torch.zeros_like(W)
    sq = torch.zeros_like(W)
    for step in range(1, steps + 1):
        xs, ys = [], []
        for X, y in data:
            idx = rng.integers(0, len(y), size=batch)
            xs.append(X[idx])
            ys.append(y[idx])
        XB = torch.from_numpy(np.stack(xs)).to(device)
        YB = torch.from_numpy(np.stack(ys)).to(device)
        Wv = W.requires_grad_(True)
        logits = batched_forward(unflatten_batch(model, Wv), XB)
        C = logits.shape[-1]
        loss = F.cross_entropy(logits.reshape(-1, C), YB.reshape(-1), reduction="sum") / batch
        if mu > 0:
            loss = loss + 0.5 * mu * (Wv - w0).pow(2).sum()
        G, = torch.autograd.grad(loss, Wv)
        W = W.detach()
        sc = torch.clamp(clip / (G.norm(dim=1, keepdim=True) + 1e-12), max=1.0)
        G = G * sc
        if opt == "adam":   # Adam(beta1=0.9, beta2=0.999), state reset every round
            buf = 0.9 * buf + 0.1 * G
            sq = 0.999 * sq + 0.001 * G * G
            W = W - lr * (buf / (1 - 0.9 ** step)) / (torch.sqrt(sq / (1 - 0.999 ** step)) + 1e-8)
        else:
            buf = momentum * buf + G
            W = W - lr * buf
    return W.detach().cpu()


def make_root_losses(model, Xroot, yroot, n_classes):
    """Per-class mean loss of every candidate model on the root set: (K, C)."""
    onehot = F.one_hot(yroot, n_classes).float()
    counts = onehot.sum(0).clamp(min=1)

    def root_losses(Wc):
        with torch.no_grad():
            K = Wc.shape[0]
            logits = batched_forward(unflatten_batch(model, Wc), Xroot.unsqueeze(0).expand(K, -1, -1))
            N, C = logits.shape[1], logits.shape[2]
            l = F.cross_entropy(logits.reshape(K * N, C), yroot.repeat(K), reduction="none")
            return ((l.reshape(K, N) @ onehot) / counts).numpy()
    return root_losses


def make_root_record_losses(model, Xroot, yroot):
    """Per-record loss of every candidate model on the root set: (K, N)."""
    def record_losses(Wc):
        with torch.no_grad():
            K = Wc.shape[0]
            logits = batched_forward(unflatten_batch(model, Wc), Xroot.unsqueeze(0).expand(K, -1, -1))
            N, C = logits.shape[1], logits.shape[2]
            l = F.cross_entropy(logits.reshape(K * N, C), yroot.repeat(K), reduction="none")
            return l.reshape(K, N).numpy()
    return record_losses


def setup(cfg):
    """Load data, partition it, choose malicious clients, build the model."""
    torch.set_num_threads(1)
    d = load_processed(os.path.join(DATA_DIR, f"{cfg['dataset']}.npz"))
    Xtr, ytr, gtr = d["X_train"], d["y_train"], d["g_train"]
    Xte, yte = d["X_test"], d["y_test"]
    if os.environ.get("TABFL_TUNE"):
        # design / hyper-parameter selection mode: the test set is never touched;
        # 15% of the training pool is held out and used as the evaluation set
        vr = np.random.default_rng(99)
        perm = vr.permutation(len(ytr))
        nv = int(0.15 * len(ytr))
        Xte, yte = Xtr[perm[:nv]], ytr[perm[:nv]]
        Xtr, ytr, gtr = Xtr[perm[nv:]], ytr[perm[nv:]], gtr[perm[nv:]]
    n_classes = len(d["class_names"])
    root, rest = root_split(ytr, cfg["root_per_class"], seed=0)
    if cfg["dataset"] == "toniot":
        parts = device_partition(gtr[rest], cfg["K"], seed=cfg["seed"])
    else:
        parts = dirichlet_partition(ytr[rest], cfg["K"], cfg["beta"], seed=cfg["seed"])
    parts = [rest[p] for p in parts]
    # gateway roles known to the operator (device type served); none for the Dirichlet split
    groups = (np.array([int(np.bincount(gtr[p]).argmax()) for p in parts])
              if cfg["dataset"] == "toniot" else None)
    rng = np.random.default_rng(1000 + cfg["seed"])
    n_mal = int(round(cfg["rho"] * cfg["K"]))
    mal = np.zeros(cfg["K"], dtype=bool)
    mal[rng.choice(cfg["K"], n_mal, replace=False)] = True
    ev_rng = np.random.default_rng(5)
    ev_idx = ev_rng.choice(len(yte), min(cfg["eval_subset"], len(yte)), replace=False)
    torch.manual_seed(cfg["seed"])
    model = CNNBiGRU(Xtr.shape[1], n_classes)
    return dict(d=d, Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte, n_classes=n_classes, root=root,
                parts=parts, mal=mal, ev_idx=ev_idx, model=model, groups=groups,
                root_groups=gtr[root])


def run_experiment(cfg: dict) -> dict:
    cfg = {**DEFAULTS, **cfg}
    S = setup(cfg)
    Xtr, ytr, Xte, yte = S["Xtr"], S["ytr"], S["Xte"], S["yte"]
    C, K, mal = S["n_classes"], cfg["K"], S["mal"]
    model = S["model"]
    slices = layer_slices(model)
    w = get_flat(model)
    d_params = w.numel()
    rng = np.random.default_rng(cfg["seed"])
    gen = torch.Generator().manual_seed(cfg["seed"])
    trig = atk.trigger_features(Xtr.shape[1])

    # client datasets (poisoned copies are built once for data-level attacks)
    clean = [(Xtr[p], ytr[p]) for p in S["parts"]]
    poisoned = []
    for k, (Xk, yk) in enumerate(clean):
        if mal[k] and cfg["attack"] == "label_flip":
            poisoned.append((Xk, atk.label_flip(yk, C)))
        elif mal[k] and cfg["attack"] == "backdoor":
            poisoned.append(atk.backdoor_poison(Xk, yk, 0, np.random.default_rng(cfg["seed"] + k)))
        else:
            poisoned.append((Xk, yk))
    n = torch.tensor([float(len(c[1])) for c in clean])

    Xroot = torch.from_numpy(Xtr[S["root"]])
    yroot = torch.from_numpy(ytr[S["root"]])
    probe = CNNBiGRU(Xtr.shape[1], C)

    method = cfg["method"]
    agg = None if method == "Centralized" else A.build(method)
    if agg is not None:
        agg.reset(K)
    f_assumed = max(2, int(round(cfg["rho"] * K)))
    rl = make_root_losses(probe, Xroot, yroot, C)
    rrl = make_root_record_losses(probe, Xroot, yroot)
    ctx = {"f": f_assumed, "slices": slices,
           "root_losses": lambda U_: rl(torch.cat([w[None], w + U_])),
           "root_record_losses": lambda U_: rrl(torch.cat([w[None], w + U_])),
           "groups": S["groups"], "root_groups": S["root_groups"], "root_y": ytr[S["root"]]}

    hist = {"round": [], "macro_f1": [], "accuracy": [], "agg_ms": []}
    trust_log, sel_log = [], []
    Xev, yev = Xte[S["ev_idx"]], yte[S["ev_idx"]]

    if method == "Centralized":
        Xall = np.concatenate([c[0] for c in clean])
        yall = np.concatenate([c[1] for c in clean])
        for r in range(1, cfg["rounds"] + 1):
            local_train(model, Xall, yall, K * cfg["local_steps"], cfg["batch"], cfg["lr"],
                        cfg["momentum"], rng, opt=cfg["opt"])
            p = predict(model, Xev)
            hist["round"].append(r)
            hist["macro_f1"].append(_metrics(yev, p)["macro_f1"])
            hist["accuracy"].append(_metrics(yev, p)["accuracy"])
            hist["agg_ms"].append(0.0)
        w = get_flat(model)
    else:
        for r in range(1, cfg["rounds"] + 1):
            active = r >= cfg["onset"] if cfg["onset"] > 0 else True
            attacking = mal & active & (cfg["attack"] != "none")
            data = [poisoned[k] if attacking[k] else clean[k] for k in range(K)]
            Wl = train_clients(model, w, data, cfg["local_steps"], cfg["batch"], cfg["lr"],
                               cfg["momentum"], rng, mu=cfg["mu"] if method == "FedProx" else 0.0,
                               opt=cfg["opt"], device=DEVICE)
            U = Wl - w
            for k in np.flatnonzero(attacking):
                if cfg["attack"] == "sign_flip":
                    U[k] = atk.sign_flip(U[k])
                elif cfg["attack"] == "gaussian":
                    U[k] = atk.gaussian(U[k], gen)
                elif cfg["attack"] == "backdoor":
                    U[k] = atk.boost(U[k])
            t0 = time.perf_counter()
            delta, info = agg(U, n, ctx)
            agg_ms = 1000 * (time.perf_counter() - t0)
            w = w + delta
            set_flat(model, w)
            p = predict(model, Xev)
            m = _metrics(yev, p)
            hist["round"].append(r)
            hist["macro_f1"].append(m["macro_f1"])
            hist["accuracy"].append(m["accuracy"])
            hist["agg_ms"].append(agg_ms)
            if "trust" in info and cfg["log_trust"]:
                trust_log.append(info["trust"].round(4).tolist())
            if "selected" in info:
                sel_log.append(info["selected"])
        set_flat(model, w)

    # final evaluation on the full test set
    p = predict(model, Xte)
    final = _metrics(yte, p)
    att_idx = np.flatnonzero(yte != 0)
    p_trig = predict(model, atk.apply_trigger(Xte[att_idx], trig))
    final["asr"] = float(np.mean(p_trig == 0))  # triggered attack traffic labelled benign
    final["per_class_f1"] = f1_score(yte, p, average=None, labels=list(range(C)),
                                     zero_division=0).round(4).tolist()
    # honest-client exclusion rate (false quarantine) for selection-based rules
    if sel_log:
        sel_mask = np.zeros((len(sel_log), K), dtype=bool)
        for i, s in enumerate(sel_log):
            sel_mask[i, s] = True
        final["honest_excluded"] = float(1 - sel_mask[:, ~mal].mean()) if (~mal).any() else 0.0
        final["malicious_admitted"] = float(sel_mask[:, mal].mean()) if mal.any() else float("nan")
    out = dict(cfg=cfg, run_id=run_id(cfg), final=final, history=hist,
               malicious=np.flatnonzero(mal).tolist(), n_params=count_params(model),
               client_sizes=[len(c[1]) for c in clean],
               client_label_hist=[np.bincount(c[1], minlength=C).tolist() for c in clean])
    if trust_log:
        out["trust"] = trust_log
    if cfg.get("return_weights"):
        out["weights"] = w.numpy().tolist()
    return out


def _worker(cfg):
    path = os.path.join(RUN_DIR, run_id({**DEFAULTS, **cfg}) + ".json")
    if os.path.exists(path):
        return path, 0.0
    try:  # claim the run so that concurrent CPU / GPU runners never duplicate work
        fd = os.open(path + ".lock", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
    except FileExistsError:
        return path, -1.0
    t0 = time.time()
    res = run_experiment(cfg)
    res["wall_s"] = time.time() - t0
    res["device"] = DEVICE
    with open(path + ".tmp", "w") as fh:
        json.dump(res, fh)
    os.replace(path + ".tmp", path)
    os.remove(path + ".lock")
    return path, res["wall_s"]


def grid(name: str) -> list[dict]:
    """Experiment grid used in the paper.

    main     CICIoT2023, beta = 0.3: every rule x 4 attacks x rho in {10..40}%
             (3 seeds; rho = 10% uses seed 0 only) plus clean runs
    dynamic  attack switched on at round 25, rho = 30%, 3 seeds
    noniid   beta in {0.1, 0.5, 1.0}, clean + 4 attacks at rho = 30%, seed 0
    ablation TAB-FL variants, static and dynamic attacks at rho = 30%, 3 seeds
    toniot   TON_IoT telemetry, device partition, rho in {20, 40}%, 3 seeds
             (TAB-FL uses the gateways' device types as peer groups; the
             'w/o peer groups' ablation scores against the whole population)
    """
    seeds = [0, 1, 2]
    G = []
    if name in ("main", "all"):
        for s in seeds:
            G.append(dict(method="Centralized", seed=s))
            for m in METHODS:
                G.append(dict(method=m, seed=s))
                for a, rho in itertools.product(ATTACKS, [0.1, 0.2, 0.3, 0.4]):
                    if rho == 0.1 and s != 0:
                        continue
                    G.append(dict(method=m, attack=a, rho=rho, seed=s,
                                  log_trust=(m == "TAB-FL")))
    if name in ("dynamic", "all"):
        for s, m, a in itertools.product(seeds, METHODS, ATTACKS):
            G.append(dict(method=m, attack=a, rho=0.3, onset=25, seed=s,
                          log_trust=(m == "TAB-FL")))
    if name in ("toniot", "all"):
        for s in seeds:
            G.append(dict(dataset="toniot", method="Centralized", seed=s))
            for m in METHODS + ["TAB-FL w/o peer groups"]:
                G.append(dict(dataset="toniot", method=m, seed=s))
                for a, rho in itertools.product(ATTACKS, [0.2, 0.4]):
                    G.append(dict(dataset="toniot", method=m, attack=a, rho=rho, seed=s))
    if name in ("ablation", "all"):
        for s, m in itertools.product(seeds, ABLATIONS[1:]):
            G.append(dict(method=m, seed=s))
            for a in ATTACKS:
                G.append(dict(method=m, attack=a, rho=0.3, seed=s))
                G.append(dict(method=m, attack=a, rho=0.3, onset=25, seed=s))
    if name in ("noniid", "all"):
        for m, b in itertools.product(METHODS, [0.1, 0.5, 1.0]):
            G.append(dict(method=m, beta=b, seed=0))
            for a in ATTACKS:
                G.append(dict(method=m, attack=a, rho=0.3, beta=b, seed=0))
    seen, out = set(), []
    for g in G:
        rid = run_id({**DEFAULTS, **g})
        if rid not in seen:
            seen.add(rid)
            out.append(g)
    return out


def collect() -> None:
    import glob
    import pandas as pd
    rows, curves = [], []
    for f in sorted(glob.glob(os.path.join(RUN_DIR, "*.json"))):
        r = json.load(open(f))
        c = r["cfg"]
        fin = {k: v for k, v in r["final"].items() if k != "per_class_f1"}
        rows.append(dict(run_id=r["run_id"], dataset=c["dataset"], method=c["method"],
                         attack=c["attack"], rho=c["rho"], beta=c["beta"], onset=c["onset"],
                         seed=c["seed"], **fin,
                         mean_agg_ms=float(np.mean(r["history"]["agg_ms"])),
                         n_params=r["n_params"], wall_s=r.get("wall_s", np.nan),
                         per_class_f1=";".join(map(str, r["final"]["per_class_f1"]))))
        for i, rd in enumerate(r["history"]["round"]):
            curves.append(dict(run_id=r["run_id"], dataset=c["dataset"], method=c["method"],
                               attack=c["attack"], rho=c["rho"], beta=c["beta"], onset=c["onset"],
                               seed=c["seed"], round=rd, macro_f1=r["history"]["macro_f1"][i],
                               accuracy=r["history"]["accuracy"][i]))
    res = os.path.join(HERE, "..", "results")
    pd.DataFrame(rows).to_csv(os.path.join(res, "experiment_metrics.csv"), index=False)
    pd.DataFrame(curves).to_csv(os.path.join(res, "round_metrics.csv"), index=False)
    print(len(rows), "runs collected")


def check_flower(rounds: int = 5) -> None:
    """Feed identical client updates to (a) the simulator's TAB-FL rule and
    (b) the same rule wrapped as a Flower strategy receiving ``FitRes``
    messages, and report the largest difference between the two global models."""
    from flwr.common import Code, FitRes, Status, ndarrays_to_parameters
    cfg = {**DEFAULTS, "method": "TAB-FL", "attack": "sign_flip", "rho": 0.3}
    S = setup(cfg)
    model, C, K = S["model"], S["n_classes"], cfg["K"]
    Xtr, ytr = S["Xtr"], S["ytr"]
    clean = [(Xtr[p], ytr[p]) for p in S["parts"]]
    probe = CNNBiGRU(Xtr.shape[1], C)
    rl = make_root_losses(probe, torch.from_numpy(Xtr[S["root"]]), torch.from_numpy(ytr[S["root"]]), C)

    def ctx_for(wflat):
        return {"f": max(2, int(round(cfg["rho"] * K))), "slices": layer_slices(model),
                "root_losses": lambda U_: rl(torch.cat([wflat[None], wflat + U_]))}

    w = get_flat(model)
    direct = A.build("TAB-FL")
    direct.reset(K)
    nds0 = [p.detach().numpy().copy() for p in model.parameters()]
    strat = A.FlowerRobustStrategy(A.build("TAB-FL"), K, lambda rnd, wf: ctx_for(wf),
                                   ndarrays_to_parameters(nds0))
    rng = np.random.default_rng(0)
    for r in range(1, rounds + 1):
        Wl = train_clients(model, w, clean, cfg["local_steps"], cfg["batch"], cfg["lr"],
                           cfg["momentum"], rng, opt=cfg["opt"])
        U = Wl - w
        U[torch.from_numpy(S["mal"])] = atk.sign_flip(U[torch.from_numpy(S["mal"])])
        delta, _ = direct(U, torch.ones(K), ctx_for(w))
        results = []
        for k in np.random.default_rng(r).permutation(K):   # arrival order does not matter
            set_flat(model, w + U[k])
            nds = [p.detach().numpy().copy() for p in model.parameters()]
            results.append((None, FitRes(Status(Code.OK, ""), ndarrays_to_parameters(nds), 1,
                                         {"cid": int(k)})))
        strat.aggregate_fit(r, results, [])
        w = w + delta
        print(f"round {r}: max |w_sim - w_flower| = {float((w - strat.current).abs().max()):.2e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default=None)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--check-flower", action="store_true")
    ap.add_argument("--one", action="store_true")
    ap.add_argument("--reverse", action="store_true", help="walk the grid from the end")
    for k, v in DEFAULTS.items():
        if k != "log_trust":
            ap.add_argument(f"--{k}", type=type(v), default=v)
    a = ap.parse_args()
    os.makedirs(RUN_DIR, exist_ok=True)
    if a.collect:
        collect()
    elif a.check_flower:
        check_flower()
    elif a.grid:
        G = grid(a.grid)
        if a.reverse:
            G = G[::-1]
        print(f"{len(G)} runs in grid '{a.grid}'", flush=True)
        t0 = time.time()
        it = (map(_worker, G) if a.workers == 1 else Pool(a.workers).imap_unordered(_worker, G))
        if True:
            for i, (p, s) in enumerate(it, 1):
                print(f"[{i}/{len(G)}] {os.path.basename(p)} {s:.1f}s "
                      f"elapsed {(time.time() - t0) / 60:.1f} min", flush=True)
    elif a.one:
        cfg = {k: getattr(a, k) for k in DEFAULTS if k != "log_trust"}
        t0 = time.time()
        r = run_experiment(cfg)
        print(json.dumps(r["final"], indent=1))
        print("F1 curve:", np.round(r["history"]["macro_f1"][::5], 3).tolist())
        print(f"params {r['n_params']}  wall {time.time() - t0:.1f}s  "
              f"agg {np.mean(r['history']['agg_ms']):.1f} ms")
