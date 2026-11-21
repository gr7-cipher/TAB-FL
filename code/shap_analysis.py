"""SHAP attributions for the converged TAB-FL global model (Figure 6).

For each dataset we train TAB-FL for 50 rounds (seed 0, beta = 0.3, no
attack), then explain the model's *attack score*, i.e. the log-odds
    a(x) = logsumexp_{c != benign} z_c(x) - z_benign(x),
with SHAP's GradientExplainer (expected gradients).  A positive attribution
pushes a flow or telemetry record towards an alert.

Outputs: ../figures/fig6_shap_explainability.{pdf,png},
         ../results/shap_summary.json, ../results/models/*.pt
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

import train_fl as T
from generate_plots import FIG, RES, plt, save
from models import CNNBiGRU, set_flat

N_BACKGROUND = 300
N_EXPLAIN = 1500
TOP = 12


class AttackScore(torch.nn.Module):
    def __init__(self, net, benign=0):
        super().__init__()
        self.net, self.benign = net, benign

    def forward(self, x):
        z = self.net(x)
        others = torch.cat([z[:, :self.benign], z[:, self.benign + 1:]], 1)
        return (torch.logsumexp(others, 1) - z[:, self.benign]).unsqueeze(1)


def pretty(name: str) -> str:
    return (name.replace("_", " ").replace("dev ", "device: ")
            .replace("FC1 Read Input Register", "Modbus FC1 input reg.")
            .replace("FC2 Read Discrete Value", "Modbus FC2 discrete")
            .replace("FC3 Read Holding Register", "Modbus FC3 holding reg.")
            .replace("FC4 Read Coil", "Modbus FC4 coil"))


def explain(dataset: str):
    import shap
    os.makedirs(os.path.join(RES, "models"), exist_ok=True)
    mpath = os.path.join(RES, "models", f"tabfl_{dataset}_seed0.pt")
    d = T.load_processed(os.path.join(T.DATA_DIR, f"{dataset}.npz"))
    feats = [str(f) for f in d["feature_names"]]
    classes = [str(c) for c in d["class_names"]]
    net = CNNBiGRU(d["X_train"].shape[1], len(classes))
    if os.path.exists(mpath):
        net.load_state_dict(torch.load(mpath))
    else:
        r = T.run_experiment({"dataset": dataset, "method": "TAB-FL", "seed": 0, "return_weights": True})
        set_flat(net, torch.tensor(r["weights"], dtype=torch.float32))
        torch.save(net.state_dict(), mpath)
    net.eval()
    rng = np.random.default_rng(0)
    Xtr, Xte, yte = d["X_train"], d["X_test"], d["y_test"]
    bg = torch.from_numpy(Xtr[rng.choice(len(Xtr), N_BACKGROUND, replace=False)])
    # stratified explanation sample so that rare attack classes are represented
    idx = np.concatenate([rng.choice(np.flatnonzero(yte == c), min(N_EXPLAIN // len(classes),
                                                                  np.sum(yte == c)), replace=False)
                          for c in range(len(classes))])
    Xe = torch.from_numpy(Xte[idx])
    expl = shap.GradientExplainer(AttackScore(net), bg)
    sv = expl.shap_values(Xe, nsamples=200, rseed=0)
    sv = np.asarray(sv).reshape(len(idx), -1)
    imp = np.abs(sv).mean(0)
    order = np.argsort(-imp)
    per_class = {}
    for c in range(len(classes)):
        m = yte[idx] == c
        ci = np.abs(sv[m]).mean(0)
        per_class[classes[c]] = [(feats[j], round(float(ci[j]), 4)) for j in np.argsort(-ci)[:3]]
    summary = dict(dataset=dataset, n_explained=int(len(idx)), n_background=N_BACKGROUND,
                   top_features=[(feats[j], round(float(imp[j]), 4)) for j in order[:TOP]],
                   per_class_top3=per_class)
    return summary, sv, Xte[idx], feats, order


def beeswarm(ax, sv, X, feats, order, top=TOP):
    rng = np.random.default_rng(1)
    for r, j in enumerate(order[:top][::-1]):
        v = X[:, j]
        lo, hi = np.percentile(v, 5), np.percentile(v, 95)
        c = np.clip((v - lo) / (hi - lo + 1e-12), 0, 1)
        y = r + rng.uniform(-0.3, 0.3, size=len(v))
        ax.scatter(sv[:, j], y, c=c, cmap="coolwarm", s=2.0, alpha=0.6, linewidths=0, rasterized=True)
    ax.set_yticks(range(top))
    ax.set_yticklabels([pretty(feats[j]) for j in order[:top][::-1]])
    ax.axvline(0, color="#777777", lw=0.6)
    ax.set_xlabel("SHAP value (impact on attack log-odds)")


def main():
    s1, sv1, X1, f1, o1 = explain("ciciot2023")
    s2, sv2, X2, f2, o2 = explain("toniot")
    with open(os.path.join(RES, "shap_summary.json"), "w") as fh:
        json.dump([s1, s2], fh, indent=1)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.0), gridspec_kw=dict(width_ratios=[0.8, 1.15, 0.8]))
    imp1 = np.abs(sv1).mean(0)
    axes[0].barh(range(TOP), imp1[o1[:TOP]][::-1], color="#0072B2")
    axes[0].set_yticks(range(TOP))
    axes[0].set_yticklabels([pretty(f1[j]) for j in o1[:TOP][::-1]])
    axes[0].set_xlabel("mean |SHAP|")
    axes[0].set_title("(a) CICIoT2023: global importance")
    beeswarm(axes[1], sv1, X1, f1, o1)
    axes[1].set_title("(b) CICIoT2023: per-flow attributions")
    sm = plt.cm.ScalarMappable(cmap="coolwarm")
    cb = fig.colorbar(sm, ax=axes[1], fraction=0.04, pad=0.02, ticks=[0, 1])
    cb.ax.set_yticklabels(["low", "high"])
    cb.set_label("feature value", labelpad=-6)
    imp2 = np.abs(sv2).mean(0)
    axes[2].barh(range(TOP), imp2[o2[:TOP]][::-1], color="#009E73")
    axes[2].set_yticks(range(TOP))
    axes[2].set_yticklabels([pretty(f2[j]) for j in o2[:TOP][::-1]])
    axes[2].set_xlabel("mean |SHAP|")
    axes[2].set_title("(c) TON_IoT telemetry")
    for ax in axes:
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "fig6_shap_explainability")
    print(json.dumps([s1["top_features"][:6], s2["top_features"][:6]], indent=0))


if __name__ == "__main__":
    main()
