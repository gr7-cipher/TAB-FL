"""Build Figures 2-5 and the LaTeX result tables from the logged runs.

Inputs  : ../results/experiment_metrics.csv, ../results/round_metrics.csv
          and ../results/runs/*.json (trust traces)
Outputs : ../figures/fig{2..5}_*.pdf/.png (300 dpi) and ../results/tables/*.tex

Run ``python train_fl.py --collect`` first.  Figure 6 (SHAP) is produced by
``shap_analysis.py`` and Figure 1 is the TikZ diagram in figures/.
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
FIG = os.path.join(HERE, "..", "figures")
TAB = os.path.join(RES, "tables")

METHODS = ["FedAvg", "FedProx", "Krum", "Multi-Krum", "Median", "Trimmed Mean", "TAB-FL"]
ATTACKS = ["label_flip", "sign_flip", "gaussian", "backdoor"]
ATTACK_NAMES = {"none": "No attack", "label_flip": "Label flip", "sign_flip": "Sign flip",
                "gaussian": "Gaussian noise", "backdoor": "Backdoor + scaling"}
# Okabe-Ito palette (colour-blind safe); TAB-FL in vermillion, drawn last and thicker
COLORS = {"FedAvg": "#000000", "FedProx": "#999999", "Krum": "#56B4E9", "Multi-Krum": "#0072B2",
          "Median": "#009E73", "Trimmed Mean": "#E69F00", "TAB-FL": "#D55E00",
          "Centralized": "#CC79A7"}
STYLES = {"FedAvg": "-", "FedProx": "--", "Krum": ":", "Multi-Krum": "-.", "Median": "-",
          "Trimmed Mean": "--", "TAB-FL": "-", "Centralized": ":"}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.grid": True, "grid.color": "#dddddd", "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42,
})


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=300)
    plt.close(fig)


def load(include_misf: bool = False):
    """Load the merged results.  Runs with a misspecified attacker count
    (f_assumed > 0) are only used by the misf table and are dropped otherwise."""
    M = pd.read_csv(os.path.join(RES, "experiment_metrics.csv"))
    R = pd.read_csv(os.path.join(RES, "round_metrics.csv"))
    for D in (M, R):
        if "f_assumed" not in D:
            D["f_assumed"] = 0
    if not include_misf:
        M, R = M[M.f_assumed == 0], R[R.f_assumed == 0]
    return M, R


def misf_table() -> None:
    """Macro-F1 (and backdoor ASR) when Krum-type rules get a wrong attacker count."""
    M, _ = load(include_misf=True)
    cols = [(a, r) for a in ["sign_flip", "gaussian", "label_flip", "backdoor"] for r in (0.2, 0.4)]

    def cell(ds, m, fa, a, rho, metric="macro_f1"):
        # f = 8 is the true count at 40 %, and f = 2 the oracle choice for clean runs
        true_f = max(2, int(round(rho * 20)))
        d = sel(M, dataset=ds, method=m, attack=a, rho=rho, beta=0.3, onset=0,
                f_assumed=0 if fa in (0, true_f) else fa)
        return d[metric].mean(), len(d)

    rows, blocks = [], [("ciciot2023", [("Multi-Krum", 0), ("Multi-Krum", 2), ("Multi-Krum", 8),
                                        ("Trimmed Mean", 0), ("Trimmed Mean", 2), ("Trimmed Mean", 8),
                                        ("TAB-FL", 0)]),
                        ("toniot", [("Multi-Krum", 0), ("Multi-Krum", 2), ("Multi-Krum", 8), ("TAB-FL", 0)])]
    for ds, rules in blocks:
        name = "CICIoT2023" if ds == "ciciot2023" else "TON\\_IoT"
        rows.append(f"\\multicolumn{{12}}{{l}}{{\\emph{{{name}}}}} \\\\")
        for m, fa in rules:
            label = {0: "true $f$" if m != "TAB-FL" else "no $f$", 2: "$f=2$", 8: "$f=8$"}[fa]
            cl, n0 = cell(ds, m, fa, "none", 0.0)
            vals = [f"{cl:.3f}"]
            for a, rho in cols:
                v, n = cell(ds, m, fa, a, rho)
                vals.append(f"{v:.3f}" if n else "--")
            asr, n = cell(ds, m, fa, "backdoor", 0.4, "asr")
            vals.append(f"{asr:.2f}" if n else "--")
            rows.append(f"{m} & {label} & " + " & ".join(vals) + " \\\\")
        rows.append("\\midrule")
    with open(os.path.join(TAB, "misf.tex"), "w") as fh:
        fh.write("\n".join(rows[:-1]) + "\n")
    print("\n".join(rows[:-1]))


def sel(df, **kw):
    m = np.ones(len(df), dtype=bool)
    for k, v in kw.items():
        m &= np.asarray(np.isclose(df[k], v) if isinstance(v, float) else (df[k] == v))
    return df[m]


# ------------------------------------------------------------------ Figure 2
def fig2(R):
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.9), sharex=True, sharey=True)
    for i, rho in enumerate([0.2, 0.4]):
        for j, a in enumerate(ATTACKS):
            ax = axes[i, j]
            for m in METHODS:
                d = sel(R, dataset="ciciot2023", method=m, attack=a, rho=rho, beta=0.3, onset=0)
                if d.empty:
                    continue
                g = d.groupby("round")["macro_f1"]
                mu, sd = g.mean(), g.std().fillna(0)
                lw = 1.5 if m == "TAB-FL" else 0.9
                ax.plot(mu.index, mu.values, STYLES[m], color=COLORS[m], lw=lw, label=m,
                        zorder=5 if m == "TAB-FL" else 2)
                ax.fill_between(mu.index, mu - sd, mu + sd, color=COLORS[m], alpha=0.12, lw=0)
            ax.set_ylim(0, 0.65)
            ax.set_xlim(1, 50)
            if i == 0:
                ax.set_title(ATTACK_NAMES[a])
            if j == 0:
                ax.set_ylabel(f"Macro-F1 ($\\rho$ = {int(rho * 100)}%)")
            if i == 1:
                ax.set_xlabel("Communication round")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=7, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, "fig2_convergence_comparison")


# ------------------------------------------------------------------ Figure 3
def fig3(M):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5))
    x = np.arange(len(ATTACKS))
    wdt = 0.115
    for k, metric in enumerate(["macro_f1", "fpr"]):
        ax = axes[k]
        for i, m in enumerate(METHODS):
            mu, sd = [], []
            for a in ATTACKS:
                d = sel(M, dataset="ciciot2023", method=m, attack=a, rho=0.3, beta=0.3, onset=0)[metric]
                mu.append(d.mean())
                sd.append(d.std() if len(d) > 1 else 0)
            mu_a, sd_a = np.array(mu), np.array(sd)
            lo = np.minimum(sd_a, mu_a)                    # keep bars inside [0, 1]
            hi = np.minimum(sd_a, 1 - mu_a) if metric == "fpr" else sd_a
            ax.bar(x + (i - 3) * wdt, mu, wdt, yerr=np.vstack([lo, hi]), color=COLORS[m], label=m,
                   edgecolor="white", linewidth=0.3, error_kw=dict(lw=0.6, capsize=1.5))
        ax.set_xticks(x)
        ax.set_xticklabels([ATTACK_NAMES[a] for a in ATTACKS])
        ax.set_ylabel("Macro-F1" if metric == "macro_f1" else "False positive rate")
        ax.set_title("(a) Macro-F1 (higher is better)" if k == 0 else "(b) FPR on benign traffic (lower is better)")
        ax.set_ylim(0, 0.65 if metric == "macro_f1" else 1.05)
        ax.grid(axis="x", visible=False)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=7, frameon=False, bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "fig3_attack_robustness_bar")


# ------------------------------------------------------------------ Figure 4
def _trust(run_id):
    p = os.path.join(RES, "runs", run_id + ".json")
    r = json.load(open(p))
    return np.array(r["trust"]), np.array(r["malicious"], dtype=int)


def fig4():
    panels = [("ciciot2023_TAB-FL_backdoor_r30_b0.3_on0_s0", "(a) Backdoor + scaling from round 1"),
              ("ciciot2023_TAB-FL_sign_flip_r30_b0.3_on25_s0", "(b) Sign flip switched on at round 25"),
              ("ciciot2023_TAB-FL_label_flip_r30_b0.3_on25_s0", "(c) Label flip switched on at round 25")]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=True)
    for ax, (rid, title) in zip(axes, panels):
        T, mal = _trust(rid)
        K = T.shape[1]
        order = list(mal) + [k for k in range(K) if k not in set(mal)]
        im = ax.imshow(T[:, order].T, aspect="auto", cmap="viridis", vmin=0, vmax=1,
                       extent=(0.5, T.shape[0] + 0.5, K - 0.5, -0.5), interpolation="nearest")
        ax.axhline(len(mal) - 0.5, color="white", lw=0.8)
        if "on25" in rid:
            ax.axvline(25, color="#D55E00", lw=1.0, ls="--")
        ax.set_title(title)
        ax.set_xlabel("Communication round")
        ax.grid(False)
        ax.set_yticks([len(mal) / 2 - 0.5, len(mal) + (K - len(mal)) / 2 - 0.5])
        ax.set_yticklabels(["Byzantine", "honest"], rotation=90, va="center")
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.01)
    cb.set_label("Trust $T_i^{(t)}$")
    save(fig, "fig4_trust_dynamics_heatmap")


# ------------------------------------------------------------------ Figure 5
ABL = ["TAB-FL", "TAB-FL w/o EMA", "TAB-FL w/o clipping", "TAB-FL w/o validation",
       "TAB-FL w/o geometry"]
ABL_SHORT = {"TAB-FL": "TAB-FL (full)", "TAB-FL w/o EMA": "w/o EMA trust",
             "TAB-FL w/o clipping": "w/o norm clipping", "TAB-FL w/o validation": "w/o root-set check",
             "TAB-FL w/o geometry": "w/o geometric check"}


def fig5(M):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw=dict(width_ratios=[1, 1.25]))
    ax = axes[0]
    betas = [0.1, 0.3, 0.5, 1.0]
    for m in METHODS:
        ys = []
        for b in betas:
            d = sel(M, dataset="ciciot2023", method=m, rho=0.3, beta=b, onset=0, seed=0)
            d = d[d.attack.isin(ATTACKS)]
            ys.append(d.macro_f1.mean() if len(d) == len(ATTACKS) else np.nan)
        ax.plot(betas, ys, STYLES[m], marker="o", ms=3, color=COLORS[m], lw=1.5 if m == "TAB-FL" else 0.9,
                label=m)
    ax.set_xscale("log")
    ax.set_xticks(betas)
    ax.set_xticklabels([str(b) for b in betas])
    ax.set_xlabel("Dirichlet $\\beta$ (smaller = more heterogeneous)")
    ax.set_ylabel("Macro-F1, mean over 4 attacks")
    ax.set_title("(a) Sensitivity to non-IID heterogeneity ($\\rho$ = 30%)")
    ax.legend(ncol=4, frameon=False, fontsize=6.3, loc="upper center", bbox_to_anchor=(0.5, -0.2))

    ax = axes[1]
    cats = ATTACKS + ["dyn"]
    x = np.arange(len(cats))
    wdt = 0.16
    pal = ["#D55E00", "#56B4E9", "#E69F00", "#009E73", "#CC79A7"]
    for i, v in enumerate(ABL):
        mu, sd = [], []
        for a in cats:
            if a == "dyn":
                d = sel(M, dataset="ciciot2023", method=v, rho=0.3, beta=0.3, onset=25)
                d = d[d.attack.isin(ATTACKS)].groupby("seed").macro_f1.mean()
            else:
                d = sel(M, dataset="ciciot2023", method=v, attack=a, rho=0.3, beta=0.3, onset=0).macro_f1
            mu.append(d.mean())
            sd.append(d.std() if len(d) > 1 else 0)
        ax.bar(x + (i - 2) * wdt, mu, wdt, yerr=sd, color=pal[i], label=ABL_SHORT[v],
               edgecolor="white", linewidth=0.3, error_kw=dict(lw=0.6, capsize=1.5))
    ax.set_xticks(x)
    ax.set_xticklabels(["Label\nflip", "Sign\nflip", "Gaussian\nnoise", "Backdoor\n+ scaling",
                        "Dynamic\n(mean of 4)"], fontsize=6.5)
    ax.set_ylabel("Macro-F1")
    ax.set_ylim(0, 0.65)
    ax.set_title("(b) Component ablation ($\\rho$ = 30%, $\\beta$ = 0.3)")
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=3, frameon=False, fontsize=6.3, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    fig.tight_layout()
    save(fig, "fig5_ablation_and_noniid")


# ------------------------------------------------------------------ tables
def recovery_stats(R, method, attack, dataset="ciciot2023", onset=25, tol=0.02):
    """Pre-attack F1, worst drop after onset and rounds to recover within ``tol``."""
    out = []
    for s, d in sel(R, dataset=dataset, method=method, attack=attack, rho=0.3, beta=0.3,
                    onset=onset).groupby("seed"):
        f = d.sort_values("round").macro_f1.to_numpy()
        pre = f[onset - 6:onset - 1].mean()          # rounds 20-24
        post = f[onset - 1:]                          # rounds 25-50
        drop = max(0.0, pre - post.min())
        rec = np.nan
        for i in range(len(post) - 2):
            if np.all(post[i:i + 3] >= pre - tol):
                rec = i
                break
        out.append((pre, drop, rec, f[-1]))
    return np.array(out)


def fmt(mu, sd=None, bold=False, pct=False):
    s = f"{mu:.3f}" if not pct else f"{100 * mu:.1f}"
    if sd is not None and not np.isnan(sd):
        s += f"$\\pm${sd:.3f}" if not pct else f"$\\pm${100 * sd:.1f}"
    return f"\\textbf{{{s}}}" if bold else s


def tables(M, R):
    os.makedirs(TAB, exist_ok=True)
    lines = {}
    # Table: main results at rho = 30%
    rows = []
    for a in ["none"] + ATTACKS:
        rho = 0.0 if a == "none" else 0.3
        stats = {}
        for m in METHODS:
            d = sel(M, dataset="ciciot2023", method=m, attack=a, rho=rho, beta=0.3, onset=0)
            stats[m] = {k: (d[k].mean(), d[k].std()) for k in
                        ["accuracy", "macro_f1", "detection_rate", "fpr", "asr"]}
        best = max(METHODS, key=lambda m: stats[m]["macro_f1"][0])
        for i, m in enumerate(METHODS):
            s = stats[m]
            cells = [fmt(*s["accuracy"]), fmt(*s["macro_f1"], bold=(m == best)),
                     fmt(*s["detection_rate"]), fmt(*s["fpr"]), fmt(s["asr"][0])]
            lead = f"\\multirow{{7}}{{*}}{{{ATTACK_NAMES[a]}}}" if i == 0 else ""
            rows.append(f"{lead} & {m} & " + " & ".join(cells) + " \\\\")
        rows.append("\\midrule")
    lines["main"] = "\n".join(rows[:-1])

    # Table: macro-F1 vs rho for each attack (mean over seeds)
    rows = []
    for a in ATTACKS:
        for i, m in enumerate(METHODS):
            vals = []
            for rho in [0.1, 0.2, 0.3, 0.4]:
                d = sel(M, dataset="ciciot2023", method=m, attack=a, rho=rho, beta=0.3, onset=0).macro_f1
                vals.append(d.mean())
            lead = f"\\multirow{{7}}{{*}}{{{ATTACK_NAMES[a]}}}" if i == 0 else ""
            rows.append(f"{lead} & {m} & " + " & ".join(f"{v:.3f}" for v in vals) + " \\\\")
        rows.append("\\midrule")
    lines["rho"] = "\n".join(rows[:-1])

    # Table: dynamic attack recovery
    rows = []
    for m in METHODS:
        cells = []
        for a in ATTACKS:
            st = recovery_stats(R, m, a)
            if len(st) == 0:
                cells += ["--", "--"]
                continue
            drop = st[:, 1].mean()
            rec = st[:, 2]
            n_rec = np.sum(~np.isnan(rec))
            rec_s = (f"{np.nanmean(rec):.1f}" if n_rec == len(rec) else
                     (f"{np.nanmean(rec):.1f}$^\\dagger$" if n_rec > 0 else "n.r."))
            cells += [f"{drop:.3f}", rec_s]
        rows.append(f"{m} & " + " & ".join(cells) + " \\\\")
    lines["dynamic"] = "\n".join(rows)

    # Table: TON_IoT
    rows = []
    ton_methods = METHODS + ["TAB-FL w/o peer groups"]
    for a in ["none"] + ATTACKS:
        for i, m in enumerate(ton_methods):
            cells = []
            for rho in ([0.0] if a == "none" else [0.2, 0.4]):
                d = sel(M, dataset="toniot", method=m, attack=a, rho=rho, beta=0.3, onset=0)
                cells += [fmt(d.macro_f1.mean(), d.macro_f1.std()), fmt(d.fpr.mean())]
            if a == "none":
                cells += ["", ""]
            lead = f"\\multirow{{{len(ton_methods)}}}{{*}}{{{ATTACK_NAMES[a]}}}" if i == 0 else ""
            name = "\\quad w/o peer groups" if m == "TAB-FL w/o peer groups" else m
            rows.append(f"{lead} & {name} & " + " & ".join(cells) + " \\\\")
        rows.append("\\midrule")
    lines["toniot"] = "\n".join(rows[:-1])

    # Table: selection quality (false quarantine / malicious admission)
    rows = []
    for m in ["Krum", "Multi-Krum", "TAB-FL"]:
        cells = []
        for a in ATTACKS:
            d = sel(M, dataset="ciciot2023", method=m, attack=a, rho=0.3, beta=0.3, onset=0)
            cells += [f"{100 * d.honest_excluded.mean():.1f}", f"{100 * d.malicious_admitted.mean():.1f}"]
        d0 = sel(M, dataset="ciciot2023", method=m, attack="none", rho=0.0, beta=0.3, onset=0)
        rows.append(f"{m} & {100 * d0.honest_excluded.mean():.1f} & " + " & ".join(cells) + " \\\\")
    lines["selection"] = "\n".join(rows)

    # TON_IoT selection quality (printed for the text)
    for m in ["Multi-Krum", "TAB-FL", "TAB-FL w/o peer groups"]:
        for a, rho in [("none", 0.0)] + [(a, r) for a in ATTACKS for r in (0.2, 0.4)]:
            d = sel(M, dataset="toniot", method=m, attack=a, rho=rho, beta=0.3, onset=0)
            print(f"TON sel {m:24s} {a:10s} {rho:.1f} F1 {d.macro_f1.mean():.3f} FPR {d.fpr.mean():.3f} "
                  f"ASR {d.asr.mean():.3f} Hex {100 * d.honest_excluded.mean():.1f} "
                  f"Madm {100 * d.malicious_admitted.mean():.1f}")

    # Centralized reference
    c = sel(M, dataset="ciciot2023", method="Centralized")
    ct = sel(M, dataset="toniot", method="Centralized")
    lines["central"] = (f"CICIoT2023 centralized: acc {c.accuracy.mean():.3f}+-{c.accuracy.std():.3f} "
                        f"F1 {c.macro_f1.mean():.3f}+-{c.macro_f1.std():.3f} FPR {c.fpr.mean():.3f}; "
                        f"TON_IoT centralized: acc {ct.accuracy.mean():.3f} F1 {ct.macro_f1.mean():.3f}+-"
                        f"{ct.macro_f1.std():.3f} FPR {ct.fpr.mean():.3f}")
    for k, v in lines.items():
        with open(os.path.join(TAB, f"{k}.tex"), "w") as fh:
            fh.write(v + "\n")
    print(lines["central"])


if __name__ == "__main__":
    M, R = load()
    for f, args in [(fig2, (R,)), (fig3, (M,)), (fig4, ()), (fig5, (M,)), (tables, (M, R)),
                    (misf_table, ())]:
        try:
            f(*args)
            print("ok", f.__name__)
        except Exception as e:  # keep going so partial results can be inspected
            print("skipped", f.__name__, type(e).__name__, e)
