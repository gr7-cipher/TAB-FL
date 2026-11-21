"""Server-side aggregation rules.

Every rule receives a (K x d) matrix ``U`` of client updates (local model
minus current global model), the clients' sample counts ``n`` and a context
dict, and returns the aggregated update plus diagnostic information.

Context keys
------------
f          : number of Byzantine clients the rule is told to tolerate
slices     : list of (name, slice) giving each layer's range in the vector
root_losses: callable mapping a (K, d) update matrix to a (K+1, C) array of
             per-class root-set losses; row 0 is the current global model
             (only used by TAB-FL)

``FlowerRobustStrategy`` at the bottom wraps any rule as a Flower strategy
so it can be dropped into a Flower deployment unchanged.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch


class Aggregator:
    name = "base"

    def reset(self, n_clients: int) -> None:  # stateful rules override
        pass

    def __call__(self, U: torch.Tensor, n: torch.Tensor, ctx: dict):
        raise NotImplementedError


class FedAvg(Aggregator):
    name = "FedAvg"

    def __call__(self, U, n, ctx):
        w = n / n.sum()
        return (w[:, None] * U).sum(0), {}


class FedProx(FedAvg):
    """Server side identical to FedAvg; the proximal term lives in the client."""
    name = "FedProx"


def _krum_scores(U: torch.Tensor, f: int) -> torch.Tensor:
    K = U.shape[0]
    D = torch.cdist(U, U) ** 2
    m = max(1, K - f - 2)
    D.fill_diagonal_(float("inf"))
    return torch.sort(D, dim=1).values[:, :m].sum(1)


class Krum(Aggregator):
    name = "Krum"

    def __call__(self, U, n, ctx):
        s = _krum_scores(U, ctx["f"])
        i = int(torch.argmin(s))
        return U[i].clone(), {"selected": [i]}


class MultiKrum(Aggregator):
    name = "Multi-Krum"

    def __call__(self, U, n, ctx):
        K = U.shape[0]
        s = _krum_scores(U, ctx["f"])
        m = max(1, K - ctx["f"])
        sel = torch.argsort(s)[:m]
        return U[sel].mean(0), {"selected": sel.tolist()}


class Median(Aggregator):
    name = "Median"

    def __call__(self, U, n, ctx):
        return U.median(dim=0).values, {}


class TrimmedMean(Aggregator):
    name = "Trimmed Mean"

    def __call__(self, U, n, ctx):
        K = U.shape[0]
        b = min(ctx["f"], (K - 1) // 2)
        S = torch.sort(U, dim=0).values
        return S[b:K - b].mean(0), {}


@dataclass
class TABFLConfig:
    alpha: float = 0.7        # EMA memory of the trust score
    t0: float = 1.0           # initial trust
    kappa: float = 2.0        # tolerance band, in robust standard deviations
    tau_min: float = 0.3      # absolute quarantine floor
    eta: float = 0.6          # relative quarantine threshold (fraction of median trust)
    clip_c: float = 1.5       # norm clip relative to ||median update||
    use_ema: bool = True
    use_clip: bool = True
    use_geo: bool = True      # layer-wise cosine + norm-ratio evidence
    use_val: bool = True      # root-set validation-loss evidence
    use_groups: bool = True   # peer-group references when the operator supplies gateway roles
    kappa_global: float = 4.0  # wider band of the global guard used with peer groups


def robust_z(x: np.ndarray, ref: np.ndarray | None = None) -> np.ndarray:
    """(x - median) / (1.4826 * MAD) with statistics taken over x[ref];
    the MAD is floored to avoid blow-ups."""
    r = x if ref is None else x[ref]
    med = np.median(r)
    mad = 1.4826 * np.median(np.abs(r - med))
    scale = max(mad, 1e-3 * (abs(med) + 1e-12), 1e-12)
    return (x - med) / scale


def robust_z_grouped(x: np.ndarray, gid: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Group-centred robust z-score.  Each x_i is centred, leave-one-out, on the
    median of the *other* trusted members of its group (the global trusted
    median when there are none), which avoids the exact zeros that a 2- or
    3-member median would produce, and scaled by the pooled MAD of the centred
    values of the trusted clients."""
    med_all = np.median(x[ref])
    c = np.empty_like(x)
    for i in range(len(x)):
        peers = (gid == gid[i]) & ref
        peers[i] = False
        c[i] = x[i] - (np.median(x[peers]) if peers.any() else med_all)
    mad = 1.4826 * np.median(np.abs(c[ref]))
    scale = max(mad, 1e-3 * (abs(med_all) + 1e-12), 1e-12)
    return c / scale


class TABFL(Aggregator):
    """Trust-Aware Byzantine-resilient aggregation (dual-layer defence).

    Layer 1 (consistency scoring).  For each client update U_i we measure
      c_i : mean layer-wise cosine similarity to the coordinate-wise median,
      r_i : log of the norm ratio ||U_i|| / ||median||,
      l_i : mean loss of (global + U_i) on a small server-side root set,
      g_i : the largest per-class loss reduction that U_i achieves on the
            root set relative to the current global model,
    and turn each into a robust z-score against the population of clients
    of the same round (median / MAD).  Deviations inside a tolerance band of
    kappa robust standard deviations cost nothing, so honest but
    heterogeneous (non-IID) clients keep phi_i close to one:
        phi_i = exp(-1/2 * sum_ch max(0, z_ch,i - kappa)^2),
    with one-sided z for cosine (too low), loss (too high) and gain (too
    low) and a two-sided z for the norm ratio.
    The coordinate-wise median and the robust statistics are computed over
    the clients that were trusted in the previous round, so that quarantined
    clients cannot inflate the dispersion estimate.
    Layer 2 (reputation).  T_i <- alpha T_i + (1 - alpha) phi_i; clients with
    T_i < tau_t = max(tau_min, eta * median_j T_j) are quarantined for the
    round, and the survivors' updates, clipped to clip_c * ||median||, are
    averaged with weights T_i / sum_j T_j.
    """

    name = "TAB-FL"

    def __init__(self, cfg: TABFLConfig | None = None, name: str | None = None):
        self.cfg = cfg or TABFLConfig()
        if name:
            self.name = name
        self.T = None

    def reset(self, n_clients: int) -> None:
        self.T = np.full(n_clients, self.cfg.t0, dtype=np.float64)
        self.ref_mask = np.ones(n_clients, dtype=bool)

    def evidence(self, U, ctx):
        if self.cfg.use_groups and ctx.get("groups") is not None:
            return self.evidence_grouped(U, ctx)
        return self.evidence_global(U, ctx)

    @staticmethod
    def _layer_cos(U, refs, slices):
        """Mean layer-wise cosine of every row of U to its own reference row."""
        cs = []
        for _, sl in slices:
            a, b = U[:, sl], refs[:, sl]
            cs.append(((a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + 1e-12)).numpy())
        return np.mean(np.stack(cs, 0), 0)

    def evidence_grouped(self, U, ctx):
        """Peer-group variant for deployments where the operator knows the role
        (device / service type) of every gateway.  Honest gateways of different
        roles legitimately disagree with the population, by an amount that is
        typical of the role.  Every signal is therefore centred on the median
        of the trusted gateways of the same role before it is scaled by the
        pooled robust dispersion (group-centred z-scores), and the validation
        signals are measured on the root records of that role.  A global guard
        with the wider band kappa_global keeps the uncentred z-scores of the
        geometric signals, so a role whose gateways are all compromised is not
        shielded by its own group level."""
        cfg = self.cfg
        K = U.shape[0]
        R = self.ref_mask
        gid = np.asarray(ctx["groups"])
        ref = U[torch.from_numpy(R)].median(dim=0).values
        ref_norm = float(ref.norm()) + 1e-12
        norms = U.norm(dim=1).numpy() + 1e-12
        pen = np.zeros(K)
        cos_l = np.zeros(K)
        losses = np.zeros(K)
        if cfg.use_geo:
            cos_l = self._layer_cos(U, ref.repeat(K, 1), ctx["slices"])
            lnr = np.log(norms / ref_norm)
            z_cos = -robust_z_grouped(cos_l, gid, R)
            z_norm = np.abs(robust_z_grouped(lnr, gid, R))
            z_cosg = -robust_z(cos_l, R)
            z_normg = np.abs(robust_z(lnr, R))
            pen += np.maximum(0.0, z_cos - cfg.kappa) ** 2
            pen += np.maximum(0.0, z_norm - cfg.kappa) ** 2
            pen += np.maximum(0.0, z_cosg - cfg.kappa_global) ** 2
            pen += np.maximum(0.0, z_normg - cfg.kappa_global) ** 2
        if cfg.use_val:
            L = np.asarray(ctx["root_record_losses"](U))  # (K+1, N); row 0 = global model
            yr, rg = np.asarray(ctx["root_y"]), np.asarray(ctx["root_groups"])
            rel_loss = np.zeros(K)
            gain = np.zeros(K)
            for g in np.unique(gid):
                sub = rg == g
                if sub.sum() < 10:                        # too few root records of this role
                    sub = np.ones_like(rg, dtype=bool)
                cls = np.unique(yr[sub])
                M = np.stack([L[:, sub & (yr == c)].mean(1) for c in cls], 1)  # (K+1, C_g)
                base, Mc = M[0], M[1:]
                rows = np.flatnonzero(gid == g)
                rel_loss[rows] = np.log(Mc[rows].mean(1) + 1e-12) - np.log(base.mean() + 1e-12)
                gain[rows] = ((base[None, :] - Mc[rows]) / (base[None, :] + 1e-12)).max(1)
            losses = rel_loss
            z_loss = robust_z_grouped(rel_loss, gid, R)
            z_gain = -robust_z_grouped(gain, gid, R)
            pen += np.maximum(0.0, z_loss - cfg.kappa) ** 2
            pen += np.maximum(0.0, z_gain - cfg.kappa) ** 2
        # clipping threshold per role: median norm of the role's trusted gateways
        clip_ref = np.full(K, ref_norm)
        for g in np.unique(gid):
            m = (gid == g) & R
            if m.sum() >= 2:
                clip_ref[gid == g] = np.median(norms[m])
        phi = np.exp(-0.5 * pen)
        self._z = {k: v for k, v in locals().items() if k.startswith("z_")}
        return phi, {"cos": cos_l, "norm": norms, "ref_norm": ref_norm, "root_loss": losses,
                     "clip_ref": clip_ref}

    def evidence_global(self, U, ctx):
        cfg = self.cfg
        K = U.shape[0]
        R = self.ref_mask                                # clients trusted in the previous round
        ref = U[torch.from_numpy(R)].median(dim=0).values
        ref_norm = float(ref.norm()) + 1e-12
        norms = U.norm(dim=1).numpy() + 1e-12
        cos_l = np.zeros(K)
        losses = np.zeros(K)
        pen = np.zeros(K)
        if cfg.use_geo:
            cs = []
            for _, sl in ctx["slices"]:
                r = ref[sl]
                rn = r.norm() + 1e-12
                cs.append(((U[:, sl] @ r) / (U[:, sl].norm(dim=1) * rn + 1e-12)).numpy())
            cos_l = np.mean(np.stack(cs, 0), 0)
            z_cos = -robust_z(cos_l, R)                   # large when direction disagrees
            z_norm = np.abs(robust_z(np.log(norms / ref_norm), R))
            pen += np.maximum(0.0, z_cos - cfg.kappa) ** 2
            pen += np.maximum(0.0, z_norm - cfg.kappa) ** 2
        if cfg.use_val:
            Lc = np.asarray(ctx["root_losses"](U))       # (K+1, C); row 0 = global model
            base, Lc = Lc[0], Lc[1:]
            losses = Lc.mean(1)
            gain = (base[None, :] - Lc).max(1)           # largest per-class improvement
            z_loss = robust_z(np.log(losses + 1e-12), R)  # large when the update hurts overall
            z_gain = -robust_z(gain, R)                   # large when it helps no class
            pen += np.maximum(0.0, z_loss - cfg.kappa) ** 2
            pen += np.maximum(0.0, z_gain - cfg.kappa) ** 2
        phi = np.exp(-0.5 * pen)
        self._z = {k: v for k, v in locals().items() if k.startswith("z_")}
        return phi, {"cos": cos_l, "norm": norms, "ref_norm": ref_norm, "root_loss": losses}

    def __call__(self, U, n, ctx):
        cfg = self.cfg
        K = U.shape[0]
        if self.T is None:
            self.reset(K)
        phi, ev = self.evidence(U, ctx)
        if cfg.use_ema:
            self.T = cfg.alpha * self.T + (1.0 - cfg.alpha) * phi
        else:
            self.T = phi.copy()
        tau = max(cfg.tau_min, cfg.eta * float(np.median(self.T)))
        keep = np.flatnonzero(self.T >= tau)
        if len(keep) == 0:  # degenerate round: fall back to the most trusted half
            keep = np.argsort(-self.T)[: max(1, K // 2)]
        # reference population for the next round's statistics (at least half the clients)
        ref_next = np.zeros(K, dtype=bool)
        ref_next[keep] = True
        if ref_next.sum() < (K + 1) // 2:
            ref_next[np.argsort(-self.T)[: (K + 1) // 2]] = True
        self.ref_mask = ref_next
        V = U[keep]
        if cfg.use_clip:
            if "clip_ref" in ev:   # peer groups: clip relative to the gateway's own group
                c = cfg.clip_c * torch.tensor(np.maximum(ev["clip_ref"][keep], 1e-12),
                                              dtype=U.dtype)[:, None]
            else:
                c = cfg.clip_c * ev["ref_norm"]
            nv = V.norm(dim=1, keepdim=True)
            V = V * torch.clamp(c / (nv + 1e-12), max=1.0)
        w = torch.tensor(self.T[keep] / self.T[keep].sum(), dtype=U.dtype)
        delta = (w[:, None] * V).sum(0)
        return delta, {"trust": self.T.copy(), "phi": phi, "tau": tau, "selected": keep.tolist(),
                       "cos": ev["cos"], "norm": ev["norm"]}


def build(name: str) -> Aggregator:
    table = {
        "FedAvg": FedAvg, "FedProx": FedProx, "Krum": Krum, "Multi-Krum": MultiKrum,
        "Median": Median, "Trimmed Mean": TrimmedMean, "TAB-FL": TABFL,
    }
    if name in table:
        return table[name]()
    ablations = {
        "TAB-FL w/o EMA": TABFLConfig(use_ema=False),
        "TAB-FL w/o clipping": TABFLConfig(use_clip=False),
        "TAB-FL w/o validation": TABFLConfig(use_val=False),
        "TAB-FL w/o geometry": TABFLConfig(use_geo=False),
        "TAB-FL w/o peer groups": TABFLConfig(use_groups=False),
    }
    if name in ablations:
        return TABFL(ablations[name], name=name)
    raise KeyError(name)


# --------------------------------------------------------------------- Flower

try:
    import flwr as fl
    from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

    class FlowerRobustStrategy(fl.server.strategy.FedAvg):
        """Flower strategy that delegates aggregation to one of the rules above.

        ``ctx_fn(server_round, global_flat)`` must return the context dict
        (``f``, ``slices`` and, for TAB-FL, ``root_loss``).  Clients are
        identified by ``FitRes.metrics['cid']`` so that per-client trust
        survives across rounds.
        """

        def __init__(self, aggregator: Aggregator, n_clients: int, ctx_fn, initial_parameters, **kw):
            super().__init__(initial_parameters=initial_parameters, fraction_fit=1.0,
                             min_fit_clients=n_clients, min_available_clients=n_clients, **kw)
            self.agg = aggregator
            self.agg.reset(n_clients)
            self.ctx_fn = ctx_fn
            nds = parameters_to_ndarrays(initial_parameters)
            self.shapes = [a.shape for a in nds]
            self.current = self._flat(nds)

        @staticmethod
        def _flat(nds):
            return torch.cat([torch.as_tensor(a, dtype=torch.float32).reshape(-1) for a in nds])

        def _unflat(self, vec):
            out, s = [], 0
            for sh in self.shapes:
                k = int(np.prod(sh))
                out.append(vec[s:s + k].reshape(sh).numpy().copy())
                s += k
            return out

        def aggregate_fit(self, server_round, results, failures):
            if not results:
                return None, {}
            results = sorted(results, key=lambda r: int(r[1].metrics["cid"]))
            U = torch.stack([self._flat(parameters_to_ndarrays(r.parameters)) - self.current
                             for _, r in results])
            n = torch.tensor([float(r.num_examples) for _, r in results])
            delta, info = self.agg(U, n, self.ctx_fn(server_round, self.current))
            self.current = self.current + delta
            metrics = {"n_selected": len(info.get("selected", range(len(results))))}
            return ndarrays_to_parameters(self._unflat(self.current)), metrics
except ImportError:  # Flower is optional for the single-process simulator
    FlowerRobustStrategy = None
