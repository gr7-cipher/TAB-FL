"""System-efficiency measurements reported in the paper.

* client model size (fp32 and int8-equivalent) and parameter count
* edge inference latency: single-threaded CPU, batch of 1 (median and p95)
* communication volume per round (upload + download, K clients)
* server aggregation time per rule, micro-benchmarked on real client updates
  captured from one CICIoT2023 round (K = 20), including TAB-FL's root-set
  forward passes

Writes ../results/efficiency.json
"""
from __future__ import annotations

import json
import os
import platform
import time

import numpy as np
import torch

import aggregators as A
import train_fl as T
from models import CNNBiGRU, count_params, get_flat, layer_slices

torch.set_num_threads(1)


def latency(model, n_features, reps=2000):
    model.eval()
    x = torch.randn(1, n_features)
    with torch.no_grad():
        for _ in range(100):
            model(x)
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            model(x)
            ts.append(time.perf_counter() - t0)
        xb = torch.randn(256, n_features)
        t0 = time.perf_counter()
        for _ in range(20):
            model(xb)
        per_sample_batched = (time.perf_counter() - t0) / (20 * 256)
    ts = np.array(ts) * 1000
    return dict(median_ms=float(np.median(ts)), p95_ms=float(np.percentile(ts, 95)),
                batched256_us_per_sample=float(per_sample_batched * 1e6))


def agg_benchmark(reps=30):
    cfg = {**T.DEFAULTS, "method": "TAB-FL", "attack": "sign_flip", "rho": 0.3}
    S = T.setup(cfg)
    model, C, K = S["model"], S["n_classes"], cfg["K"]
    Xtr, ytr = S["Xtr"], S["ytr"]
    clean = [(Xtr[p], ytr[p]) for p in S["parts"]]
    w = get_flat(model)
    rng = np.random.default_rng(0)
    U = T.train_clients(model, w, clean, cfg["local_steps"], cfg["batch"], cfg["lr"],
                        cfg["momentum"], rng, opt=cfg["opt"]) - w
    mal = torch.from_numpy(S["mal"])
    U[mal] = -3.0 * U[mal]
    probe = CNNBiGRU(Xtr.shape[1], C)
    rl = T.make_root_losses(probe, torch.from_numpy(Xtr[S["root"]]), torch.from_numpy(ytr[S["root"]]), C)
    ctx = {"f": 6, "slices": layer_slices(model), "root_losses": lambda U_: rl(torch.cat([w[None], w + U_]))}
    n = torch.ones(K)
    out = {}
    for name in T.METHODS:
        agg = A.build(name)
        agg.reset(K)
        agg(U, n, ctx)
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            agg(U, n, ctx)
            ts.append(time.perf_counter() - t0)
        out[name] = float(np.median(ts) * 1000)
    return out


def main():
    res = {"host": f"{platform.machine()} / {platform.system()} {platform.release()}, 1 CPU thread",
           "torch": torch.__version__}
    for ds, F, C in [("ciciot2023", 39, 8), ("toniot", 26, 8)]:
        m = CNNBiGRU(F, C)
        d = count_params(m)
        res[ds] = dict(n_features=F, params=d, size_kb_fp32=d * 4 / 1024, size_kb_int8=d / 1024,
                       comm_mb_per_round=2 * 20 * d * 4 / 1e6,
                       comm_mb_per_client_round=2 * d * 4 / 1e6,
                       latency=latency(m, F))
    res["aggregation_ms"] = agg_benchmark()
    p = os.path.join(T.HERE, "..", "results", "efficiency.json")
    with open(p, "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
