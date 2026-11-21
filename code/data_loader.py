"""Dataset preparation and federated partitioning for TAB-FL.

Two real datasets are supported:

* CICIoT2023 (Neto et al., 2023): flow features extracted from traffic of
  105 physical IoT devices.  We read a subset of the official MergedXX.csv
  files, map the 34 fine-grained labels to the 8 attack categories used in
  the original paper, and draw a class-capped random subsample.
* TON_IoT telemetry (Moustafa, 2021; Alsaedi et al., 2020): the
  Train_Test_IoT_* sensor files (fridge, GPS tracker, garage door, Modbus,
  motion light, thermostat, weather).  Each device type has its own sensor
  fields, so we build a union feature space with a device indicator.

Run ``python data_loader.py --raw ../data/raw --out ../data/processed``
once to create the processed .npz files used by ``train_fl.py``.
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd

CIC_CLASSES = ["Benign", "DDoS", "DoS", "Mirai", "Recon", "Spoofing", "Web", "BruteForce"]
TON_CLASSES = ["normal", "backdoor", "ddos", "injection", "password", "ransomware", "scanning", "xss"]
TON_DEVICES = ["Fridge", "GPS_Tracker", "Garage_Door", "Modbus", "Motion_Light", "Thermostat", "Weather"]


def cic_category(label: str) -> str:
    """Map a CICIoT2023 fine label to one of the 8 categories."""
    s = label.upper()
    if s == "BENIGN":
        return "Benign"
    if s.startswith("DDOS"):
        return "DDoS"
    if s.startswith("DOS"):
        return "DoS"
    if s.startswith("MIRAI"):
        return "Mirai"
    if s.startswith("RECON") or s == "VULNERABILITYSCAN":
        return "Recon"
    if s in ("MITM-ARPSPOOFING", "DNS_SPOOFING"):
        return "Spoofing"
    if s == "DICTIONARYBRUTEFORCE":
        return "BruteForce"
    if s in ("BROWSERHIJACKING", "XSS", "UPLOADING_ATTACK", "SQLINJECTION",
             "COMMANDINJECTION", "BACKDOOR_MALWARE"):
        return "Web"
    raise ValueError(f"unknown label {label}")


def _signed_log(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


def _split_and_scale(X, y, groups, seed, test_frac=0.2):
    """Stratified train/test split, then standardise with train statistics."""
    rng = np.random.default_rng(seed)
    tr_idx, te_idx = [], []
    strat = groups.astype(np.int64) * 1000 + y
    for g in np.unique(strat):
        idx = np.flatnonzero(strat == g)
        rng.shuffle(idx)
        n_te = int(round(test_frac * len(idx)))
        te_idx.append(idx[:n_te])
        tr_idx.append(idx[n_te:])
    tr_idx = np.concatenate(tr_idx)
    te_idx = np.concatenate(te_idx)
    rng.shuffle(tr_idx)
    rng.shuffle(te_idx)
    mu = X[tr_idx].mean(0)
    sd = X[tr_idx].std(0)
    sd[sd < 1e-8] = 1.0
    Xs = ((X - mu) / sd).astype(np.float32)
    return Xs, tr_idx, te_idx


def prepare_ciciot(raw_dir: str, out_path: str, files=("01", "02", "03", "04"),
                   cap_per_class: int = 20000, seed: int = 0) -> None:
    frames = []
    for f in files:
        d = pd.read_csv(os.path.join(raw_dir, f"Merged{f}.csv"))
        d["cat"] = d["Label"].map(cic_category)
        frames.append(d.drop(columns=["Label"]))
    df = pd.concat(frames, ignore_index=True)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    rng = np.random.default_rng(seed)
    keep = []
    for c in CIC_CLASSES:
        idx = np.flatnonzero(df["cat"].values == c)
        if len(idx) > cap_per_class:
            idx = rng.choice(idx, cap_per_class, replace=False)
        keep.append(idx)
    df = df.iloc[np.concatenate(keep)].reset_index(drop=True)
    feat_names = [c for c in df.columns if c != "cat"]
    X = _signed_log(df[feat_names].to_numpy(np.float64))
    y = df["cat"].map({c: i for i, c in enumerate(CIC_CLASSES)}).to_numpy(np.int64)
    groups = np.zeros(len(y), dtype=np.int64)
    Xs, tr, te = _split_and_scale(X, y, groups, seed)
    np.savez_compressed(out_path, X_train=Xs[tr], y_train=y[tr], X_test=Xs[te], y_test=y[te],
                        g_train=groups[tr], g_test=groups[te],
                        feature_names=np.array(feat_names), class_names=np.array(CIC_CLASSES))
    print("CICIoT2023:", Xs.shape, "train", len(tr), "test", len(te),
          dict(zip(CIC_CLASSES, np.bincount(y, minlength=8).tolist())))


def _ton_device_frame(path: str, device: str) -> pd.DataFrame:
    d = pd.read_csv(path, encoding="utf-8-sig")
    d.columns = [c.strip() for c in d.columns]
    out = pd.DataFrame(index=d.index)
    if device == "Fridge":
        out["fridge_temperature"] = d["fridge_temperature"].astype(float)
        out["temp_condition_high"] = (d["temp_condition"].astype(str).str.strip().str.lower() == "high").astype(float)
    elif device == "GPS_Tracker":
        out["latitude"] = d["latitude"].astype(float)
        out["longitude"] = d["longitude"].astype(float)
    elif device == "Garage_Door":
        out["door_open"] = (d["door_state"].astype(str).str.strip().str.lower() == "open").astype(float)
        sig = d["sphone_signal"].astype(str).str.strip().str.lower()
        out["sphone_signal"] = sig.isin(["1", "true"]).astype(float)
    elif device == "Modbus":
        for c in ["FC1_Read_Input_Register", "FC2_Read_Discrete_Value",
                  "FC3_Read_Holding_Register", "FC4_Read_Coil"]:
            out[c] = d[c].astype(float)
    elif device == "Motion_Light":
        out["motion_status"] = d["motion_status"].astype(float)
        out["light_on"] = (d["light_status"].astype(str).str.strip().str.lower() == "on").astype(float)
    elif device == "Thermostat":
        out["current_temperature"] = d["current_temperature"].astype(float)
        out["thermostat_status"] = d["thermostat_status"].astype(float)
    elif device == "Weather":
        out["temperature"] = d["temperature"].astype(float)
        out["pressure"] = d["pressure"].astype(float)
        out["humidity"] = d["humidity"].astype(float)
    t = pd.to_datetime(d["time"].astype(str).str.strip(), format="%H:%M:%S", errors="coerce")
    hour = (t.dt.hour + t.dt.minute / 60.0).fillna(0.0)
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    out["type"] = d["type"].astype(str).str.strip().str.lower()
    out["device"] = device
    return out


def prepare_toniot(raw_dir: str, out_path: str, seed: int = 0) -> None:
    frames = [_ton_device_frame(os.path.join(raw_dir, f"Train_Test_IoT_{dev}.csv"), dev)
              for dev in TON_DEVICES]
    df = pd.concat(frames, ignore_index=True)
    sensor_cols = [c for c in df.columns if c not in ("type", "device", "hour_sin", "hour_cos")]
    dev_id = df["device"].map({d: i for i, d in enumerate(TON_DEVICES)}).to_numpy(np.int64)
    # Standardise each sensor field over the rows of the device that owns it,
    # then fill the fields of other devices with zero.
    Xsens = np.zeros((len(df), len(sensor_cols)))
    for j, c in enumerate(sensor_cols):
        col = df[c].to_numpy(np.float64)
        m = ~np.isnan(col)
        v = col[m]
        sd = v.std() if v.std() > 1e-8 else 1.0
        Xsens[m, j] = (v - v.mean()) / sd
    onehot = np.eye(len(TON_DEVICES))[dev_id]
    X = np.concatenate([Xsens, df[["hour_sin", "hour_cos"]].to_numpy(), onehot], axis=1)
    feat_names = sensor_cols + ["hour_sin", "hour_cos"] + [f"dev_{d}" for d in TON_DEVICES]
    y = df["type"].map({c: i for i, c in enumerate(TON_CLASSES)}).to_numpy(np.int64)
    Xs, tr, te = _split_and_scale(X, y, dev_id, seed)
    np.savez_compressed(out_path, X_train=Xs[tr], y_train=y[tr], X_test=Xs[te], y_test=y[te],
                        g_train=dev_id[tr], g_test=dev_id[te],
                        feature_names=np.array(feat_names), class_names=np.array(TON_CLASSES))
    print("TON_IoT:", Xs.shape, "train", len(tr), "test", len(te),
          dict(zip(TON_CLASSES, np.bincount(y, minlength=8).tolist())))


# ---------------------------------------------------------------- partitioning

def dirichlet_partition(y: np.ndarray, n_clients: int, beta: float, seed: int,
                        min_size: int = 64) -> list[np.ndarray]:
    """Label-skewed Dirichlet split (Hsu et al., 2019)."""
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    for _ in range(1000):
        parts = [[] for _ in range(n_clients)]
        for c in classes:
            idx = np.flatnonzero(y == c)
            rng.shuffle(idx)
            p = rng.dirichlet(beta * np.ones(n_clients))
            cuts = (np.cumsum(p) * len(idx)).astype(int)[:-1]
            for k, chunk in enumerate(np.split(idx, cuts)):
                parts[k].extend(chunk.tolist())
        if min(len(p) for p in parts) >= min_size:
            break
    return [np.array(sorted(p), dtype=np.int64) for p in parts]


def device_partition(groups: np.ndarray, n_clients: int, seed: int) -> list[np.ndarray]:
    """Infrastructure partition: every client holds data of exactly one device type.

    Clients are allocated to device types in proportion to their sample counts
    (at least two per type) and each device's rows are split evenly among them.
    """
    rng = np.random.default_rng(seed)
    devs = np.unique(groups)
    counts = np.array([np.sum(groups == d) for d in devs], dtype=float)
    alloc = np.maximum(2, np.floor(counts / counts.sum() * n_clients)).astype(int)
    while alloc.sum() < n_clients:
        alloc[np.argmax(counts / alloc)] += 1
    while alloc.sum() > n_clients:
        alloc[np.argmax(alloc)] -= 1
    parts = []
    for d, a in zip(devs, alloc):
        idx = np.flatnonzero(groups == d)
        rng.shuffle(idx)
        parts.extend(np.array_split(idx, a))
    return [np.sort(p) for p in parts]


def root_split(y: np.ndarray, per_class: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Carve a small class-balanced root (anchor) set out of the training pool."""
    rng = np.random.default_rng(seed + 7)
    root = []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        root.extend(rng.choice(idx, min(per_class, len(idx) // 10), replace=False).tolist())
    root = np.array(sorted(root), dtype=np.int64)
    rest = np.setdiff1d(np.arange(len(y)), root)
    return root, rest


def load_processed(path: str) -> dict:
    d = np.load(path, allow_pickle=False)
    return {k: d[k] for k in d.files}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="../data/raw")
    ap.add_argument("--out", default="../data/processed")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    prepare_ciciot(os.path.join(a.raw, "ciciot"), os.path.join(a.out, "ciciot2023.npz"))
    prepare_toniot(os.path.join(a.raw, "toniot"), os.path.join(a.out, "toniot.npz"))
