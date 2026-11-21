"""Lightweight 1D-CNN + BiGRU edge classifier used by every federated client.

The bidirectional GRU is written with plain tensor operations (same gate
equations and initialisation as ``torch.nn.GRU``) so that the whole model can
be vectorised over clients with ``torch.func.vmap`` during simulation.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class BiGRU(nn.Module):
    def __init__(self, input_size: int, hidden: int):
        super().__init__()
        self.hidden = hidden
        k = 1.0 / math.sqrt(hidden)
        for d in ("f", "b"):
            setattr(self, f"w_ih_{d}", nn.Parameter(torch.empty(3 * hidden, input_size).uniform_(-k, k)))
            setattr(self, f"w_hh_{d}", nn.Parameter(torch.empty(3 * hidden, hidden).uniform_(-k, k)))
            setattr(self, f"b_ih_{d}", nn.Parameter(torch.empty(3 * hidden).uniform_(-k, k)))
            setattr(self, f"b_hh_{d}", nn.Parameter(torch.empty(3 * hidden).uniform_(-k, k)))

    def _run(self, x, w_ih, w_hh, b_ih, b_hh, reverse):
        H = self.hidden
        gi = x @ w_ih.t() + b_ih                     # (B, L, 3H)
        h = torch.zeros(x.shape[0], H, dtype=x.dtype)
        steps = range(x.shape[1] - 1, -1, -1) if reverse else range(x.shape[1])
        outs = []
        for t in steps:
            gh = h @ w_hh.t() + b_hh
            r = torch.sigmoid(gi[:, t, :H] + gh[:, :H])
            z = torch.sigmoid(gi[:, t, H:2 * H] + gh[:, H:2 * H])
            n = torch.tanh(gi[:, t, 2 * H:] + r * gh[:, 2 * H:])
            h = (1 - z) * n + z * h
            outs.append(h)
        if reverse:
            outs = outs[::-1]
        return torch.stack(outs, 1), h

    def forward(self, x):
        of, hf = self._run(x, self.w_ih_f, self.w_hh_f, self.b_ih_f, self.b_hh_f, False)
        ob, hb = self._run(x, self.w_ih_b, self.w_hh_b, self.b_ih_b, self.b_hh_b, True)
        return torch.cat([of, ob], 2), torch.cat([hf, hb], 1)


class CNNBiGRU(nn.Module):
    """Treats the tabular feature vector as a 1-channel sequence of length F.

    conv(1->16, k=3, stride 2) -> ReLU -> conv(16->32, k=3, stride 2) -> ReLU
    -> BiGRU(hidden 24) -> [final fwd/bwd states, temporal mean] -> MLP head.
    For F = 39 input features the GRU therefore runs over 10 positions.
    No batch normalisation is used, so there are no running statistics that
    would need separate handling during federated averaging.
    """

    def __init__(self, n_features: int, n_classes: int, c1: int = 16, c2: int = 32,
                 hidden: int = 24, head: int = 64):
        super().__init__()
        self.n_features = n_features
        self.conv1 = nn.Conv1d(1, c1, kernel_size=3, padding=1, stride=2)
        self.conv2 = nn.Conv1d(c1, c2, kernel_size=3, padding=1, stride=2)
        self.gru = BiGRU(c2, hidden)
        self.fc1 = nn.Linear(4 * hidden, head)
        self.fc2 = nn.Linear(head, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.conv1(x.unsqueeze(1)))
        h = torch.relu(self.conv2(h))                 # (B, c2, L)
        seq, last = self.gru(h.transpose(1, 2))       # (B, L, 2H), (B, 2H)
        z = torch.cat([last, seq.mean(1)], 1)
        return self.fc2(torch.relu(self.fc1(z)))


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def layer_slices(model: nn.Module) -> list[tuple[str, slice]]:
    """Index ranges of each parameter tensor inside the flattened vector."""
    out, start = [], 0
    for name, p in model.named_parameters():
        out.append((name, slice(start, start + p.numel())))
        start += p.numel()
    return out


def get_flat(model: nn.Module) -> torch.Tensor:
    return torch.nn.utils.parameters_to_vector(model.parameters()).detach().clone()


def set_flat(model: nn.Module, vec: torch.Tensor) -> None:
    torch.nn.utils.vector_to_parameters(vec, model.parameters())


def unflatten_batch(model: nn.Module, flat: torch.Tensor) -> dict[str, torch.Tensor]:
    """(K, d) flat parameter matrix -> dict of (K, *shape) tensors."""
    out, s = {}, 0
    for name, p in model.named_parameters():
        k = p.numel()
        out[name] = flat[:, s:s + k].reshape(flat.shape[0], *p.shape)
        s += k
    return out


def flatten_batch(params: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([v.reshape(v.shape[0], -1) for v in params.values()], 1)


def batched_forward(P: dict[str, torch.Tensor], X: torch.Tensor) -> torch.Tensor:
    """Forward pass of K independent CNNBiGRU models in one program.

    ``P`` maps parameter names to (K, *shape) tensors and ``X`` is (K, B, F).
    Grouped convolutions and batched matrix products keep the K models fully
    separate, so the gradient of sum_k loss_k w.r.t. P[.][k] equals the
    gradient each client would compute on its own.  Returns (K, B, C) logits.
    """
    K, B, Fdim = X.shape
    w1, b1 = P["conv1.weight"], P["conv1.bias"]          # (K, c1, 1, 3), (K, c1)
    w2, b2 = P["conv2.weight"], P["conv2.bias"]          # (K, c2, c1, 3), (K, c2)
    c1, c2 = w1.shape[1], w2.shape[1]
    h = X.transpose(0, 1)                                  # (B, K, F)
    h = torch.relu(torch.nn.functional.conv1d(h, w1.reshape(K * c1, 1, 3), b1.reshape(-1),
                                              padding=1, stride=2, groups=K))
    h = torch.relu(torch.nn.functional.conv1d(h, w2.reshape(K * c2, c1, 3), b2.reshape(-1),
                                              padding=1, stride=2, groups=K))
    L = h.shape[-1]
    h = h.reshape(B, K, c2, L).permute(1, 0, 3, 2)         # (K, B, L, c2)
    H = P["gru.w_hh_f"].shape[-1]
    outs, finals = [], []
    for d, reverse in (("f", False), ("b", True)):
        w_ih, w_hh = P[f"gru.w_ih_{d}"], P[f"gru.w_hh_{d}"]
        b_ih, b_hh = P[f"gru.b_ih_{d}"], P[f"gru.b_hh_{d}"]
        gi = torch.einsum("kblc,kgc->kblg", h, w_ih) + b_ih[:, None, None, :]
        hs = torch.zeros(K, B, H, dtype=X.dtype, device=X.device)
        seq = []
        steps = range(L - 1, -1, -1) if reverse else range(L)
        for t in steps:
            gh = torch.bmm(hs, w_hh.transpose(1, 2)) + b_hh[:, None, :]
            g = gi[:, :, t, :]
            r = torch.sigmoid(g[..., :H] + gh[..., :H])
            z = torch.sigmoid(g[..., H:2 * H] + gh[..., H:2 * H])
            n = torch.tanh(g[..., 2 * H:] + r * gh[..., 2 * H:])
            hs = (1 - z) * n + z * hs
            seq.append(hs)
        if reverse:
            seq = seq[::-1]
        outs.append(torch.stack(seq, 2))                   # (K, B, L, H)
        finals.append(hs)
    seq = torch.cat(outs, -1)
    z = torch.cat([finals[0], finals[1], seq.mean(2)], -1)  # (K, B, 4H)
    z = torch.relu(torch.bmm(z, P["fc1.weight"].transpose(1, 2)) + P["fc1.bias"][:, None, :])
    return torch.bmm(z, P["fc2.weight"].transpose(1, 2)) + P["fc2.bias"][:, None, :]
