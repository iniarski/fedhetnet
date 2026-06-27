"""
fl_utils.py – Federated Learning client creation utilities.

Supports three data modes controlled via cfg.data.mode:

  centralized   Full dataset as a single pseudo-client.
  fl_iid        Dataset randomly partitioned IID across n_clients.
  fl_custom     Each client owns its own file; a val_fraction of every
                client's data is held out and pooled into a shared
                validation set. No val_path required.

Public API
----------
  build_clients(cfg, X_train, y_train)
      -> (List[FLClient], X_val | None, y_val | None)

  sample_round(clients, n, round_num, base_seed)
      -> FedJAX client tuples

  client_summary(clients)
      -> prints per-client statistics
"""

from __future__ import annotations

import dataclasses
import logging
from typing import List, Optional, Sequence, Tuple

import fedjax
import jax
import numpy as np
from omegaconf import DictConfig

from utils.lib5gpl2_utils import make_dataset, per_file_split

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class FLClient:
    client_id: int
    x: np.ndarray   # raw arrays, not a ClientDataset
    y: np.ndarray
    n_samples: int = dataclasses.field(init=False)

    def __post_init__(self):
        self.n_samples = int(self.y.shape[0])

    def make_dataset(self) -> fedjax.ClientDataset:
        return fedjax.ClientDataset({"x": self.x, "y": self.y})


def _compute_stats(X_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    n_features = X_list[0].shape[-1]
    mean  = np.zeros(n_features, dtype=np.float64)
    M2    = np.zeros(n_features, dtype=np.float64)
    count = 0
    for x in X_list:
        flat       = x.reshape(-1, n_features).astype(np.float64)
        n          = len(flat)
        batch_mean = flat.mean(axis=0)
        delta      = batch_mean - mean
        new_count  = count + n
        mean       = (count * mean + n * batch_mean) / new_count
        M2        += flat.var(axis=0) * n + delta ** 2 * count * n / new_count
        count      = new_count
    std = np.sqrt(M2 / count) + 1e-8
    return mean.astype(np.float32), std.astype(np.float32)


def _normalize_inplace(X_list: List[np.ndarray],
                        mean: np.ndarray,
                        std: np.ndarray) -> None:
    safe_std = np.where(std < 1e-8, 1.0, std)
    for i in range(len(X_list)):
        normalized = (X_list[i].astype(np.float32) - mean) / safe_std
        normalized = np.clip(normalized, -65504, 65504)
        X_list[i] = normalized.astype(np.float16)


# ---------------------------------------------------------------------------
# Mode: fl_iid
# ---------------------------------------------------------------------------

def make_iid_clients(
    X_list: List[np.ndarray],
    y_list: List[np.ndarray],
    n_clients: int,
    seed: int,
    X_val: List[np.ndarray] = None,
    y_val: List[np.ndarray] = None,
) -> Tuple[List[FLClient], List[np.ndarray], List[np.ndarray]]:
    n_total = len(X_list)
    rng     = np.random.default_rng(seed)
    perm    = rng.permutation(n_total)
    client_indices = np.array_split(perm, n_clients)

    # Compute normalization stats from client 0's partition only,
    # then apply the same stats to all clients and val so every
    # client trains in an identical feature space.
    first_valid = next(idx for idx in client_indices if len(idx) > 0)
    ref_X = [X_list[j] for j in first_valid]
    ref_mean, ref_std = _compute_stats(ref_X)
    log.info(
        "IID FL: normalization stats from client-0 partition "
        "(%d sequences). Applying to all clients and val.", len(first_valid)
    )

    clients = []
    for i, idx in enumerate(client_indices):
        if len(idx) == 0:
            continue
        sx = np.stack([X_list[j] for j in idx])
        sy = np.stack([y_list[j] for j in idx])

        # normalize with reference stats (safe clip included)
        sx_list = list(sx)
        _normalize_inplace(sx_list, ref_mean, ref_std)
        sx = np.stack(sx_list)

        clients.append(FLClient(
            client_id=i,
            x=sx,
            y=sy
        ))

    if X_val is not None:
        _normalize_inplace(X_val, ref_mean, ref_std)

    log.info("IID FL: %d clients, ~%d samples/client.",
             len(clients), n_total // n_clients)
    return clients, X_val, y_val


# ---------------------------------------------------------------------------
# Mode: fl_custom
# ---------------------------------------------------------------------------

def make_custom_clients(client_cfgs, seq_len, seed, class_sampling):
    clients    = []
    X_val_pool = []
    y_val_pool = []

    # Load client 0 first to compute the reference normalization stats,
    # then re-use those stats for every subsequent client and val split.
    ref_mean: Optional[np.ndarray] = None
    ref_std:  Optional[np.ndarray] = None

    for spec in client_cfgs:
        client_class_sampling = list(spec.get("class_sampling", class_sampling))
        client_seq_len        = int(spec.get("seq_len", seq_len))
        client_split        = int(spec.get("split", 1))
        client_id        = int(spec.get("id", 42))

        splits = per_file_split(str(spec.path), client_split, client_id)
        
        for i, split in enumerate(splits):
            X_train, y_train, X_val, y_val = make_dataset(
                data_source=split,
                batch_size=client_seq_len,
                seed=seed,
                class_sampling=client_class_sampling,
            )

            if ref_mean is None:
                # First client — compute reference stats and log them
                ref_mean, ref_std = _compute_stats(X_train)
                log.info(
                    "Custom FL: normalization stats from client %s "
                    "(%d sequences). Applying to all clients and val.",
                    100 * client_id + i, len(X_train),
                )

            # Normalize train and val with the shared reference stats
            _normalize_inplace(X_train, ref_mean, ref_std)
            _normalize_inplace(X_val,   ref_mean, ref_std)

            X_val_pool.extend(X_val)
            y_val_pool.extend(y_val)

            clients.append(FLClient(
                client_id=100 * client_id + i,
                x=np.stack(X_train),
                y=np.stack(y_train),
            ))

    return clients, X_val_pool, y_val_pool
# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_clients(
    cfg: DictConfig,
    X_train: Optional[List[np.ndarray]] = None,
    y_train: Optional[List[np.ndarray]] = None,
    X_val: Optional[List[np.ndarray]] = None,
    y_val: Optional[List[np.ndarray]] = None,
) -> Tuple[List[FLClient], Optional[List[np.ndarray]], Optional[List[np.ndarray]]]:
    mode = cfg.data.mode

    if mode == "centralized":
        _require_xy(X_train, y_train, mode)
        return make_centralized_client(X_train, y_train)

    elif mode == "fl_iid":
        _require_xy(X_train, y_train, mode)
        return make_iid_clients(
            X_train, y_train,
            n_clients=int(cfg.data.n_clients),
            seed=int(cfg.train.seed),
            X_val=X_val,
            y_val=y_val
        )

    elif mode == "fl_custom":
        return make_custom_clients(
            client_cfgs=cfg.data.clients,
            seq_len=int(cfg.train.seq_len),
            seed=int(cfg.train.seed),
            class_sampling=list(cfg.data.get("class_sampling", [0.01, 1, 3, 2, 2, 8])),
        )

    else:
        raise ValueError(
            f"Unknown data.mode '{mode}'. "
            "Valid choices: centralized | fl_iid | fl_custom"
        )


# ---------------------------------------------------------------------------
# Round sampling
# ---------------------------------------------------------------------------

def sample_round(
    clients: List[FLClient],
    n_per_round: int,
    round_num: int,
    base_seed: int,
) -> List[Tuple[int, fedjax.ClientDataset, jax.Array]]:
    if len(clients) == 1:
        prng = jax.random.PRNGKey(base_seed + round_num)
        return [(clients[0].client_id, clients[0].make_dataset(), prng)]

    rng     = np.random.default_rng(base_seed + round_num)
    indices = rng.choice(
        len(clients),
        size=min(n_per_round, len(clients)),
        replace=False,
    )
    return [
        (
            clients[i].client_id,
            clients[i].make_dataset(),
            jax.random.PRNGKey(base_seed + round_num * len(clients) + i),
        )
        for i in indices
    ]


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def client_summary(clients: List[FLClient]) -> None:
    """Print per-client sample count and class distribution."""
    print(f"\n{'─'*65}")
    print(f"  {'Client':>8}  {'Train samples':>14}  Class distribution")
    print(f"{'─'*65}")
    for c in clients:
        y      = c.y.flatten()
        total  = len(y)
        unique, counts = np.unique(y, return_counts=True)
        dist   = "  ".join(
            f"cls{u}:{cnt}({100*cnt/total:.0f}%)"
            for u, cnt in zip(unique, counts)
        )
        print(f"  {c.client_id:>8}  {c.n_samples:>14}  {dist}")
    print(f"{'─'*65}\n")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _require_xy(X, y, mode: str) -> None:
    if X is None or y is None:
        raise ValueError(
            f"X_train and y_train must be provided for data.mode='{mode}'."
        )


def normalize_centralized(
    X_train: List[np.ndarray],
    X_val:   List[np.ndarray],
) -> None:
    """Normalize train in-place, apply same stats to val. Call after make_dataset."""
    mean, std = _compute_stats(X_train)
    _normalize_inplace(X_train, mean, std)
    _normalize_inplace(X_val,   mean, std)
