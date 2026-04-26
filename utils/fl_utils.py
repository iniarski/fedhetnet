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

from utils.lib5gpl2_utils import make_dataset

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class FLClient:
    """Holds a single client's identity and local training dataset."""
    client_id: int
    dataset:   fedjax.ClientDataset
    n_samples: int = dataclasses.field(init=False)

    def __post_init__(self):
        y = self.dataset.all_examples()["y"]
        self.n_samples = int(y.shape[0])

    def __repr__(self) -> str:
        return f"FLClient(id={self.client_id}, n_samples={self.n_samples})"


# ---------------------------------------------------------------------------
# Mode: centralized
# ---------------------------------------------------------------------------

def make_centralized_client(
    X_list: List[np.ndarray],
    y_list: List[np.ndarray],
) -> Tuple[List[FLClient], None, None]:
    all_x = np.stack(X_list)
    all_y = np.stack(y_list)
    log.info("Centralized mode: single client with %d samples.", len(all_x))
    clients = [FLClient(
        client_id=0,
        dataset=fedjax.ClientDataset({"x": all_x, "y": all_y}),
    )]
    return clients, None, None


# ---------------------------------------------------------------------------
# Mode: fl_iid
# ---------------------------------------------------------------------------

def make_iid_clients(
    X_list: List[np.ndarray],
    y_list: List[np.ndarray],
    n_clients: int,
    seed: int,
) -> Tuple[List[FLClient], None, None]:
    all_x = np.stack(X_list)
    all_y = np.stack(y_list)

    rng  = np.random.default_rng(seed)
    perm = rng.permutation(len(all_x))

    splits_x = np.array_split(all_x[perm], n_clients)
    splits_y = np.array_split(all_y[perm], n_clients)

    clients = [
        FLClient(
            client_id=i,
            dataset=fedjax.ClientDataset({"x": sx, "y": sy}),
        )
        for i, (sx, sy) in enumerate(zip(splits_x, splits_y))
        if len(sx) > 0
    ]
    log.info("IID FL: %d clients, ~%d samples/client.",
             len(clients), len(all_x) // n_clients)
    return clients, None, None


# ---------------------------------------------------------------------------
# Mode: fl_custom
# ---------------------------------------------------------------------------

def make_custom_clients(client_cfgs, seq_len, seed, class_sampling, val_fraction=0.2):
    clients    = []
    X_val_pool = []
    y_val_pool = []

    rng = np.random.default_rng(seed)

    for spec in client_cfgs:
        client_class_sampling = list(spec.get("class_sampling", class_sampling))
        client_seq_len        = int(spec.get("seq_len", seq_len))

        # Use make_dataset's own split — X_val is a proper holdout
        X_train, y_train, X_val, y_val = make_dataset(
            script_path=str(spec.path),
            batch_size=client_seq_len,
            seed=seed,
            class_sampling=client_class_sampling,
        )

        # Pool the proper val splits across clients
        X_val_pool.extend(X_val)
        y_val_pool.extend(y_val)

        client = FLClient(
            client_id=int(spec.id),
            dataset=fedjax.ClientDataset({
                "x": np.stack(X_train),
                "y": np.stack(y_train),
            }),
        )
        clients.append(client)

    return clients, X_val_pool, y_val_pool
# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_clients(
    cfg: DictConfig,
    X_train: Optional[List[np.ndarray]] = None,
    y_train: Optional[List[np.ndarray]] = None,
) -> Tuple[List[FLClient], Optional[List[np.ndarray]], Optional[List[np.ndarray]]]:
    """
    Dispatch to the correct client-creation function.

    Returns:
        (clients, X_val, y_val)

        For centralized / fl_iid modes, X_val and y_val are None — the
        caller is expected to supply a pre-loaded val set (from make_dataset).
        For fl_custom, X_val and y_val are the pooled held-out sequences.
    """
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
        )

    elif mode == "fl_custom":
        val_fraction = float(cfg.data.get("val_fraction", 0.2))
        return make_custom_clients(
            client_cfgs=cfg.data.clients,
            seq_len=int(cfg.train.seq_len),
            seed=int(cfg.train.seed),
            class_sampling=list(cfg.data.get("class_sampling", [0.01, 1, 3, 2, 2, 8])),
            val_fraction=val_fraction,
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
        return [(clients[0].client_id, clients[0].dataset, prng)]

    rng     = np.random.default_rng(base_seed + round_num)
    indices = rng.choice(
        len(clients),
        size=min(n_per_round, len(clients)),
        replace=False,
    )
    return [
        (
            clients[i].client_id,
            clients[i].dataset,
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
        y      = c.dataset.all_examples()["y"].flatten()
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
