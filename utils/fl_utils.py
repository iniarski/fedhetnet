from __future__ import annotations

import dataclasses
import logging
from typing import List, Optional, Sequence, Tuple

import utils.fedjax_compat
import fedjax
import jax
import numpy as np
from omegaconf import DictConfig

from utils.lib5gpl2_utils import make_dataset

log = logging.getLogger(__name__)

@dataclasses.dataclass
class FLClient:
    client_id: int
    dataset: fedjax.ClientDataset
    n_samples: int = dataclasses.field(init=False)

    def __post_init__(self):
        # ClientDataset wraps a dict; 'y' always present and 1-D or 2-D
        y = self.dataset.all_examples()["y"]
        self.n_samples = int(y.shape[0])

    def __repr__(self) -> str:
        return f"FLClient(id={self.client_id}, n_samples={self.n_samples})"



def make_centralized_client(
    X_list: List[np.ndarray],
    y_list: List[np.ndarray],
) -> List[FLClient]:
    """
    Wrap the entire training set as a single FLClient (client_id=0).

    Running FedAvg with one client that sees the full dataset is identical
    to standard mini-batch SGD, so this mode preserves numerical parity
    with the original centralised training script while reusing the FL loop.
    """
    all_x = np.stack(X_list)  # (N, seq_len, n_features)
    all_y = np.stack(y_list)  # (N, seq_len)

    log.info("Centralized mode: single client with %d samples.", len(all_x))
    return [
        FLClient(
            client_id=0,
            dataset=fedjax.ClientDataset({"x": all_x, "y": all_y}),
        )
    ]



def make_iid_clients(
    X_list: List[np.ndarray],
    y_list: List[np.ndarray],
    n_clients: int,
    seed: int,
) -> List[FLClient]:
    """
    Randomly shuffle and evenly partition the dataset into n_clients shards.

    Because the permutation is global the resulting shards are IID — each
    client's class distribution mirrors the population distribution.

    Args:
        X_list:    List of feature arrays, each shape (seq_len, n_features).
        y_list:    Corresponding label arrays, each shape (seq_len,).
        n_clients: Number of clients to create.
        seed:      NumPy seed for reproducible shuffling.

    Returns:
        List of FLClient, one per shard.
    """
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
        if len(sx) > 0   # drop empty shards that can arise with small datasets
    ]

    log.info(
        "IID FL: %d clients, ~%d samples/client.",
        len(clients),
        len(all_x) // n_clients,
    )
    return clients



def make_custom_clients(
    client_cfgs: Sequence[DictConfig],
    seq_len: int,
    seed: int,
    float_tokenization: bool,
    class_sampling: List[float],
) -> List[FLClient]:
    """
    Load each client's dataset from an independent file path.

    Each entry in client_cfgs must have:
        id    (int)  – unique client identifier
        path  (str)  – path accepted by make_dataset's script_path argument

    Optionally, each entry may override:
        class_sampling  (list[float]) – per-class weights for this client
        seq_len         (int)         – sequence length for this client

    This mirrors a real-world silo-FL scenario where each participating site
    owns a private dataset generated from a different traffic capture.

    Returns:
        List of FLClient, one per entry in client_cfgs.
    """
    clients = []
    for spec in client_cfgs:
        client_class_sampling = list(
            spec.get("class_sampling", class_sampling)
        )
        client_seq_len = int(spec.get("seq_len", seq_len))

        X, y, _, _ = make_dataset(
            script_path=str(spec.path),
            batch_size=client_seq_len,
            seed=seed,
            class_sampling=client_class_sampling,
            float_tokenization=float_tokenization,
        )

        client = FLClient(
            client_id=int(spec.id),
            dataset=fedjax.ClientDataset({
                "x": np.stack(X),
                "y": np.stack(y),
            }),
        )
        clients.append(client)
        log.info("Custom client %d loaded from '%s' (%d samples).",
                 client.client_id, spec.path, client.n_samples)

    return clients


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_clients(
    cfg: DictConfig,
    X_train: Optional[List[np.ndarray]] = None,
    y_train: Optional[List[np.ndarray]] = None,
) -> List[FLClient]:
    """
    Dispatch to the correct client-creation function based on cfg.data.mode.

    Args:
        cfg:     Full Hydra config. Must contain a `data` sub-config with at
                 least a `mode` key.
        X_train: Pre-loaded training feature sequences. Required for
                 `centralized` and `fl_iid` modes; ignored for `fl_custom`.
        y_train: Corresponding labels. Same requirements as X_train.

    Returns:
        List[FLClient] ready to be passed to sample_round().

    Raises:
        ValueError: Unknown mode or missing required arguments.
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
        return make_custom_clients(
            client_cfgs=cfg.data.clients,
            seq_len=int(cfg.train.seq_len),
            seed=int(cfg.train.seed),
            float_tokenization=bool(cfg.float),
            class_sampling=list(cfg.data.get("class_sampling", [0.1, 1, 3, 2, 2, 8])),
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
    """
    Sample up to n_per_round clients for a single FL round.

    For centralized mode (single client) the lone client is always returned
    regardless of n_per_round, so the training loop needs no special-casing.

    Args:
        clients:     Full pool of FLClients.
        n_per_round: Maximum number of clients to sample.
        round_num:   Current round index (used to derive per-round RNG).
        base_seed:   Global seed from cfg.train.seed.

    Returns:
        List of (client_id, ClientDataset, jax.PRNGKey) tuples accepted by
        fedjax algorithm.apply().
    """
    if len(clients) == 1:
        # Centralized: always use the only client
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
# Diagnostics helper
# ---------------------------------------------------------------------------

def client_summary(clients: List[FLClient]) -> None:
    """Print a per-client sample count and basic class distribution."""
    print(f"\n{'─'*55}")
    print(f"  {'Client':>8}  {'Samples':>10}  {'Classes (approx)':>20}")
    print(f"{'─'*55}")
    for c in clients:
        y = c.dataset.all_examples()["y"].flatten()
        unique, counts = np.unique(y, return_counts=True)
        dist = ", ".join(f"{u}:{cnt}" for u, cnt in zip(unique, counts))
        print(f"  {c.client_id:>8}  {c.n_samples:>10}  {dist:>20}")
    print(f"{'─'*55}\n")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _require_xy(X, y, mode: str) -> None:
    if X is None or y is None:
        raise ValueError(
            f"X_train and y_train must be provided for data.mode='{mode}'."
        )
