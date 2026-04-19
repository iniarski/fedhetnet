
import functools as ft
import os
import time

import utils.fedjax_compat
import fedjax
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from tqdm import trange

import utils.export
from models.cnn.model import CNN, CNNConfig
from utils.train_utils import forward, init
from utils.lib5gpl2_utils import make_dataset

from utils.fl_utils import build_clients, client_summary, sample_round

N_FEATURES = 33


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def focal_loss(batch, preds, gamma: float = 2.0, eps: float = 1e-7, num_classes: int = 6):
    """Focal softmax cross-entropy. Signature: (batch, preds) per FedJAX convention."""
    y     = batch["y"]
    probs = jax.nn.softmax(preds, axis=-1)
    ohe   = jax.nn.one_hot(y, num_classes=num_classes)
    p_t   = jnp.sum(ohe * probs, axis=-1)
    return -jnp.mean((1 - p_t) ** gamma * jnp.log(p_t + eps))


# ---------------------------------------------------------------------------
# FedJAX model wrapper
# ---------------------------------------------------------------------------

def build_fedjax_model(cnn: CNN, input_dtype, batch_size: int, seq_len: int, n_features: int):
    """
    Wrap ml5g's CNN in a fedjax.Model.

    The full Flax variables dict (params + optional batch_stats) is used as
    FedJAX's 'params' object so ml5g's forward() helper works unmodified.
    """

    def _init(rng, sample_batch):
        return init(cnn, rng, sample_batch["x"], print_summary=True)

    def _apply_for_train(params, batch, rng):
        logits, _ = forward(cnn, params, rng, batch["x"])
        return logits

    def _apply_for_eval(params, batch):
        logits, _ = forward(cnn, params, jax.random.PRNGKey(0), batch["x"])
        return logits

    return fedjax.Model(
        init=_init,
        apply_for_train=_apply_for_train,
        apply_for_eval=_apply_for_eval,
        train_loss=focal_loss,
        eval_metrics={"accuracy": fedjax.metrics.Accuracy()},
    )


# ---------------------------------------------------------------------------
# Centralised validation
# ---------------------------------------------------------------------------

def run_validation(apply_for_eval, variables, X_val, y_val, n_val_steps: int) -> dict:
    y_label, y_pred_list, val_loss_total = [], [], 0.0

    for x, y in zip(X_val[:n_val_steps], y_val[:n_val_steps]):
        batch  = {"x": jnp.expand_dims(x, 0), "y": jnp.expand_dims(y, 0)}
        logits = apply_for_eval(variables, batch)
        val_loss_total += float(focal_loss(batch, logits))
        y_label.append(np.array(y).flatten())
        y_pred_list.append(np.array(logits.argmax(axis=-1)).flatten())

    y_true = np.concatenate(y_label)
    y_hat  = np.concatenate(y_pred_list)

    metrics = {
        "val/loss": val_loss_total / n_val_steps,
        "val/acc":  accuracy_score(y_true, y_hat),
        "val/prec": precision_score(y_true, y_hat, average="macro", zero_division=0),
        "val/rec":  recall_score(y_true, y_hat, average="macro", zero_division=0),
        "val/f1":   f1_score(y_true, y_hat, average="macro", zero_division=0),
    }

    prec_pc, rec_pc, f1_pc, _ = precision_recall_fscore_support(
        y_true, y_hat, labels=list(range(6)), zero_division=0
    )
    for i in range(6):
        metrics[f"val/class_{i}_prec"] = float(prec_pc[i])
        metrics[f"val/class_{i}_rec"]  = float(rec_pc[i])
        metrics[f"val/class_{i}_f1"]   = float(f1_pc[i])

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="configs/", config_name="cnn")
def main(cfg: DictConfig) -> None:
    n_features  = N_FEATURES
    input_dtype = jnp.float32 if cfg.float else jnp.uint8
    mode        = cfg.data.mode

    key      = jax.random.PRNGKey(cfg.train.seed)
    init_key = jax.random.split(key, 1)[0]

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    cnn          = CNN(CNNConfig(**cfg.model))
    fedjax_model = build_fedjax_model(
        cnn, input_dtype,
        batch_size=cfg.train.batch_size,
        seq_len=cfg.train.seq_len,
        n_features=n_features,
    )

    # ------------------------------------------------------------------
    # Dataset loading
    # ------------------------------------------------------------------
    # fl_custom loads data per-client inside build_clients(); the shared
    # train split is only needed for centralized / fl_iid.
    X_train = y_train = None
    if mode in ("centralized", "fl_iid"):
        X_train, y_train, X_val, y_val = make_dataset(
            script_path=cfg.data.dataset_path,
            batch_size=cfg.train.seq_len,
            seed=cfg.train.seed,
            class_sampling=list(cfg.data.get("class_sampling", [0.1, 1, 3, 2, 2, 8])),
        )
    else:  # fl_custom: validation set comes from cfg.data.val_path
        _, _, X_val, y_val = make_dataset(
            script_path=cfg.data.val_path,
            batch_size=cfg.train.seq_len,
            seed=cfg.train.seed,
            class_sampling=list(cfg.data.get("class_sampling", [0.1, 1, 3, 2, 2, 8])),
        )

    # ------------------------------------------------------------------
    # Build federated clients
    # ------------------------------------------------------------------
    clients = build_clients(cfg, X_train, y_train)
    client_summary(clients)

    # ------------------------------------------------------------------
    # FedAvg algorithm
    # ------------------------------------------------------------------
    client_optimizer = fedjax.optimizers.sgd(cfg.data.client_lr, cfg.data.client_momentum)
    server_optimizer = fedjax.optimizers.adam(cfg.data.server_lr)

    algorithm = fedjax.algorithms.fed_avg.federated_averaging(
        grad_fn=fedjax.model_grad(fedjax_model),
        client_optimizer=client_optimizer,
        server_optimizer=server_optimizer,
        client_batch_hparams=fedjax.ShuffleRepeatBatchHParams(
            batch_size=cfg.train.batch_size,
            num_epochs=cfg.data.local_epochs,
        ),
    )

    sample_input = jnp.zeros(
        (cfg.train.batch_size, cfg.train.seq_len, n_features), dtype=input_dtype
    )
    sample_batch = {
        "x": sample_input,
        "y": jnp.zeros((cfg.train.batch_size, cfg.train.seq_len), dtype=jnp.uint8),
    }
    init_params  = fedjax_model.init(init_key, sample_batch)
    server_state = algorithm.init(init_params)

    # ------------------------------------------------------------------
    # Checkpointing & logging
    # ------------------------------------------------------------------
    ckpt_path      = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    ckpt_variables = os.path.join(ckpt_path, "export")
    OmegaConf.save(config=cfg, f=os.path.join(ckpt_path, "config.yaml"))

    wandb_group = {
        "centralized": "byte_model_central",
        "fl_iid":      "byte_model_fl_iid",
        "fl_custom":   "byte_model_fl_custom",
    }[mode]

    if cfg.train.logging == "wandb":
        wandb.require("core")
        wandb.init(
            project="ml5g",
            group=wandb_group,
            name=f'{cfg.name}_{"float" if cfg.float else "byte"}_{mode}',
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    n_rounds        = int(cfg.data.n_rounds)
    clients_per_rnd = int(cfg.data.clients_per_round)

    # ------------------------------------------------------------------
    # Training rounds
    # ------------------------------------------------------------------
    for round_num in trange(n_rounds, desc=f"[{mode}] rounds"):
        log_dict    = {"round": round_num}
        round_start = time.perf_counter()

        # Sample clients for this round
        round_clients = sample_round(
            clients,
            n_per_round=clients_per_rnd,
            round_num=round_num,
            base_seed=cfg.train.seed,
        )

        server_state, client_diagnostics = algorithm.apply(server_state, round_clients)
        log_dict["perf/round_time"] = time.perf_counter() - round_start

        if client_diagnostics:
            losses = [
                float(v.get("loss", v.get("train_loss", 0.0)))
                for v in client_diagnostics.values()
            ]
            log_dict["train/mean_client_loss"] = float(np.mean(losses))
            log_dict["train/n_clients"]         = len(round_clients)

        # ---- Validation ---------------------------------------------
        if round_num % cfg.train.val_freq == 0 or round_num == n_rounds - 1:
            val_start   = time.perf_counter()
            val_metrics = run_validation(
                fedjax_model.apply_for_eval,
                server_state.params,
                X_val, y_val,
                n_val_steps=cfg.train.n_val_steps,
            )
            log_dict.update(val_metrics)
            log_dict["perf/val_time"] = time.perf_counter() - val_start

        # ---- Logging ------------------------------------------------
        if round_num % cfg.train.log_freq == 0 or round_num == n_rounds - 1:
            print(log_dict)
            if cfg.train.logging == "wandb":
                wandb.log(log_dict)

        # ---- Checkpoint ---------------------------------------------
        if round_num % cfg.train.save_freq == 0 or round_num == n_rounds - 1:
            variables = server_state.params
            utils.export.deploy(
                dict(
                    factory=lambda: variables,
                    predict=ft.partial(cnn.apply, training=False),
                ),
                ckpt_variables,
                dict(factory=(), predict=(variables, sample_input)),
            )

    # ------------------------------------------------------------------
    # Optional NPZ dump
    # ------------------------------------------------------------------
    if cfg.save_npz:
        variables = server_state.params
        y_label, y_pred_list = [], []
        for x, y in zip(X_val, y_val):
            batch  = {"x": jnp.expand_dims(x, 0), "y": jnp.expand_dims(y, 0)}
            logits = fedjax_model.apply_for_eval(variables, batch)
            y_label.append(np.array(y))
            y_pred_list.append(np.array(logits))

        np.savez_compressed(
            f"cnn_dilstm_{mode}.npz",
            labels=np.array(y_label),
            rnn_outputs=np.array(y_pred_list),
        )


if __name__ == "__main__":
    main()
