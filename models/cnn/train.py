import functools as ft
import os
import time

import utils.fedjax_compat  # must come before fedjax – patches jax.tree_* aliases
import fedjax
import hydra
import jax
import jax.numpy as jnp
import optax
import numpy as np
import tensorflow as tf
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
from algorithms.scaffold import scaffold
from utils.data.csvlogger import CSVLogger
from models.cnn.model import CNN, CNNConfig
from utils.train_utils import forward, get_dataset, get_optimizer, gradient_step, init
from utils.lib5gpl2_utils import make_dataset
from utils.fl_utils import build_clients, client_summary, sample_round, normalize_centralized

N_FEATURES = 33
N_CLASSES  = 7


# ---------------------------------------------------------------------------
# CSV logging helpers
# ---------------------------------------------------------------------------

def make_fieldnames(mode: str) -> list[str]:
    """
    Return the complete, ordered list of CSV column names for a given mode.
    All columns are declared upfront so the header is written once and every
    row has the same shape — missing values on non-val steps are written as "".
    """
    # Step/round counter — name differs per mode but mapped to same column
    index_col = ["step"] if mode == "centralized" else ["round"]

    train_cols = (
        ["train/loss"]                 if mode == "centralized" else
        ["train/mean_delta_l2_norm",
         "train/max_delta_l2_norm",
         "train/n_clients"]
    )

    val_cols = [
        "val/loss", "val/acc", "val/prec", "val/rec", "val/f1",
        *[f"val/class_{i}_{m}"
          for i in range(N_CLASSES)
          for m in ("prec", "rec", "f1")],
    ]

    perf_cols = (
        ["perf/step_time", "perf/val_time"] if mode == "centralized" else
        ["perf/round_time", "perf/val_time"]
    )

    return index_col + train_cols + val_cols + perf_cols


def csv_log(logger: CSVLogger, log_dict: dict) -> None:
    """Write log_dict to CSV, filling any missing fieldnames with empty string."""
    row = {k: log_dict.get(k, "") for k in logger.fieldnames}
    logger.log(row)


# ---------------------------------------------------------------------------
# Shared loss  (focal softmax cross-entropy)
# ---------------------------------------------------------------------------

def focal_loss_fn(model, variables, key, x, y,
                  gamma: float = 2.0, eps: float = 1e-7):
    logits, state = forward(model, variables, key, *x)
    probs    = jax.nn.softmax(logits, axis=-1)
    ohe      = jax.nn.one_hot(y, num_classes=N_CLASSES)
    p_t      = jnp.sum(ohe * probs, axis=-1)
    loss     = -jnp.mean((1 - p_t) ** gamma * jnp.log(p_t + eps))
    return loss, state


def focal_loss_fedjax(batch, preds, gamma: float = 3.0, eps: float = 1e-7):
    y        = batch["y"]
    # log_prob = jax.nn.log_softmax(preds, axis=-1)
    probs = jax.nn.softmax(preds, axis=-1)
    ohe      = jax.nn.one_hot(y, num_classes=N_CLASSES)
    # log_p_t  = jnp.sum(ohe * log_prob, axis=-1)
    # log_p_t  = jnp.maximum(log_p_t, jnp.log(eps))   # floor at -16.1, matches eps version
    # p_t      = jnp.exp(log_p_t)
    p_t = jnp.sum(ohe * probs, axis = -1)
    # return -jnp.mean((1 - p_t) ** gamma * log_p_t)
    return -jnp.mean((1 - p_t) ** gamma * jnp.log(p_t + eps))

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def run_validation_centralized(loss_fn, forward_fn, variables,
                                val_ds, n_val_steps, val_key):
    val_loss = 0.0
    y_label, y_pred = [], []

    for _ in range(n_val_steps):
        val_key, subkey = jax.random.split(val_key)
        *x, y = next(val_ds)
        loss, _ = loss_fn(variables, subkey, x, y)
        pred, _ = forward_fn(variables, subkey, *x)
        val_loss += loss
        y_label.append(y.flatten())
        y_pred.append(pred.argmax(axis=-1).flatten())

    return _compute_metrics(val_loss / n_val_steps, y_label, y_pred), val_key


def run_validation_fl(apply_for_eval, variables, X_val, y_val,
                      n_val_steps, batch_size, seed=0):
    idx = np.random.default_rng(seed).permutation(len(X_val))
    val_loss  = 0.0
    y_label, y_pred = [], []

    for i in range(n_val_steps):
        start   = i * batch_size
        end     = start + batch_size
        batch_idx = idx[start:end]
        x      = jnp.stack([X_val[j] for j in batch_idx])
        y      = jnp.stack([y_val[j] for j in batch_idx])
        batch  = {"x": x, "y": y}
        logits = apply_for_eval(variables, batch)
        val_loss += float(focal_loss_fedjax(batch, logits))
        y_label.append(np.array(y).flatten())
        y_pred.append(np.array(logits.argmax(axis=-1)).flatten())

    return _compute_metrics(val_loss / max(1, len(y_label)), y_label, y_pred)


def _compute_metrics(mean_loss, y_label, y_pred):
    y_true = np.concatenate(y_label)
    y_hat  = np.concatenate(y_pred)
    metrics = {
        "val/loss": float(mean_loss),
        "val/acc":  accuracy_score(y_true, y_hat),
        "val/prec": precision_score(y_true, y_hat, average="macro", zero_division=0),
        "val/rec":  recall_score(y_true, y_hat, average="macro", zero_division=0),
        "val/f1":   f1_score(y_true, y_hat, average="macro", zero_division=0),
    }
    prec_pc, rec_pc, f1_pc, _ = precision_recall_fscore_support(
        y_true, y_hat, labels=list(range(N_CLASSES)), zero_division=0
    )
    for i in range(N_CLASSES):
        metrics[f"val/class_{i}_prec"] = float(prec_pc[i])
        metrics[f"val/class_{i}_rec"]  = float(rec_pc[i])
        metrics[f"val/class_{i}_f1"]   = float(f1_pc[i])
    return metrics


# ---------------------------------------------------------------------------
# FedJAX model wrapper  (FL modes only)
# ---------------------------------------------------------------------------

def build_fedjax_model(cnn: CNN):
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
        train_loss=focal_loss_fedjax,
        eval_metrics={"accuracy": fedjax.metrics.Accuracy()},
    )


# ---------------------------------------------------------------------------
# Training loops
# ---------------------------------------------------------------------------

def train_centralized(cfg, cnn, variables, X_train, y_train, X_val, y_val,
                      ckpt_variables, key, logger: CSVLogger):
    """Original step-based training loop — no FedJAX involved."""
    key, data_key, train_key, val_key = jax.random.split(key, 4)

    optimizer = get_optimizer(
        **cfg.optimizer,
        n_steps=cfg.data.n_rounds,
        grad_accum=cfg.train.grad_accum,
    )
    opt_state = optimizer.init(variables["params"])

    loss_fn    = jax.jit(ft.partial(focal_loss_fn, cnn))
    step_fn    = jax.jit(ft.partial(gradient_step, optimizer, loss_fn))
    forward_fn = jax.jit(ft.partial(forward, cnn))

    input_dtype = jnp.float16 if cfg.float else jnp.uint8
    output_signature = (
        tf.TensorSpec(shape=(X_train[0].shape[0], X_train[0].shape[1]), dtype=input_dtype),
        tf.TensorSpec(shape=(X_train[0].shape[0],), dtype=tf.uint8),
    )
    train_ds = get_dataset(X_train, y_train, train_key,
                           batch_size=cfg.train.batch_size,
                           generator_signature=output_signature)
    val_ds   = get_dataset(X_val,   y_val,   val_key,
                           batch_size=cfg.train.batch_size,
                           generator_signature=output_signature)

    n_steps = int(cfg.data.n_rounds)

    for step in trange(n_steps, desc="[centralized] steps"):
        log_dict   = {}
        train_loss = 0.0
        start      = time.perf_counter()

        for _ in range(cfg.train.grad_accum):
            train_key, subkey = jax.random.split(train_key)
            *x, y = next(train_ds)
            variables, opt_state, loss = step_fn(variables, opt_state, subkey, x, y)
            train_loss += loss

        log_dict["step"]           = step
        log_dict["train/loss"]     = float(train_loss / cfg.train.grad_accum)
        log_dict["perf/step_time"] = time.perf_counter() - start

        if step % cfg.data.val_freq == 0 or step == n_steps - 1:
            val_start = time.perf_counter()
            val_metrics, val_key = run_validation_centralized(
                loss_fn, forward_fn, variables,
                val_ds, cfg.train.n_val_steps, val_key,
            )
            log_dict.update(val_metrics)
            log_dict["perf/val_time"] = time.perf_counter() - val_start

        if step % cfg.data.val_freq == 0 or step == n_steps - 1:
            print(log_dict)
            csv_log(logger, log_dict)
            if cfg.train.logging == "wandb":
                wandb.log(log_dict)

        if step % cfg.train.save_freq == 0 or step == n_steps - 1:
            utils.export.deploy(
                dict(factory=lambda: variables,
                     predict=ft.partial(cnn.apply, training=False)),
                ckpt_variables,
                dict(factory=(), predict=(variables, *x)),
            )

    return variables


def train_fl(cfg, cnn, variables, clients, X_val, y_val,
             ckpt_variables, key, mode, logger: CSVLogger):
    fedjax_model = build_fedjax_model(cnn)
    client_summary(clients)

    if cfg.algorithm.use_server_optimizer:
        server_optimizer = fedjax.optimizers.adam(cfg.algorithm.server_lr)
    else:
        server_optimizer = fedjax.optimizers.sgd(1.0, 0.0)
    client_optimizer = fedjax.optimizers.adam(cfg.algorithm.client_lr,)
    # client_optimizer = fedjax.optimizers.sgd(cfg.algorithm.client_lr, cfg.algorithm.client_momentum)
    client_optimizer = fedjax.optimizers.create_optimizer_from_optax(
        optax.chain(
            optax.clip_by_global_norm(1.0),
            get_optimizer(**cfg.optimizer)
        )
    )

    algorithm = None

    if cfg.algorithm.name == 'fed_avg':

        algorithm = fedjax.algorithms.fed_avg.federated_averaging(
            grad_fn=fedjax.model_grad(fedjax_model),
            client_optimizer=client_optimizer,
            server_optimizer=server_optimizer,
            client_batch_hparams=fedjax.ShuffleRepeatBatchHParams(
                batch_size=cfg.train.batch_size,
                num_epochs=cfg.data.local_epochs,
            ),
        )
    elif cfg.algorithm.name == 'fed_prox':

        def per_example_loss(params, batch, rng):
            logits, _ = forward(cnn, params, rng, batch["x"])
            y         = batch["y"]
            probs     = jax.nn.softmax(logits, axis=-1)
            ohe       = jax.nn.one_hot(y, num_classes=N_CLASSES)
            p_t       = jnp.sum(ohe * probs, axis=-1)
            gamma, eps = 3.0, 1e-7
            return (1 - p_t) ** gamma * -jnp.log(p_t + eps)   # (B, T) — no mean

        algorithm = fedjax.algorithms.fed_prox.fed_prox(
            per_example_loss=per_example_loss,
            client_optimizer=client_optimizer,
            server_optimizer=server_optimizer,
            client_batch_hparams=fedjax.ShuffleRepeatBatchHParams(
                batch_size=cfg.train.batch_size,
                num_epochs=cfg.data.local_epochs,
            ),
            proximal_weight=cfg.algorithm.proximal_weight,   # mu — 0.0 == FedAvg
        )
    elif cfg.algorithm.name == 'scaffold':
        grad_fn = fedjax.model_grad(fedjax_model)
        
        client_lr = getattr(cfg.algorithm, 'client_lr', None)

        algorithm = scaffold(
            grad_fn=grad_fn,
            client_optimizer=client_optimizer,
            server_optimizer=server_optimizer,
            client_batch_hparams=fedjax.ShuffleRepeatBatchHParams(
                batch_size=cfg.train.batch_size,
                num_epochs=cfg.data.local_epochs,
            ),
            client_learning_rate=client_lr
        )
    else:
        raise ValueError(f"Bad algorithm: {cfg.algorithm}")

    input_dtype  = jnp.float32 if cfg.float else jnp.uint8
    sample_input = jnp.zeros(
        (cfg.train.batch_size, cfg.train.seq_len, N_FEATURES), dtype=input_dtype
    )
    server_state    = algorithm.init(variables)
    n_rounds        = int(cfg.data.n_rounds)
    clients_per_rnd = int(cfg.data.clients_per_round)

    # How many batches does one client actually yield per round?
    one_client = sample_round(clients, n_per_round=1, round_num=0, base_seed=cfg.train.seed)[0]
    _, client_ds, _ = one_client
    batches = list(client_ds.shuffle_repeat_batch(
        fedjax.ShuffleRepeatBatchHParams(
            batch_size=cfg.train.batch_size,
            num_epochs=cfg.data.local_epochs,  # or wherever this param lives
        )
    ))

    for round_num in trange(n_rounds, desc=f"[{mode}] rounds"):
        log_dict    = {"round": round_num}
        round_start = time.perf_counter()

        round_clients = sample_round(
            clients,
            n_per_round=clients_per_rnd,
            round_num=round_num,
            base_seed=cfg.train.seed,
        )

        server_state, client_diagnostics = algorithm.apply(server_state, round_clients)
        log_dict["perf/round_time"] = time.perf_counter() - round_start

        if client_diagnostics:
            norms = [float(v["delta_l2_norm"]) for v in client_diagnostics.values()]
            log_dict["train/mean_delta_l2_norm"] = float(np.mean(norms))
            log_dict["train/max_delta_l2_norm"]  = float(np.max(norms))
            log_dict["train/n_clients"]           = len(round_clients)

        if round_num % cfg.data.val_freq == 0 or round_num == n_rounds - 1:
            val_start   = time.perf_counter()
            val_metrics = run_validation_fl(
                fedjax_model.apply_for_eval,
                server_state.params,
                X_val, y_val,
                n_val_steps=cfg.train.n_val_steps,
                batch_size=cfg.train.batch_size,
            )
            log_dict.update(val_metrics)
            log_dict["perf/val_time"] = time.perf_counter() - val_start

        if round_num % cfg.data.val_freq == 0 or round_num == n_rounds - 1:
            print(log_dict)
            csv_log(logger, log_dict)
            if cfg.train.logging == "wandb":
                wandb.log(log_dict)

        if round_num % cfg.train.save_freq == 0 or round_num == n_rounds - 1:
            variables = server_state.params
            utils.export.deploy(
                dict(factory=lambda: variables,
                     predict=ft.partial(cnn.apply, training=False)),
                ckpt_variables,
                dict(factory=(), predict=(variables, sample_input)),
            )

    return server_state.params


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="configs/", config_name="cnn")
def main(cfg: DictConfig) -> None:
    input_dtype = jnp.float16 if cfg.float else jnp.uint8
    mode        = cfg.data.mode

    key      = jax.random.PRNGKey(cfg.train.seed)
    init_key = jax.random.split(key, 1)[0]

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    cnn      = CNN(CNNConfig(**cfg.model))
    init_x   = jnp.empty(
        (cfg.train.batch_size, cfg.train.seq_len, N_FEATURES), dtype=input_dtype
    )
    variables = init(cnn, init_key, init_x, print_summary=True)

    # ------------------------------------------------------------------
    # Dataset loading + client construction
    # ------------------------------------------------------------------
    class_sampling = list(cfg.data.get("class_sampling", [1] * 7))

    if mode == "centralized":
        X_train, y_train, X_val, y_val = make_dataset(
            data_source=cfg.data.dataset_path,
            batch_size=cfg.train.seq_len,
            seed=cfg.train.seed,
            class_sampling=class_sampling,
        )
        normalize_centralized(X_train, X_val)
        clients = None   # not used in centralized path

    elif mode == "fl_iid":
        X_train, y_train, X_val, y_val = make_dataset(
            data_source=cfg.data.dataset_path,
            batch_size=cfg.train.seq_len,
            seed=cfg.train.seed,
            class_sampling=class_sampling,
        )
        clients, X_val, y_val = build_clients(cfg, X_train, y_train, X_val, y_val)

    elif mode == "fl_custom":
        # build_clients loads each client's file and holds out val_fraction
        # from every client, pooling those sequences into X_val / y_val.
        # No dataset_path or val_path needed in the config.
        clients, X_val, y_val = build_clients(cfg)
        X_train = y_train = None   # not used

    else:
        raise ValueError(f"Unknown data.mode '{mode}'")

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

    elif cfg.train.logging == "csv":
        csv_path = os.path.join(ckpt_path, "centralized.csv" if mode == "centralized" else f"{mode}_{cfg.algorithm.name}.csv")

        # ------------------------------------------------------------------
        # Dispatch
        # ------------------------------------------------------------------
        with CSVLogger(csv_path, make_fieldnames(mode)) as logger:
            if mode == "centralized":
                variables = train_centralized(
                    cfg, cnn, variables, X_train, y_train, X_val, y_val,
                    ckpt_variables, key, logger,
                )
            else:
                variables = train_fl(
                    cfg, cnn, variables, clients, X_val, y_val,
                    ckpt_variables, key, mode, logger,
                )

    # ------------------------------------------------------------------
    # Optional NPZ dump  (outside logger context — logger already closed)
    # ------------------------------------------------------------------
    if cfg.save_npz:
        forward_fn = jax.jit(ft.partial(forward, cnn))
        y_label, y_pred_list = [], []
        for x, y in zip(X_val, y_val):
            pred, _ = forward_fn(variables, jax.random.PRNGKey(0),
                                 jnp.expand_dims(x, 0))
            y_label.append(np.array(y))
            y_pred_list.append(np.array(pred))

        np.savez_compressed(
            f"cnn_{mode}.npz",
            labels=np.array(y_label),
            rnn_outputs=np.array(y_pred_list),
        )


if __name__ == "__main__":
    main()
