import jax
import optax
import tensorflow as tf
import numpy as np
import functools as ft


from numpy.typing import ArrayLike
from types import FunctionType
N_FEATURES = 33

def get_optimizer(
        optimizer, lr, b1=0.9, b2=0.95, eps=1e-8, weight_decay=0.0, warmup_pct=0.1,
        n_steps=int(1e5), div_factor=25, final_div_factor=1e4, lr_schedule="constant", grad_accum=1, momentum=0.9
):
    if lr_schedule == "cosine":
        lr = optax.cosine_onecycle_schedule(n_steps, lr, warmup_pct, div_factor, final_div_factor)
    elif lr_schedule == "constant":
        lr = optax.constant_schedule(lr)

    if optimizer == "adam":
        optimizer = optax.adam(lr, b1, b2, eps)
    elif optimizer == "adamw":
        optimizer = optax.adamw(lr, b1, b2, eps, weight_decay=weight_decay)
    elif optimizer == "sgd":
        optimizer = optax.sgd(lr, momentum=momentum)

    if grad_accum > 1:
        optimizer = optax.MultiSteps(optimizer, grad_accum)

    return optimizer


def gradient_step(optimizer, loss_fn, variables, opt_state, *loss_params):
    params = variables.pop('params')
    state = variables

    (loss, state), grads = jax.value_and_grad(lambda p: loss_fn({'params': p, **state}, *loss_params), has_aux=True)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    return {'params': params, **state}, opt_state, loss


def init(model, key, *x, print_summary=False):
    params_key, gpt_key, dropout_key = jax.random.split(key, 3)
    variables = model.init({'params': params_key, 'gpt': gpt_key, 'dropout': dropout_key}, *x)

    if print_summary:
        print(model.tabulate(jax.random.key(0), *x, compute_flops=True))

    return variables


def init_cache(model, *x):
    variables = model.init({'params': jax.random.PRNGKey(0)}, *x, training=False)
    return variables['cache']


def forward(model, variables, key, *x, method=None):
    return model.apply(variables, *x, rngs=key, mutable=list(set(variables) - {'params'}), method=method)


def grad_norm(loss_fn, variables, key, x, y):
    params = variables['params']
    state = {k: v for k, v in variables.items() if k != 'params'}
    (loss, _), grads = jax.value_and_grad(loss_fn, has_aux=True)({'params': params, **state}, key, x, y)
    return optax.tree_utils.tree_l2_norm(grads), loss

def default_sampler(tokens : ArrayLike, labels : ArrayLike, key : jax.random.PRNGKey):
    while True:
        key, subkey = jax.random.split(key)
        idx = jax.random.randint(subkey, (), minval=0, maxval=len(tokens))
        yield tokens[idx], labels[idx]

def offset_sampler(tokens : ArrayLike, labels : ArrayLike, key : jax.random.PRNGKey, target_len : int):
    while True:
        key, subkey = jax.random.split(key)
        idx = jax.random.randint(subkey, (), minval=0, maxval=len(tokens))
        sample_len = tokens[idx].shape[0]
        offset = jax.random.randint(subkey, (), minval=0, maxval=sample_len - target_len)
        yield tokens[idx][offset:target_len+offset], labels[idx][offset:target_len+offset]

def get_dataset(
    tokens : ArrayLike,
    labels : ArrayLike,
    key : jax.random.PRNGKey,
    batch_size : int | None = None,
    generator_signature : tuple[tf.TensorSpec] = (tf.TensorSpec(shape=(None, N_FEATURES), dtype=tf.uint8), tf.TensorSpec(shape=(None,), dtype=tf.uint8)),
    sampler : FunctionType = default_sampler
 ) :
    ds = tf.data.Dataset.from_generator(
        ft.partial(sampler, tokens, labels, key),
        output_signature=generator_signature
    )
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    if batch_size:
        ds = ds.batch(batch_size)
    return ds.as_numpy_iterator()
