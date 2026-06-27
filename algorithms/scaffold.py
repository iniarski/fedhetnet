import jax
import jax.numpy as jnp
from typing import Any, Callable, Mapping, Sequence, Tuple, Dict

from fedjax.core import client_datasets
from fedjax.core import dataclasses
from fedjax.core import federated_algorithm
from fedjax.core import federated_data
from fedjax.core import for_each_client
from fedjax.core import optimizers
from fedjax.core import tree_util
from fedjax.core.typing import BatchExample, Params, PRNGKey

Grads = Params

@dataclasses.dataclass
class ServerState:
    """State of server passed between rounds.
    
    Attributes:
      params: Global model weights.
      opt_state: Server optimizer state.
      server_cv: Global control variate.
      client_cvs: A dictionary mapping client_ids to their local control variates.
    """
    params: Params
    opt_state: optimizers.OptState
    server_cv: Params
    client_cvs: Dict[federated_data.ClientId, Params]


def create_train_for_each_client(grad_fn, client_optimizer, client_learning_rate):
    """Builds client_init, client_step, client_final for for_each_client."""

    def client_init(shared_input, client_input):
        # shared_input comes from the server; client_input comes from the batch_clients tuple
        server_params, server_cv = shared_input
        client_rng, client_cv = client_input
        
        opt_state = client_optimizer.init(server_params)
        client_step_state = {
            'params': server_params,
            'opt_state': opt_state,
            'rng': client_rng,
            'server_params': server_params,
            'server_cv': server_cv,
            'client_cv': client_cv,
            'num_steps': jnp.array(0, dtype=jnp.int32) # Track K for the CV update
        }
        return client_step_state

    def client_step(client_step_state, batch):
        rng, use_rng = jax.random.split(client_step_state['rng'])
        
        # 1. Standard gradient
        grads = grad_fn(client_step_state['params'], batch, use_rng)
        
        # 2. SCAFFOLD Correction: g = g - c_i + c
        cv_diff = jax.tree_util.tree_map(lambda sc, cc: sc - cc, 
                                         client_step_state['server_cv'], 
                                         client_step_state['client_cv'])
        corrected_grads = jax.tree_util.tree_map(lambda g, diff: g + diff, grads, cv_diff)
        
        # 3. Apply optimizer step
        opt_state, params = client_optimizer.apply(corrected_grads,
                                                   client_step_state['opt_state'],
                                                   client_step_state['params'])
        return {
            'params': params,
            'opt_state': opt_state,
            'rng': rng,
            'server_params': client_step_state['server_params'],
            'server_cv': client_step_state['server_cv'],
            'client_cv': client_step_state['client_cv'],
            'num_steps': client_step_state['num_steps'] + 1
        }

    def client_final(shared_input, client_step_state):
        server_params, server_cv = shared_input
        
        # delta_params = x - y
        delta_params = jax.tree_util.tree_map(lambda a, b: a - b,
                                              server_params,
                                              client_step_state['params'])
        
        # delta_c_i = (1 / (K * eta)) * delta_params - server_cv
        K = client_step_state['num_steps']
        eta = client_learning_rate
        
        def calc_delta_cv(dp, scv):
            return dp / (K * eta) - scv
            
        delta_cv = jax.tree_util.tree_map(calc_delta_cv, delta_params, server_cv)
        
        return delta_params, delta_cv

    return for_each_client.for_each_client(client_init, client_step, client_final)


def scaffold(
    grad_fn: Callable[[Params, BatchExample, PRNGKey], Grads],
    client_optimizer: optimizers.Optimizer,
    server_optimizer: optimizers.Optimizer,
    client_batch_hparams: client_datasets.ShuffleRepeatBatchHParams,
    client_learning_rate: float
) -> federated_algorithm.FederatedAlgorithm:
    """Builds SCAFFOLD algorithm."""

    train_for_each_client = create_train_for_each_client(grad_fn, client_optimizer, client_learning_rate)

    def init(params: Params) -> ServerState:
        opt_state = server_optimizer.init(params)
        server_cv = tree_util.tree_zeros_like(params)
        client_cvs = {} # Dictionary to store stateful client CVs
        return ServerState(params, opt_state, server_cv, client_cvs)

    def apply(
        server_state: ServerState,
        clients: Sequence[Tuple[federated_data.ClientId, client_datasets.ClientDataset, PRNGKey]]
    ) -> Tuple[ServerState, Mapping[federated_data.ClientId, Any]]:
        
        client_num_examples = {cid: len(cds) for cid, cds, _ in clients}
        
        # Inject the client's local control variate into the client input tuple
        batch_clients = []
        for cid, cds, crng in clients:
            client_cv = server_state.client_cvs.get(cid, tree_util.tree_zeros_like(server_state.params))
            batch_clients.append((cid, cds.shuffle_repeat_batch(client_batch_hparams), (crng, client_cv)))
            
        client_diagnostics = {}
        delta_params_sum = tree_util.tree_zeros_like(server_state.params)
        delta_cv_sum = tree_util.tree_zeros_like(server_state.server_cv)
        num_examples_sum = 0.
        
        shared_input = (server_state.params, server_state.server_cv)
        new_client_cvs = dict(server_state.client_cvs)
        
        # Iterative accumulation to save memory
        for client_id, (delta_params, delta_cv) in train_for_each_client(shared_input, batch_clients):
            num_examples = client_num_examples[client_id]
            
            delta_params_sum = tree_util.tree_add(
                delta_params_sum, tree_util.tree_weight(delta_params, num_examples))
            delta_cv_sum = tree_util.tree_add(
                delta_cv_sum, tree_util.tree_weight(delta_cv, num_examples))
            num_examples_sum += num_examples
            
            # Update local control variate in the dictionary state: c_i = c_i + delta_cv
            old_client_cv = server_state.client_cvs.get(client_id, tree_util.tree_zeros_like(server_state.params))
            new_client_cvs[client_id] = tree_util.tree_add(old_client_cv, delta_cv)
            
            client_diagnostics[client_id] = {
                'delta_l2_norm': tree_util.tree_l2_norm(delta_params),
                'delta_cv_l2_norm': tree_util.tree_l2_norm(delta_cv)
            }
            
        mean_delta_params = tree_util.tree_inverse_weight(delta_params_sum, num_examples_sum)
        mean_delta_cv = tree_util.tree_inverse_weight(delta_cv_sum, num_examples_sum)
        
        # Update server weights
        opt_state, params = server_optimizer.apply(mean_delta_params,
                                                   server_state.opt_state,
                                                   server_state.params)
        # Update server control variate
        server_cv = tree_util.tree_add(server_state.server_cv, mean_delta_cv)
        
        next_server_state = ServerState(params, opt_state, server_cv, new_client_cvs)
        
        return next_server_state, client_diagnostics

    return federated_algorithm.FederatedAlgorithm(init, apply)