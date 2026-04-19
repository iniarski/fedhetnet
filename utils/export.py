import sys
import zipfile
from dataclasses import is_dataclass, dataclass
from typing import Callable, Any

import cloudpickle as cp
import jax
from jax import export as jax_export


registry = set()


class ExportableMeta(type):
    def __new__(cls, name, bases, dct):
        assert cls.__module__ != "__main__", "Exportable classes must not be in __main__"
        cls = super().__new__(cls, name, bases, dct)

        if not is_dataclass(cls):
            cls = dataclass(cls)

        cls = jax.tree_util.register_dataclass(cls)
        cls = jax_export.register_pytree_node_serialization(
            cls,
            serialized_name=f"{cls.__module__}.{name}",
            serialize_auxdata=lambda aux: cp.dumps(aux),
            deserialize_auxdata=lambda b: cp.loads(b)
        )
        registry.add(cls)

        return cls


@dataclass
class FlaxPartial:
    """Export friendly version of a partial supporting flax models with arguments"""
    model_t: type
    args: list
    kwargs: dict

    @property
    def model(self):
        return self.model_t(*self.args, **self.kwargs)


def _all_val_types(obj: dict[str, tuple]):
    for v in obj.values():
        for o in v:
            yield type(o)


def deploy(funs: dict[str, Callable], path: str, arguments: dict[str, tuple]):
    version_info = sys.version_info

    with zipfile.ZipFile(
            f'{path}.cp{version_info.major}{version_info.minor}.zip', 'w', zipfile.ZIP_DEFLATED
    ) as z:
        argtypes = {a for a in _all_val_types(arguments)}
        exportables: set[type] = {a for a in argtypes if a in registry}

        for e in exportables:
            with z.open(f'types/{e.__module__}.{e.__name__}', mode='w') as f:
                f.write(cp.dumps(e))

        for name, fun in funs.items():
            exported = jax_export.export(jax.jit(fun), platforms=['tpu', 'cpu', 'cuda', 'rocm'])(*arguments[name])
            serialized = exported.serialize()

            with z.open(name, mode='w') as f:
                f.write(serialized)

def make_polymorphic_batch(tree: Any, constraints: tuple = ()) -> Any:
    """
    Convert a pytree of arrays into a batch of arrays with polymorphic batch shapes.
    `b` is the symbolic batch dimension and `c` is a symbolic helper dimension that can be
    used to express constraints on `b` (e.g. `b == 47 * c`).

    Args:
        tree: A pytree of arrays.
        constraints: A tuple of constraints for the symbolic batch dimension.
            See `jax_export.symbolic_shape` for details.

    Returns:
        A pytree of `ShapeDtypeStruct` with symbolic batch dimension.
    """

    def symbolic_batch_dim(x):
        return jax.ShapeDtypeStruct((b,) + x.shape[1:], x.dtype)

    b, c = jax_export.symbolic_shape('b, c', constraints=constraints)
    return jax.tree.map(symbolic_batch_dim, tree)
