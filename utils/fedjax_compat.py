# FedJAX compatibility shim: restore removed jax.tree_* aliases
import jax, jax.tree_util as _tu
_missing = {
    "tree_map":      _tu.tree_map,
    "tree_leaves":   _tu.tree_leaves,
    "tree_flatten":  _tu.tree_flatten,
    "tree_unflatten":_tu.tree_unflatten,
    "tree_multimap": _tu.tree_map,   # was an alias before removal
}
for _k, _v in _missing.items():
    if not hasattr(jax, _k):
        setattr(jax, _k, _v)
