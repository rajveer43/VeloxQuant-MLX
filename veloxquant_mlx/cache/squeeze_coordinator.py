"""SqueezeAttention cross-layer coordinator — one-shot data-driven re-budgeting.

The repo's contract is one cache object per layer; ``mlx_lm.generate`` iterates
them independently. SqueezeAttention needs a *global* view to reallocate a fixed
total budget across layers by their measured attention concentration. Rather than
modify the model forward pass, all ``SqueezeAttentionCache`` instances of a model
hold a reference to a single shared ``SqueezeCoordinator`` (injected at
``KVCacheBuilder.for_model`` build time).

The re-budget is **one-shot at the prefill boundary**: during the first
``update_and_fetch`` (the prompt), each layer reports its concentration score.
Once every attention layer has reported, the coordinator computes the per-layer
budget schedule with ``squeeze_budgets`` and publishes it; each layer then pulls
its resolved budget and applies it to every head's eviction state. Decode steps
run against the frozen schedule — no further re-budgeting.

Unlike the XQuant / MiniCache coordinators (which exchange *tensors* every step),
this coordinator exchanges only per-layer scalars and runs its allocation exactly
once. Single-threaded by construction (mlx generate is sequential), so plain
dicts need no locking.
"""

from __future__ import annotations

from typing import Optional

from veloxquant_mlx.quantizers.squeeze import squeeze_budgets


class SqueezeCoordinator:
    """Shared per-model re-budgeting state for one generation.

    Args:
        n_layers:   Number of attention-bearing layers that will report.
        avg_budget: Target mean per-layer budget (the uniform baseline).
        n_sink:     Sink tokens each layer protects (min-budget floor).
        strength:   Reallocation strength in [0, 1]; 0.0 == uniform H2O.
    """

    def __init__(
        self,
        n_layers: int,
        avg_budget: int,
        n_sink: int,
        strength: float,
    ) -> None:
        self._n_layers = int(n_layers)
        self._avg_budget = int(avg_budget)
        self._n_sink = int(n_sink)
        self._strength = float(strength)

        # layer_id -> reported concentration score (None until reported)
        self._concentrations: dict[int, float] = {}
        # layer_id -> resolved budget (populated once all layers have reported)
        self._resolved: dict[int, int] = {}
        self._layer_order: list[int] = []
        self._done = False

    def reset(self) -> None:
        """Clear all reported/resolved state (e.g. between generations)."""
        self._concentrations.clear()
        self._resolved.clear()
        self._layer_order.clear()
        self._done = False

    def report_concentration(self, layer_id: int, concentration: float) -> None:
        """Record a layer's prefill concentration score.

        Idempotent per layer: the *first* report for a layer wins (prefill only);
        later reports (decode steps) are ignored so the schedule stays frozen.

        Once all ``n_layers`` have reported, the budget schedule is computed and
        cached so subsequent ``resolved_budget`` calls return real values.

        Args:
            layer_id:      The reporting layer's index.
            concentration: Its ``concentration_score`` over the prefill keys.
        """
        if self._done or layer_id in self._concentrations:
            return
        self._concentrations[layer_id] = float(concentration)
        self._layer_order.append(layer_id)

        if len(self._concentrations) >= self._n_layers:
            self._finalize()

    def _finalize(self) -> None:
        """Compute the per-layer budget schedule once every layer has reported."""
        order = sorted(self._concentrations.keys())
        conc_vec = [self._concentrations[lid] for lid in order]
        schedule = squeeze_budgets(
            conc_vec,
            avg_budget=self._avg_budget,
            n_sink=self._n_sink,
            strength=self._strength,
        )
        self._resolved = {lid: schedule[k] for k, lid in enumerate(order)}
        self._done = True

    def resolved_budget(self, layer_id: int) -> Optional[int]:
        """Return the layer's resolved budget, or ``None`` if not yet finalised.

        Args:
            layer_id: The querying layer's index.

        Returns:
            The resolved budget once all layers have reported; ``None`` before
            that (the caller keeps using its average fallback until then).
        """
        if not self._done:
            return None
        return self._resolved.get(layer_id)

    @property
    def is_finalized(self) -> bool:
        """True once every layer has reported and the schedule is computed."""
        return self._done

    @property
    def n_layers(self) -> int:
        return self._n_layers

    @property
    def avg_budget(self) -> int:
        return self._avg_budget

    @property
    def strength(self) -> float:
        return self._strength


__all__ = ["SqueezeCoordinator"]
