"""Approach C: Additive kernel cost decomposition proxy.

Full model cost = sum over layers of per-kernel costs looked up from a
calibration table produced by KernelHarness.to_cost_table().

Cost is assumed additive (works well for GKR-based provers like Expander;
less accurate for EZKL/Halo2 where power-of-two circuit boundaries create
non-additive jumps).  When a required shape is not in the calibration table
the proxy either falls back to the ONNX estimate (fallback_to_onnx=True) or
raises a KeyError.
"""

from __future__ import annotations

import json
import logging
import math
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from proxy.features import TransformerConfig

if TYPE_CHECKING:
    from proxy.data import ProxyDataset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

# A cost table maps (kernel, seq_len, d_model, n_heads, ff_mult) → cost (int).
# Keys are 5-tuples encoded as JSON-safe lists when serialised.
_CostKey = tuple[Any, ...]
_CostTable = dict[_CostKey, int]

# Kernel names used internally.
_KERNEL_NAMES = ("attn_qk", "attn_av", "softmax", "layernorm", "ffn", "embedding")

# ---------------------------------------------------------------------------
# Interpolation helpers
# ---------------------------------------------------------------------------


def _unique_sorted(values: list[int]) -> list[int]:
    """Return sorted unique values from *values*."""
    return sorted(set(values))


def _nearest(target: int, grid: list[int]) -> int:
    """Return the grid point nearest to *target* (ties broken toward smaller)."""
    return min(grid, key=lambda x: (abs(x - target), x))


def _two_nearest(target: int, grid: list[int]) -> tuple[int, int]:
    """Return the two grid points that bracket *target* for linear interpolation.

    If *target* is outside the grid range, both returned points are the
    boundary values (extrapolation clamps to the edge).
    """
    if target <= grid[0]:
        return grid[0], grid[0]
    if target >= grid[-1]:
        return grid[-1], grid[-1]
    for i in range(len(grid) - 1):
        if grid[i] <= target <= grid[i + 1]:
            return grid[i], grid[i + 1]
    return grid[-1], grid[-1]


def _lerp(lo_val: float, hi_val: float, lo: int, hi: int, target: int) -> float:
    """Linearly interpolate between *lo_val* and *hi_val* at *target*."""
    if lo == hi:
        return lo_val
    t = (target - lo) / (hi - lo)
    return lo_val + t * (hi_val - lo_val)


# ---------------------------------------------------------------------------
# KernelDecompositionProxy
# ---------------------------------------------------------------------------


class KernelDecompositionProxy:  # pylint: disable=too-many-instance-attributes
    """Approach C: additive kernel cost decomposition.

    Full model cost = sum over layers of:
      n_layers * (attn_qk_cost + attn_av_cost + softmax_cost + layernorm_cost * 2)
      + n_layers * ffn_cost
      + embedding_cost

    Each term comes from the kernel calibration table (KernelHarness.to_cost_table).
    Falls back to the ONNX proxy if a shape is not in the table.
    """

    def __init__(
        self,
        cost_table: Optional[dict[Any, int]] = None,
        interpolation: str = "nearest",
        fallback_to_onnx: bool = True,
    ) -> None:
        """Initialise the proxy.

        Parameters
        ----------
        cost_table:
            Dict mapping kernel shape tuples to integer costs.  Keys must be
            ``(kernel, seq_len, d_model, n_heads, ff_mult)`` tuples or the
            JSON-serialised equivalent produced by
            ``KernelHarness.to_cost_table()``.  When ``None`` the table starts
            empty and all lookups will fall back to the ONNX proxy.
        interpolation:
            ``'nearest'`` — snap each requested dimension to the nearest grid
            point before lookup.
            ``'linear'`` — linearly interpolate between the two nearest grid
            points for each dimension independently (seq_len axis only; other
            dimensions still use nearest).
        fallback_to_onnx:
            When ``True`` (default), shapes missing from the table are priced
            via ``zkml_bench.metrics.zk_proxy.compute_proxy``.  When ``False``
            a ``KeyError`` is raised instead.
        """
        if interpolation not in ("nearest", "linear"):
            raise ValueError(
                f"interpolation must be 'nearest' or 'linear'; got {interpolation!r}"
            )

        self._interpolation = interpolation
        self._fallback_to_onnx = fallback_to_onnx
        self._scale: float = 1.0
        self._fitted: bool = False

        # Normalise the cost table: all keys stored as tuples.
        self._cost_table: _CostTable = {}
        if cost_table is not None:
            for raw_key, cost in cost_table.items():
                if isinstance(raw_key, tuple):
                    norm_key: _CostKey = raw_key
                else:
                    # raw_key is Any (from dict[Any, int]); tuple() accepts Any
                    norm_key = tuple(raw_key)
                self._cost_table[norm_key] = int(cost)

        # Pre-compute per-kernel grids for fast nearest/linear lookups.
        self._seq_len_grid: dict[str, list[int]] = {}
        self._d_model_grid: dict[str, list[int]] = {}
        self._n_heads_grid: dict[str, list[int]] = {}
        self._ff_mult_grid: dict[str, list[float]] = {}
        self._rebuild_grids()

    # ------------------------------------------------------------------
    # Grid management
    # ------------------------------------------------------------------

    def _rebuild_grids(self) -> None:
        """Recompute per-kernel shape grids from the current cost table."""
        seq_lens: dict[str, list[int]] = {k: [] for k in _KERNEL_NAMES}
        d_models: dict[str, list[int]] = {k: [] for k in _KERNEL_NAMES}
        n_heads_map: dict[str, list[int]] = {k: [] for k in _KERNEL_NAMES}
        ff_mults: dict[str, list[float]] = {k: [] for k in _KERNEL_NAMES}

        for key in self._cost_table:
            kernel = str(key[0])
            if kernel not in _KERNEL_NAMES:
                continue
            seq_len = int(key[1])
            d_model = int(key[2])
            n_heads = int(key[3]) if len(key) > 3 else 1
            ff_mult = float(key[4]) if len(key) > 4 else 1.0

            seq_lens[kernel].append(seq_len)
            d_models[kernel].append(d_model)
            n_heads_map[kernel].append(n_heads)
            ff_mults[kernel].append(ff_mult)

        for kernel in _KERNEL_NAMES:
            self._seq_len_grid[kernel] = _unique_sorted(seq_lens[kernel])
            self._d_model_grid[kernel] = _unique_sorted(d_models[kernel])
            self._n_heads_grid[kernel] = _unique_sorted(
                [int(x) for x in n_heads_map[kernel]]
            )
            self._ff_mult_grid[kernel] = sorted(set(ff_mults[kernel]))

    # ------------------------------------------------------------------
    # Table lookup helpers
    # ------------------------------------------------------------------

    def _lookup(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
        self,
        kernel: str,
        seq_len: int,
        d_model: int,
        n_heads: int = 1,
        ff_mult: float = 1.0,
    ) -> Optional[float]:
        """Return the cost for *(kernel, seq_len, d_model, n_heads, ff_mult)*.

        Returns ``None`` when the shape is not in the table and
        interpolation cannot cover it.
        """
        seq_grid = self._seq_len_grid.get(kernel, [])
        if not seq_grid:
            return None  # no data for this kernel

        d_grid = self._d_model_grid.get(kernel, [])
        h_grid = self._n_heads_grid.get(kernel, [])
        f_grid = self._ff_mult_grid.get(kernel, [])

        # For all dimensions except seq_len we always use nearest-neighbour.
        eff_d = _nearest(d_model, d_grid) if d_grid else d_model
        eff_h = _nearest(n_heads, h_grid) if h_grid else n_heads
        eff_f = (
            min(f_grid, key=lambda x: (abs(x - ff_mult), x)) if f_grid else ff_mult
        )

        if self._interpolation == "nearest":
            eff_s = _nearest(seq_len, seq_grid)
            key: _CostKey = (kernel, eff_s, eff_d, eff_h, eff_f)
            cost = self._cost_table.get(key)
            if cost is not None:
                return float(cost)
            # Try integer ff_mult fallback (some tables store int keys).
            key_int_f: _CostKey = (kernel, eff_s, eff_d, eff_h, int(eff_f))
            if key_int_f in self._cost_table:
                return float(self._cost_table[key_int_f])
            return None

        # --- linear interpolation along seq_len only ---
        lo_s, hi_s = _two_nearest(seq_len, seq_grid)
        key_lo: _CostKey = (kernel, lo_s, eff_d, eff_h, eff_f)
        key_hi: _CostKey = (kernel, hi_s, eff_d, eff_h, eff_f)

        # Fallback to int ff_mult if needed.
        if key_lo not in self._cost_table:
            key_lo = (kernel, lo_s, eff_d, eff_h, int(eff_f))
        if key_hi not in self._cost_table:
            key_hi = (kernel, hi_s, eff_d, eff_h, int(eff_f))

        lo_cost = self._cost_table.get(key_lo)
        hi_cost = self._cost_table.get(key_hi)

        if lo_cost is None or hi_cost is None:
            return None

        return _lerp(float(lo_cost), float(hi_cost), lo_s, hi_s, seq_len)

    def _onnx_fallback(self, config: TransformerConfig) -> float:
        """Return ONNX proxy cost for *config* as a float.

        Raises ``ImportError`` when ``zkml_bench`` is not available and
        ``fallback_to_onnx`` was requested.
        """
        # pylint: disable=import-outside-toplevel
        import torch
        from torch import nn
        from zkml_bench.metrics.zk_proxy import compute_proxy

        # Build a minimal model accepted by compute_proxy (nn.Module + sample input).
        class _DummyModel(nn.Module):
            """Minimal transformer-like model for ONNX proxy measurement."""

            def __init__(self, cfg: TransformerConfig) -> None:
                super().__init__()
                self.embed: nn.Embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=cfg.d_model,
                    nhead=cfg.n_heads,
                    dim_feedforward=int(cfg.d_model * cfg.ff_mult),
                    activation="relu",
                    batch_first=True,
                )
                self.encoder: nn.TransformerEncoder = nn.TransformerEncoder(
                    encoder_layer, num_layers=cfg.n_layers
                )
                self.head: nn.Linear = nn.Linear(cfg.d_model, cfg.vocab_size)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                """Run the dummy model forward pass."""
                return self.head(self.encoder(self.embed(x)))  # type: ignore[no-any-return]

        model = _DummyModel(config)
        model.eval()
        sample = torch.zeros(1, config.seq_len, dtype=torch.long)
        result = compute_proxy(model, sample)
        return float(result.constraint_estimate)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, dataset: "ProxyDataset") -> None:
        """Optionally calibrate a global scale factor from *dataset*.

        This is a no-op in the sense that no ML model is trained — the cost
        table drives all predictions.  However, if the dataset contains
        labelled records with ground-truth ZK costs, a global multiplicative
        scale factor is derived as::

            scale = mean(ground_truth) / mean(predicted)

        and subsequently applied in :meth:`score`.

        Parameters
        ----------
        dataset:
            A :class:`~proxy.data.ProxyDataset` instance.  Records without a
            ``best_label`` are silently skipped.
        """
        from proxy.data import ProxyDataset  # pylint: disable=import-outside-toplevel

        if not isinstance(dataset, ProxyDataset):
            raise TypeError(
                f"dataset must be a ProxyDataset instance; got {type(dataset)!r}"
            )

        ground_truths: list[float] = []
        predictions: list[float] = []

        for record in dataset._records:  # pylint: disable=protected-access
            label = record.best_label()
            if label is None:
                continue
            try:
                pred = self.score(record.config)
            except Exception:  # pylint: disable=broad-except
                logger.debug(
                    "score() failed for config %s during fit; skipping",
                    record.config,
                    exc_info=True,
                )
                continue
            ground_truths.append(label)
            predictions.append(pred)

        if len(ground_truths) >= 1:
            mean_gt = float(np.mean(ground_truths))
            mean_pred = float(np.mean(predictions))
            if mean_pred != 0.0 and math.isfinite(mean_pred) and math.isfinite(mean_gt):
                self._scale = mean_gt / mean_pred
            else:
                logger.warning(
                    "fit(): mean predicted cost is zero or non-finite; "
                    "scale factor left at 1.0"
                )
        else:
            logger.warning(
                "fit(): no labelled records found in dataset; "
                "scale factor left at 1.0"
            )

        self._fitted = True

    def score(self, config: TransformerConfig) -> float:
        """Predict log(ZK cost) as a sum of kernel costs.

        Parameters
        ----------
        config:
            Transformer architecture to evaluate.

        Returns
        -------
        float
            ``log(total_cost) * scale`` where scale is 1.0 until
            :meth:`fit` is called with a labelled dataset.

        Raises
        ------
        KeyError
            If a required kernel shape is missing from the table and
            ``fallback_to_onnx=False``.
        """
        breakdown = self.decompose(config)
        total = breakdown["total"]
        if total <= 0:
            return float("-inf")
        log_cost = math.log(total)
        return log_cost * self._scale

    def uncertainty(self, config: TransformerConfig) -> float:
        """Return the fraction of kernel lookups that required ONNX fallback.

        Returns
        -------
        float
            0.0 if all kernels were found in the calibration table,
            1.0 if every kernel fell back to the ONNX proxy.
        """
        _, fallback_count, total_count = self._decompose_internal(config)
        if total_count == 0:
            return 1.0
        return fallback_count / total_count

    def decompose(self, config: TransformerConfig) -> dict[str, float]:
        """Return cost breakdown by kernel type.

        Example return value::

            {
              'attn_qk': 1234567,
              'attn_av': 987654,
              'softmax': 234567,
              'layernorm': 456789,
              'ffn': 3456789,
              'embedding': 0,
              'total': 6370377,
              'log_total': 15.67,
              'fallback_fraction': 0.0,
            }

        Parameters
        ----------
        config:
            Transformer architecture to evaluate.

        Returns
        -------
        dict[str, float]
            Kernel-level cost breakdown plus summary statistics.
        """
        costs, fallback_count, total_count = self._decompose_internal(config)

        total = sum(costs.values())
        log_total = math.log(total) if total > 0 else float("-inf")
        fallback_fraction = fallback_count / total_count if total_count > 0 else 1.0

        return {
            "attn_qk": costs["attn_qk"],
            "attn_av": costs["attn_av"],
            "softmax": costs["softmax"],
            "layernorm": costs["layernorm"],
            "ffn": costs["ffn"],
            "embedding": costs["embedding"],
            "total": total,
            "log_total": log_total,
            "fallback_fraction": fallback_fraction,
        }

    def _lookup_with_fallback(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        kernel: str,
        seq_len: int,
        d_model: int,
        n_heads: int,
        ff_mult: float,
        config: TransformerConfig,
        counters: list[int],
    ) -> float:
        """Look up a kernel cost; increment counters[0] (total) and counters[1] (fallbacks).

        Parameters
        ----------
        kernel:
            Kernel name (e.g. ``'attn_qk'``).
        seq_len, d_model, n_heads, ff_mult:
            Shape dimensions for the lookup.
        config:
            Full config used for ONNX fallback if needed.
        counters:
            Mutable two-element list ``[total_count, fallback_count]``.
        """
        counters[0] += 1
        cost = self._lookup(kernel, seq_len, d_model, n_heads, ff_mult)
        if cost is not None:
            return cost

        counters[1] += 1
        if not self._fallback_to_onnx:
            raise KeyError(
                f"kernel={kernel!r} shape=(seq_len={seq_len}, "
                f"d_model={d_model}, n_heads={n_heads}, ff_mult={ff_mult}) "
                "not in cost table and fallback_to_onnx=False"
            )
        logger.debug(
            "Kernel %r shape not in table; using ONNX fallback for config %s",
            kernel,
            config,
        )
        try:
            return self._onnx_fallback(config)
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("ONNX fallback failed: %s; returning 0", exc)
            return 0.0

    def _decompose_internal(
        self, config: TransformerConfig
    ) -> tuple[dict[str, float], int, int]:
        """Compute per-kernel costs.

        Returns a 3-tuple of:

        - per-kernel cost dict (values are already multiplied by n_layers)
        - number of kernels that fell back to ONNX
        - total number of kernel lookups attempted
        """
        # counters[0] = total_count, counters[1] = fallback_count
        counters: list[int] = [0, 0]

        def _get(
            kernel: str,
            d_model: int,
            n_heads: int = 1,
            ff_mult: float = 1.0,
        ) -> float:
            """Short-form wrapper — seq_len always comes from config."""
            return self._lookup_with_fallback(
                kernel, config.seq_len, d_model, n_heads, ff_mult, config, counters
            )

        # Per-layer costs.
        # attn_qk / attn_av: operate on (seq, d_model), split by n_heads.
        attn_qk = _get("attn_qk", config.d_model, config.n_heads)
        attn_av = _get("attn_av", config.d_model, config.n_heads)
        # softmax: operates on (batch, heads, seq, seq) → d_model=1
        softmax = _get("softmax", d_model=1, n_heads=config.n_heads)
        # layernorm: appears twice per block
        layernorm_single = _get("layernorm", config.d_model)
        ffn = _get("ffn", config.d_model, ff_mult=config.ff_mult)
        # embedding: depends on vocab_size → usually falls back to ONNX
        embedding = _get("embedding", config.d_model)

        costs: dict[str, float] = {
            "attn_qk": attn_qk * config.n_layers,
            "attn_av": attn_av * config.n_layers,
            "softmax": softmax * config.n_layers,
            "layernorm": layernorm_single * 2 * config.n_layers,
            "ffn": ffn * config.n_layers,
            "embedding": embedding,
        }

        return costs, counters[1], counters[0]

    # ------------------------------------------------------------------
    # Factory / persistence
    # ------------------------------------------------------------------

    @classmethod
    def from_kernel_harness_path(cls, path: str) -> "KernelDecompositionProxy":
        """Load kernel benchmarks from a JSON file and build a cost table.

        The file must contain a JSON object in the format produced by
        ``KernelHarness.to_cost_table()``.  Keys are serialised as
        ``"kernel|seq_len|d_model|n_heads|ff_mult"`` pipe-delimited strings
        or as JSON arrays; both formats are supported.

        Parameters
        ----------
        path:
            Absolute path to the JSON file.

        Returns
        -------
        KernelDecompositionProxy
            Initialised proxy with the loaded cost table.
        """
        with open(path, encoding="utf-8") as fh:
            raw: dict[str, Any] = json.load(fh)

        cost_table: _CostTable = {}
        for raw_key, cost in raw.items():
            # Support both list keys (JSON arrays) and pipe-separated strings.
            parsed_key: _CostKey
            if isinstance(raw_key, str) and "|" in raw_key:
                parts = raw_key.split("|")
                kernel = parts[0]
                seq_len = int(parts[1])
                d_model = int(parts[2])
                n_heads = int(parts[3]) if len(parts) > 3 else 1
                ff_mult = float(parts[4]) if len(parts) > 4 else 1.0
                parsed_key = (kernel, seq_len, d_model, n_heads, ff_mult)
            elif isinstance(raw_key, str):
                # Try JSON-decode as list.
                try:
                    parts_list = json.loads(raw_key)
                    parsed_key = tuple(parts_list)
                except json.JSONDecodeError:
                    logger.warning("Skipping unrecognised cost table key: %r", raw_key)
                    continue
            else:
                parsed_key = tuple(raw_key)

            cost_table[parsed_key] = int(cost)

        return cls(cost_table=cost_table)

    def is_fitted(self) -> bool:
        """Return ``True`` if :meth:`fit` has been called or a non-empty cost table was loaded."""
        return self._fitted or bool(self._cost_table)

    def save(self, path: str) -> None:
        """Serialise the proxy to a JSON file at *path*.

        The saved file stores the cost table, interpolation mode, fallback
        flag, and the fitted scale factor so the proxy can be restored
        exactly with :meth:`load`.

        Parameters
        ----------
        path:
            Absolute path to write to (will be created or overwritten).
        """
        # Serialise cost table keys as pipe-separated strings for JSON
        # compatibility (tuples are not valid JSON keys).
        serialised_table: dict[str, int] = {}
        for key, cost in self._cost_table.items():
            str_key = "|".join(str(k) for k in key)
            serialised_table[str_key] = cost

        payload = {
            "version": 1,
            "interpolation": self._interpolation,
            "fallback_to_onnx": self._fallback_to_onnx,
            "scale": self._scale,
            "fitted": self._fitted,
            "cost_table": serialised_table,
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

    @classmethod
    def load(cls, path: str) -> "KernelDecompositionProxy":
        """Load a proxy that was previously saved with :meth:`save`.

        Parameters
        ----------
        path:
            Absolute path to the JSON file.

        Returns
        -------
        KernelDecompositionProxy
            Fully restored proxy, including scale factor.
        """
        with open(path, encoding="utf-8") as fh:
            payload: dict[str, Any] = json.load(fh)

        # Restore cost table.
        raw_table: dict[str, int] = payload.get("cost_table", {})
        cost_table: _CostTable = {}
        for str_key, cost in raw_table.items():
            parts = str_key.split("|")
            kernel = parts[0]
            seq_len = int(parts[1])
            d_model = int(parts[2])
            n_heads = int(parts[3]) if len(parts) > 3 else 1
            ff_mult = float(parts[4]) if len(parts) > 4 else 1.0
            cost_table[(kernel, seq_len, d_model, n_heads, ff_mult)] = int(cost)

        proxy = cls(
            cost_table=cost_table,
            interpolation=payload.get("interpolation", "nearest"),
            fallback_to_onnx=bool(payload.get("fallback_to_onnx", True)),
        )
        proxy._scale = float(payload.get("scale", 1.0))
        proxy._fitted = bool(payload.get("fitted", False))
        return proxy
