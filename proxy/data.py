"""Dataset accumulating (architecture config → ZK cost) training examples.

Records are collected across multiple fidelity levels:

* ONNX_ESTIMATE  — free; computed analytically from the ONNX graph via
  ``zkml_bench``'s ``zk_proxy`` metric.
* EZKL_CALIBRATION — takes seconds; runs the EZKL settings + compile step
  (no full proof) and returns circuit parameters.
* EZKL_PROVE — takes minutes; runs a full EZKL proof and records the
  elapsed wall-clock time and circuit row count.

The dataset persists to JSON so it survives across search runs.
"""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

import numpy as np

from proxy.features import ProxyFeatures, TransformerConfig, feature_names


# ---------------------------------------------------------------------------
# Fidelity enum
# ---------------------------------------------------------------------------


class Fidelity(Enum):
    """Ordered fidelity levels (lower index = cheaper / less accurate)."""

    ONNX_ESTIMATE = "onnx_estimate"
    EZKL_CALIBRATION = "ezkl_calibration"
    EZKL_PROVE = "ezkl_prove"

    def __lt__(self, other: "Fidelity") -> bool:
        return _FIDELITY_ORDER.index(self.value) < _FIDELITY_ORDER.index(other.value)

    def __le__(self, other: "Fidelity") -> bool:
        return _FIDELITY_ORDER.index(self.value) <= _FIDELITY_ORDER.index(other.value)

    def __gt__(self, other: "Fidelity") -> bool:
        return not self.__le__(other)

    def __ge__(self, other: "Fidelity") -> bool:
        return not self.__lt__(other)


_FIDELITY_ORDER: list[str] = [
    Fidelity.ONNX_ESTIMATE.value,
    Fidelity.EZKL_CALIBRATION.value,
    Fidelity.EZKL_PROVE.value,
]


# ---------------------------------------------------------------------------
# ProxyRecord
# ---------------------------------------------------------------------------


@dataclass
class ProxyRecord:  # pylint: disable=too-many-instance-attributes
    """A single (config, features, labels) observation.

    Labels are at different fidelities; ``None`` means *not yet measured*.
    """

    config: TransformerConfig
    features: ProxyFeatures

    # --- Labels at different fidelities (None = not measured yet) ---
    onnx_estimate: Optional[int] = None
    """``constraint_estimate`` from ``zk_proxy`` (ONNX analytic estimate)."""

    ezkl_calibration: Optional[dict] = None
    """Dict with keys ``k``, ``num_rows``, ``num_cols``, … from EZKL calibration."""

    ezkl_prove_time: Optional[float] = None
    """Wall-clock seconds for a full EZKL proof run."""

    ezkl_num_rows: Optional[int] = None
    """EZKL circuit row count (strongest proxy for ZK cost)."""

    # --- Metadata ---
    record_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    backend: str = "ezkl"
    notes: str = ""

    # ------------------------------------------------------------------
    # Label helpers
    # ------------------------------------------------------------------

    def best_label(self) -> Optional[float]:
        """Return highest-fidelity available label as log(cost).

        Priority: ``ezkl_num_rows`` > ``ezkl_prove_time`` > ``onnx_estimate``.
        Returns ``None`` if no label is available.

        The three quantities are converted to a common log-scale:

        * ``ezkl_num_rows`` — circuit size; used directly: ``log(ezkl_num_rows)``.
        * ``ezkl_prove_time`` — seconds; converted to ms before log:
          ``log(ezkl_prove_time * 1000)``.
        * ``onnx_estimate`` — raw constraint count: ``log(onnx_estimate)``.
        """
        if self.ezkl_num_rows is not None and self.ezkl_num_rows > 0:
            return math.log(self.ezkl_num_rows)
        if self.ezkl_prove_time is not None and self.ezkl_prove_time > 0:
            return math.log(self.ezkl_prove_time * 1000.0)
        if self.onnx_estimate is not None and self.onnx_estimate > 0:
            return math.log(self.onnx_estimate)
        return None

    def fidelity_level(self) -> Optional[Fidelity]:
        """Return the :class:`Fidelity` of the best available label.

        Returns ``None`` if no label is available.
        """
        if self.ezkl_num_rows is not None and self.ezkl_num_rows > 0:
            return Fidelity.EZKL_PROVE
        if self.ezkl_prove_time is not None and self.ezkl_prove_time > 0:
            return Fidelity.EZKL_PROVE
        if self.ezkl_calibration is not None:
            return Fidelity.EZKL_CALIBRATION
        if self.onnx_estimate is not None and self.onnx_estimate > 0:
            return Fidelity.ONNX_ESTIMATE
        return None


# ---------------------------------------------------------------------------
# JSON serialization helpers
# ---------------------------------------------------------------------------


def _sanitize_for_json(obj: Any) -> Any:
    """Recursively convert numpy scalars / arrays to Python-native types."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return [_sanitize_for_json(v) for v in obj.tolist()]
    return obj


def _config_to_dict(config: TransformerConfig) -> dict:
    return {
        "d_model": config.d_model,
        "n_heads": config.n_heads,
        "n_layers": config.n_layers,
        "seq_len": config.seq_len,
        "ff_mult": config.ff_mult,
        "activation": config.activation,
        "use_rotary": config.use_rotary,
        "use_layernorm": config.use_layernorm,
        "vocab_size": config.vocab_size,
        "tie_embeddings": config.tie_embeddings,
    }


def _config_from_dict(d: dict) -> TransformerConfig:
    return TransformerConfig(
        d_model=d["d_model"],
        n_heads=d["n_heads"],
        n_layers=d["n_layers"],
        seq_len=d["seq_len"],
        ff_mult=float(d["ff_mult"]),
        activation=d.get("activation", "gelu"),
        use_rotary=bool(d.get("use_rotary", False)),
        use_layernorm=bool(d.get("use_layernorm", True)),
        vocab_size=int(d.get("vocab_size", 50257)),
        tie_embeddings=bool(d.get("tie_embeddings", False)),
    )


def _features_to_dict(features: ProxyFeatures) -> dict:
    return {
        "d_model": features.d_model,
        "n_heads": features.n_heads,
        "n_layers": features.n_layers,
        "seq_len": features.seq_len,
        "ff_mult": features.ff_mult,
        "head_dim": features.head_dim,
        "total_params": features.total_params,
        "decoder_params": features.decoder_params,
        "attn_macs": features.attn_macs,
        "ffn_macs": features.ffn_macs,
        "total_macs": features.total_macs,
        "activation_id": features.activation_id,
        "is_zk_friendly": features.is_zk_friendly,
        "layernorm_count": features.layernorm_count,
        "softmax_count": features.softmax_count,
        "gelu_count": features.gelu_count,
        "onnx_constraint_estimate": features.onnx_constraint_estimate,
        "onnx_nonlinear_ops": features.onnx_nonlinear_ops,
        "onnx_linear_ops": features.onnx_linear_ops,
    }


def _features_from_dict(d: dict) -> ProxyFeatures:
    return ProxyFeatures(
        d_model=int(d["d_model"]),
        n_heads=int(d["n_heads"]),
        n_layers=int(d["n_layers"]),
        seq_len=int(d["seq_len"]),
        ff_mult=float(d["ff_mult"]),
        head_dim=float(d["head_dim"]),
        total_params=int(d["total_params"]),
        decoder_params=int(d.get("decoder_params", d["total_params"])),
        attn_macs=int(d["attn_macs"]),
        ffn_macs=int(d["ffn_macs"]),
        total_macs=int(d["total_macs"]),
        activation_id=int(d["activation_id"]),
        is_zk_friendly=bool(d["is_zk_friendly"]),
        layernorm_count=int(d["layernorm_count"]),
        softmax_count=int(d["softmax_count"]),
        gelu_count=int(d["gelu_count"]),
        onnx_constraint_estimate=d.get("onnx_constraint_estimate"),
        onnx_nonlinear_ops=d.get("onnx_nonlinear_ops"),
        onnx_linear_ops=d.get("onnx_linear_ops"),
    )


def _record_to_dict(record: ProxyRecord) -> dict[str, Any]:
    """Serialize a :class:`ProxyRecord` to a JSON-compatible dict."""
    return _sanitize_for_json(
        {
            "config": _config_to_dict(record.config),
            "features": _features_to_dict(record.features),
            "onnx_estimate": record.onnx_estimate,
            "ezkl_calibration": record.ezkl_calibration,
            "ezkl_prove_time": record.ezkl_prove_time,
            "ezkl_num_rows": record.ezkl_num_rows,
            "record_id": record.record_id,
            "created_at": record.created_at,
            "backend": record.backend,
            "notes": record.notes,
        }
    )


def _record_from_dict(d: dict) -> ProxyRecord:
    """Deserialize a :class:`ProxyRecord` from a JSON-compatible dict."""
    config = _config_from_dict(d["config"])
    features = _features_from_dict(d["features"])
    return ProxyRecord(
        config=config,
        features=features,
        onnx_estimate=d.get("onnx_estimate"),
        ezkl_calibration=d.get("ezkl_calibration"),
        ezkl_prove_time=d.get("ezkl_prove_time"),
        ezkl_num_rows=d.get("ezkl_num_rows"),
        record_id=d.get("record_id", str(uuid.uuid4())),
        created_at=d.get("created_at", datetime.now(timezone.utc).isoformat()),
        backend=d.get("backend", "ezkl"),
        notes=d.get("notes", ""),
    )


# ---------------------------------------------------------------------------
# ProxyDataset
# ---------------------------------------------------------------------------


class ProxyDataset:
    """Persistent, appendable dataset of (config, features, labels) records.

    Serialises to JSON for portability and grows across runs.

    Parameters
    ----------
    path:
        Optional filesystem path to an existing JSON dataset.  If supplied
        and the file exists, the dataset is loaded from it.  If the file
        does not exist, the dataset starts empty and will be saved to this
        path on the first :meth:`save` call.
    """

    def __init__(self, path: Optional[str] = None) -> None:
        self._records: list[ProxyRecord] = []
        # Map config_hash → index in _records for O(1) deduplication.
        self._hash_index: dict[str, int] = {}
        self._path: Optional[str] = path

        if path is not None:
            try:
                loaded = ProxyDataset.load(path)
                self._records = loaded._records
                self._hash_index = loaded._hash_index
            except FileNotFoundError:
                pass  # start empty; file will be created on first save

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(self, record: ProxyRecord) -> None:
        """Append a record, deduplicating by config hash.

        If a record with the same config hash already exists, the existing
        record's labels are updated with any non-``None`` values from
        *record* (higher-fidelity labels win).
        """
        h = ProxyDataset.config_hash(record.config)
        if h in self._hash_index:
            existing = self._records[self._hash_index[h]]
            # Merge labels: non-None values update the existing record.
            if record.ezkl_num_rows is not None:
                existing.ezkl_num_rows = record.ezkl_num_rows
            if record.ezkl_prove_time is not None:
                existing.ezkl_prove_time = record.ezkl_prove_time
            if record.ezkl_calibration is not None:
                existing.ezkl_calibration = record.ezkl_calibration
            if record.onnx_estimate is not None:
                existing.onnx_estimate = record.onnx_estimate
        else:
            idx = len(self._records)
            self._records.append(record)
            self._hash_index[h] = idx

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save dataset to a JSON file at *path*."""
        payload = {
            "version": 1,
            "records": [_record_to_dict(r) for r in self._records],
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

    @classmethod
    def load(cls, path: str) -> "ProxyDataset":
        """Load a :class:`ProxyDataset` from a JSON file.

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        """
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)

        dataset = cls.__new__(cls)
        dataset._records = []
        dataset._hash_index = {}
        dataset._path = path

        for record_dict in payload.get("records", []):
            record = _record_from_dict(record_dict)
            h = ProxyDataset.config_hash(record.config)
            if h not in dataset._hash_index:
                idx = len(dataset._records)
                dataset._records.append(record)
                dataset._hash_index[h] = idx

        return dataset

    # ------------------------------------------------------------------
    # ML helpers
    # ------------------------------------------------------------------

    def to_xy(
        self,
        min_fidelity: Fidelity = Fidelity.ONNX_ESTIMATE,
        backend: Optional[str] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(X, y)`` arrays for fitting a proxy model.

        Parameters
        ----------
        min_fidelity:
            Only include records whose :meth:`ProxyRecord.fidelity_level` is
            at least *min_fidelity*.
        backend:
            When given, restrict to records with :attr:`ProxyRecord.backend`
            equal to this string.

        Returns
        -------
        X : np.ndarray, shape (n, n_features)
            Feature matrix — each row is ``ProxyFeatures.to_vector()``.
        y : np.ndarray, shape (n,)
            Log-cost labels.
        """
        n_features = len(feature_names())
        xs: list[np.ndarray] = []
        ys: list[float] = []

        for record in self._records:
            label = record.best_label()
            if label is None:
                continue
            fid = record.fidelity_level()
            if fid is None:
                continue
            if fid < min_fidelity:
                continue
            if backend is not None and record.backend != backend:
                continue
            xs.append(record.features.to_vector())
            ys.append(label)

        if not xs:
            return np.empty((0, n_features), dtype=np.float64), np.empty(
                (0,), dtype=np.float64
            )

        return np.vstack(xs), np.array(ys, dtype=np.float64)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def configs_proved(
        self, fidelity: Fidelity = Fidelity.EZKL_PROVE
    ) -> list[TransformerConfig]:
        """Return configs that have been measured at *fidelity* or above.

        Parameters
        ----------
        fidelity:
            Minimum fidelity level required.
        """
        out: list[TransformerConfig] = []
        for record in self._records:
            fid = record.fidelity_level()
            if fid is None:
                continue
            if fid >= fidelity:
                out.append(record.config)
        return out

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._records)

    def __repr__(self) -> str:
        return f"ProxyDataset(records={len(self._records)}, " f"path={self._path!r})"

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def config_hash(config: TransformerConfig) -> str:
        """Return a stable SHA-256 hex digest of the config's architecture fields.

        Uses ``json.dumps`` with ``sort_keys=True`` for determinism.
        The hash captures all fields that define the *architecture* (not
        training hyper-parameters or measurement-only values).

        Returns
        -------
        str
            64-character lowercase hex string (SHA-256).
        """
        d = {
            "d_model": config.d_model,
            "n_heads": config.n_heads,
            "n_layers": config.n_layers,
            "seq_len": config.seq_len,
            "ff_mult": config.ff_mult,
            "activation": config.activation,
            "use_rotary": config.use_rotary,
            "use_layernorm": config.use_layernorm,
            "vocab_size": config.vocab_size,
            "tie_embeddings": config.tie_embeddings,
        }
        serialized = json.dumps(d, sort_keys=True)
        return hashlib.sha256(serialized.encode()).hexdigest()

    # ------------------------------------------------------------------
    # Bootstrap from zkml-bench results
    # ------------------------------------------------------------------

    @classmethod
    def from_zkml_bench_results(  # pylint: disable=too-many-locals
        cls, path: str
    ) -> "ProxyDataset":
        """Construct a :class:`ProxyDataset` from a zkml-bench JSON result file.

        The zkml-bench result format stores a list of benchmark results under
        the key ``"results"``.  Each result must have at least one of:

        * ``zk_proxy.constraint_estimate`` — used as ``onnx_estimate``
        * ``ezkl.num_rows`` — used as ``ezkl_num_rows``

        Results with neither field are skipped.

        A synthetic :class:`TransformerConfig` is constructed from the
        available proxy metrics (``param_count``, ``flops``,
        ``nonlinear_op_count``) using a heuristic that preserves the
        analytically-computable features (param count, MACs) while choosing
        plausible architecture dimensions consistent with those measurements.

        Parameters
        ----------
        path:
            Path to a zkml-bench JSON results file.  Supports both the flat
            ``{"results": [...]}`` wrapper format and the legacy bare-list
            format.

        Returns
        -------
        ProxyDataset
            Dataset populated with one record per result entry that has a
            usable label.

        Raises
        ------
        ValueError
            If the file cannot be parsed in either supported format.
        """
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)

        # Support both {"results": [...]} wrapper and bare list
        if isinstance(raw, list):
            result_list = raw
        elif isinstance(raw, dict) and "results" in raw:
            result_list = raw["results"]
        else:
            raise ValueError(
                f"Unrecognised zkml-bench result format in {path!r}: "
                "expected a dict with 'results' key or a bare list."
            )

        dataset = cls()

        for entry in result_list:
            zk_proxy = entry.get("zk_proxy") or {}
            ezkl_data = entry.get("ezkl") or {}

            constraint_estimate: Optional[int] = zk_proxy.get("constraint_estimate")
            ezkl_num_rows_raw = ezkl_data.get("num_rows")
            ezkl_num_rows: Optional[int] = (
                int(ezkl_num_rows_raw) if ezkl_num_rows_raw is not None else None
            )

            # Skip entries with no usable label.
            if constraint_estimate is None and ezkl_num_rows is None:
                continue

            # Build a synthetic config from available metrics.
            config = _infer_config_from_bench_entry(entry, zk_proxy)
            features = ProxyFeatures.from_config(config)

            # Propagate any ONNX proxy fields that are available in the result.
            nonlinear_ops = zk_proxy.get("nonlinear_op_count")
            linear_ops = zk_proxy.get("linear_op_count")
            if constraint_estimate is not None:
                features = ProxyFeatures(
                    d_model=features.d_model,
                    n_heads=features.n_heads,
                    n_layers=features.n_layers,
                    seq_len=features.seq_len,
                    ff_mult=features.ff_mult,
                    head_dim=features.head_dim,
                    total_params=features.total_params,
                    decoder_params=features.decoder_params,
                    attn_macs=features.attn_macs,
                    ffn_macs=features.ffn_macs,
                    total_macs=features.total_macs,
                    activation_id=features.activation_id,
                    is_zk_friendly=features.is_zk_friendly,
                    layernorm_count=features.layernorm_count,
                    softmax_count=features.softmax_count,
                    gelu_count=features.gelu_count,
                    onnx_constraint_estimate=constraint_estimate,
                    onnx_nonlinear_ops=nonlinear_ops,
                    onnx_linear_ops=linear_ops,
                )

            # Determine backend.
            backend = "ezkl"
            if ezkl_data.get("available") is False and ezkl_num_rows is None:
                backend = "onnx_only"

            record = ProxyRecord(
                config=config,
                features=features,
                onnx_estimate=constraint_estimate,
                ezkl_num_rows=ezkl_num_rows,
                backend=backend,
                notes=(
                    f"zkml-bench model_name={entry.get('model_name', 'unknown')} "
                    f"dataset={entry.get('dataset', 'unknown')}"
                ),
            )
            dataset.add(record)

        return dataset


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _infer_config_from_bench_entry(  # pylint: disable=too-many-locals
    entry: dict,
    zk_proxy: dict,
) -> TransformerConfig:
    """Derive a synthetic :class:`TransformerConfig` from a zkml-bench entry.

    zkml-bench results do not include the full architecture config.  We
    reconstruct a config whose *derived* features (param count, MACs,
    activation encoding) align with the measured values.

    Strategy
    --------
    1. Choose a plausible ``d_model`` / ``n_heads`` / ``n_layers`` by
       inverting the analytical parameter-count formula.
    2. Honour the measured ``param_count`` (if available) by back-solving
       for ``d_model``; fall back to small defaults otherwise.
    3. Determine ``activation`` from ZK proxy flags.
    """
    model_name: str = str(entry.get("model_name", "")).lower()
    param_count: Optional[int] = (
        int(zk_proxy.get("param_count") or entry.get("param_count") or 0) or None
    )

    # Determine activation from proxy flags.
    relu_incompatible: bool = bool(zk_proxy.get("relu_depth_incompatible", False))
    nonlinear_breakdown: dict = zk_proxy.get("nonlinear_breakdown") or {}
    nonlinear_op_count: int = int(zk_proxy.get("nonlinear_op_count") or 0)

    if relu_incompatible or "Relu" in nonlinear_breakdown:
        activation = "relu"
    elif nonlinear_op_count == 0:
        activation = "poly2"
    else:
        activation = "relu"

    # Heuristic: pick architecture dimensions consistent with model_name.
    defaults = _model_name_defaults(model_name)
    n_layers: int = defaults["n_layers"]
    n_heads: int = defaults["n_heads"]
    seq_len: int = defaults["seq_len"]
    ff_mult: float = defaults["ff_mult"]
    vocab_size: int = defaults["vocab_size"]

    # Back-solve d_model from param_count if available.
    # Formula (from ProxyFeatures.from_config, tied=False):
    #   total = vocab*d + n_layers*(4*d^2 + 2*ff*d^2 + 4*d) + vocab*d
    #   ≈ n_layers*(4+2*ff)*d^2  for large d
    d_model = defaults["d_model"]
    if param_count is not None and param_count > 0:
        # Rough quadratic inversion ignoring linear terms:
        # param_count ≈ 2*vocab*d + n_layers*(4+2*ff_mult)*d^2
        # Treat embedding/lm_head terms as dominant for large models.
        # Use bisection over small set of standard d_model values.
        candidate_d_models = [32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024]
        best_d = d_model
        best_err = float("inf")
        for d_cand in candidate_d_models:
            if d_cand % n_heads != 0:
                continue
            estimated = _analytical_param_count(
                d_model=d_cand,
                n_layers=n_layers,
                ff_mult=ff_mult,
                vocab_size=vocab_size,
                tie_embeddings=False,
            )
            err = abs(estimated - param_count)
            if err < best_err:
                best_err = err
                best_d = d_cand

        d_model = best_d

    # Ensure d_model divisible by n_heads.
    while d_model % n_heads != 0:
        n_heads = max(1, n_heads - 1)

    return TransformerConfig(
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        seq_len=seq_len,
        ff_mult=ff_mult,
        activation=activation,
        use_rotary=False,
        use_layernorm=True,
        vocab_size=vocab_size,
        tie_embeddings=False,
    )


def _analytical_param_count(
    d_model: int,
    n_layers: int,
    ff_mult: float,
    vocab_size: int,
    tie_embeddings: bool,
) -> int:
    """Replicate the ``ProxyFeatures.from_config`` parameter-count formula."""
    embedding_params = vocab_size * d_model
    per_layer = (
        4 * d_model * d_model  # QKV + O projections
        + int(2 * ff_mult * d_model * d_model)  # FFN
        + 4 * d_model  # 2 norm layers (weight + bias each)
    )
    lm_head = 0 if tie_embeddings else vocab_size * d_model
    return embedding_params + n_layers * per_layer + lm_head


def _model_name_defaults(model_name: str) -> dict:
    """Return default architecture parameters keyed on model name heuristics."""
    defaults: dict = {
        "n_layers": 4,
        "n_heads": 4,
        "d_model": 64,
        "seq_len": 64,
        "ff_mult": 4.0,
        "vocab_size": 50257,
    }

    if "large" in model_name or "wide" in model_name:
        defaults["d_model"] = 256
        defaults["n_heads"] = 8

    elif "small" in model_name or "tiny" in model_name:
        defaults["d_model"] = 32
        defaults["n_heads"] = 4

    if "transformer" in model_name or "vit" in model_name or "bert" in model_name:
        defaults["n_layers"] = 6
        defaults["n_heads"] = 8
        defaults["d_model"] = 128

    if "mlp" in model_name:
        defaults["n_heads"] = 1
        defaults["n_layers"] = 3
        defaults["d_model"] = 64

    # Fix n_heads so d_model is divisible.
    while defaults["d_model"] % defaults["n_heads"] != 0:
        defaults["n_heads"] = max(1, defaults["n_heads"] - 1)

    return defaults
