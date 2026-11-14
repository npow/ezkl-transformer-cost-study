"""Tests for proxy.data module."""

from __future__ import annotations

import json
import math
import os
import tempfile
from typing import Optional

import numpy as np
import pytest

from proxy.data import Fidelity, ProxyDataset, ProxyRecord
from proxy.features import ProxyFeatures, TransformerConfig, feature_names

# ---------------------------------------------------------------------------
# Test constants
# ---------------------------------------------------------------------------

ZKML_BENCH_RESULTS = "/root/code/zkml-bench/results/combined/results.json"

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_config(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    d_model: int = 64,
    n_heads: int = 4,
    n_layers: int = 4,
    seq_len: int = 64,
    ff_mult: float = 4.0,
    activation: str = "relu",
    vocab_size: int = 50257,
    tie_embeddings: bool = False,
    use_rotary: bool = False,
    use_layernorm: bool = True,
) -> TransformerConfig:
    return TransformerConfig(
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        seq_len=seq_len,
        ff_mult=ff_mult,
        activation=activation,
        vocab_size=vocab_size,
        tie_embeddings=tie_embeddings,
        use_rotary=use_rotary,
        use_layernorm=use_layernorm,
    )


def _make_record(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    config: Optional[TransformerConfig] = None,
    onnx_estimate: Optional[int] = 500_000,
    ezkl_num_rows: Optional[int] = None,
    ezkl_prove_time: Optional[float] = None,
    ezkl_calibration: Optional[dict] = None,
    backend: str = "ezkl",
) -> ProxyRecord:
    if config is None:
        config = _make_config()
    features = ProxyFeatures.from_config(config)
    return ProxyRecord(
        config=config,
        features=features,
        onnx_estimate=onnx_estimate,
        ezkl_num_rows=ezkl_num_rows,
        ezkl_prove_time=ezkl_prove_time,
        ezkl_calibration=ezkl_calibration,
        backend=backend,
    )


# ---------------------------------------------------------------------------
# Fidelity ordering
# ---------------------------------------------------------------------------


class TestFidelityOrdering:
    def test_onnx_less_than_ezkl_calibration(self) -> None:
        assert Fidelity.ONNX_ESTIMATE < Fidelity.EZKL_CALIBRATION

    def test_ezkl_calibration_less_than_ezkl_prove(self) -> None:
        assert Fidelity.EZKL_CALIBRATION < Fidelity.EZKL_PROVE

    def test_ezkl_prove_not_less_than_itself(self) -> None:
        assert Fidelity.EZKL_PROVE >= Fidelity.EZKL_PROVE

    def test_onnx_less_than_or_equal_to_onnx(self) -> None:
        assert Fidelity.ONNX_ESTIMATE <= Fidelity.ONNX_ESTIMATE

    def test_ezkl_prove_greater_than_onnx(self) -> None:
        assert Fidelity.EZKL_PROVE > Fidelity.ONNX_ESTIMATE

    def test_ezkl_prove_greater_or_equal_to_calibration(self) -> None:
        assert Fidelity.EZKL_PROVE >= Fidelity.EZKL_CALIBRATION

    def test_transitive_ordering_onnx_lt_prove(self) -> None:
        assert Fidelity.ONNX_ESTIMATE < Fidelity.EZKL_PROVE


# ---------------------------------------------------------------------------
# ProxyRecord.best_label
# ---------------------------------------------------------------------------


class TestBestLabel:
    def test_no_labels_returns_none(self) -> None:
        record = _make_record(onnx_estimate=None)
        assert record.best_label() is None

    def test_onnx_only(self) -> None:
        record = _make_record(onnx_estimate=548_864)
        assert record.best_label() == pytest.approx(math.log(548_864))

    def test_ezkl_num_rows_preferred_over_onnx(self) -> None:
        record = _make_record(onnx_estimate=548_864, ezkl_num_rows=2_097_152)
        assert record.best_label() == pytest.approx(math.log(2_097_152))

    def test_ezkl_prove_time_used_when_no_num_rows(self) -> None:
        prove_time = 180.5
        record = _make_record(onnx_estimate=548_864, ezkl_prove_time=prove_time)
        expected = math.log(prove_time * 1000.0)
        assert record.best_label() == pytest.approx(expected)

    def test_ezkl_num_rows_beats_prove_time(self) -> None:
        record = _make_record(
            onnx_estimate=548_864,
            ezkl_prove_time=180.5,
            ezkl_num_rows=2_097_152,
        )
        assert record.best_label() == pytest.approx(math.log(2_097_152))

    def test_zero_values_treated_as_missing(self) -> None:
        record = _make_record(onnx_estimate=0, ezkl_num_rows=0)
        assert record.best_label() is None


# ---------------------------------------------------------------------------
# ProxyRecord.fidelity_level
# ---------------------------------------------------------------------------


class TestFidelityLevel:
    def test_none_when_no_labels(self) -> None:
        record = _make_record(onnx_estimate=None)
        assert record.fidelity_level() is None

    def test_onnx_only_returns_onnx_fidelity(self) -> None:
        record = _make_record(onnx_estimate=548_864)
        assert record.fidelity_level() == Fidelity.ONNX_ESTIMATE

    def test_ezkl_calibration_returns_calibration_fidelity(self) -> None:
        record = _make_record(
            onnx_estimate=None,
            ezkl_calibration={"k": 21, "num_rows": 2_097_152},
        )
        assert record.fidelity_level() == Fidelity.EZKL_CALIBRATION

    def test_ezkl_num_rows_returns_prove_fidelity(self) -> None:
        record = _make_record(ezkl_num_rows=2_097_152)
        assert record.fidelity_level() == Fidelity.EZKL_PROVE

    def test_ezkl_prove_time_returns_prove_fidelity(self) -> None:
        record = _make_record(ezkl_prove_time=123.4, ezkl_num_rows=None)
        assert record.fidelity_level() == Fidelity.EZKL_PROVE


# ---------------------------------------------------------------------------
# ProxyDataset – deduplication
# ---------------------------------------------------------------------------


class TestDeduplication:
    def test_same_config_not_duplicated(self) -> None:
        dataset = ProxyDataset()
        config = _make_config()
        r1 = _make_record(config=config, onnx_estimate=100_000)
        r2 = _make_record(config=config, onnx_estimate=200_000)
        dataset.add(r1)
        dataset.add(r2)
        assert len(dataset) == 1

    def test_duplicate_merges_higher_fidelity(self) -> None:
        dataset = ProxyDataset()
        config = _make_config()
        r1 = _make_record(config=config, onnx_estimate=100_000)
        r2 = _make_record(config=config, onnx_estimate=200_000, ezkl_num_rows=2_097_152)
        dataset.add(r1)
        dataset.add(r2)
        assert len(dataset) == 1
        assert dataset._records[0].ezkl_num_rows == 2_097_152

    def test_different_configs_both_kept(self) -> None:
        dataset = ProxyDataset()
        r1 = _make_record(config=_make_config(d_model=64, n_heads=4))
        r2 = _make_record(config=_make_config(d_model=128, n_heads=8))
        dataset.add(r1)
        dataset.add(r2)
        assert len(dataset) == 2

    def test_add_same_record_twice_stays_one(self) -> None:
        dataset = ProxyDataset()
        r = _make_record()
        dataset.add(r)
        dataset.add(r)
        assert len(dataset) == 1


# ---------------------------------------------------------------------------
# ProxyDataset – serialization round-trip
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_save_and_load_empty_dataset(self) -> None:
        dataset = ProxyDataset()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            dataset.save(path)
            loaded = ProxyDataset.load(path)
            assert len(loaded) == 0
        finally:
            os.unlink(path)

    def test_round_trip_preserves_all_records(self) -> None:
        dataset = ProxyDataset()
        configs = [
            _make_config(d_model=64, n_heads=4),
            _make_config(d_model=128, n_heads=8),
            _make_config(d_model=256, n_heads=8, n_layers=6),
        ]
        for i, cfg in enumerate(configs):
            r = _make_record(
                config=cfg,
                onnx_estimate=(i + 1) * 100_000,
                ezkl_num_rows=(i + 1) * 1_000_000 if i > 0 else None,
            )
            dataset.add(r)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            dataset.save(path)
            loaded = ProxyDataset.load(path)
            assert len(loaded) == len(dataset)
            for orig, reloaded in zip(dataset._records, loaded._records):
                assert orig.onnx_estimate == reloaded.onnx_estimate
                assert orig.ezkl_num_rows == reloaded.ezkl_num_rows
                assert orig.backend == reloaded.backend
                assert orig.record_id == reloaded.record_id
        finally:
            os.unlink(path)

    def test_round_trip_preserves_config_fields(self) -> None:
        config = _make_config(
            d_model=256,
            n_heads=8,
            n_layers=6,
            seq_len=128,
            ff_mult=8.0,
            activation="gelu",
            vocab_size=32000,
            tie_embeddings=True,
            use_rotary=True,
            use_layernorm=False,
        )
        record = _make_record(config=config, onnx_estimate=1_000_000)
        dataset = ProxyDataset()
        dataset.add(record)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            dataset.save(path)
            loaded = ProxyDataset.load(path)
            c = loaded._records[0].config
            assert c.d_model == 256
            assert c.n_heads == 8
            assert c.n_layers == 6
            assert c.seq_len == 128
            assert c.ff_mult == pytest.approx(8.0)
            assert c.activation == "gelu"
            assert c.vocab_size == 32000
            assert c.tie_embeddings is True
            assert c.use_rotary is True
            assert c.use_layernorm is False
        finally:
            os.unlink(path)

    def test_json_contains_no_numpy_types(self) -> None:
        """Saved JSON must be parseable without numpy type errors."""
        dataset = ProxyDataset()
        dataset.add(_make_record())
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            dataset.save(path)
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            assert "records" in data
        finally:
            os.unlink(path)

    def test_load_nonexistent_path_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            ProxyDataset.load("/tmp/_nonexistent_zkml_nas_test_path_.json")

    def test_constructor_with_nonexistent_path_starts_empty(self) -> None:
        dataset = ProxyDataset(path="/tmp/_nonexistent_zkml_nas_test_path_2.json")
        assert len(dataset) == 0

    def test_round_trip_preserves_ezkl_calibration(self) -> None:
        calib = {"k": 21, "num_rows": 2_097_152, "num_cols": 512}
        record = _make_record(
            onnx_estimate=None,
            ezkl_calibration=calib,
        )
        dataset = ProxyDataset()
        dataset.add(record)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            dataset.save(path)
            loaded = ProxyDataset.load(path)
            assert loaded._records[0].ezkl_calibration == calib
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# ProxyDataset – to_xy
# ---------------------------------------------------------------------------


class TestToXY:
    def test_empty_dataset_returns_empty_arrays(self) -> None:
        dataset = ProxyDataset()
        x_mat, y_vec = dataset.to_xy()
        assert x_mat.shape[0] == 0
        assert y_vec.shape[0] == 0
        assert x_mat.shape[1] == len(feature_names())

    def test_shapes_are_correct(self) -> None:
        dataset = ProxyDataset()
        for i in range(5):
            r = _make_record(
                config=_make_config(
                    d_model=64 * (i + 1),
                    n_heads=4 * (i + 1),
                ),
                onnx_estimate=(i + 1) * 100_000,
            )
            dataset.add(r)
        x_mat, y_vec = dataset.to_xy()
        assert x_mat.ndim == 2
        assert y_vec.ndim == 1
        assert x_mat.shape[0] == y_vec.shape[0] == 5
        assert x_mat.shape[1] == len(feature_names())

    def test_records_without_labels_excluded(self) -> None:
        dataset = ProxyDataset()
        r_with = _make_record(onnx_estimate=500_000)
        r_without = _make_record(
            config=_make_config(d_model=128, n_heads=8),
            onnx_estimate=None,
        )
        dataset.add(r_with)
        dataset.add(r_without)
        x_mat, _ = dataset.to_xy()
        assert x_mat.shape[0] == 1

    def test_min_fidelity_filters_correctly(self) -> None:
        dataset = ProxyDataset()
        # Only onnx_estimate → ONNX_ESTIMATE fidelity
        r_onnx = _make_record(
            config=_make_config(d_model=64, n_heads=4),
            onnx_estimate=500_000,
        )
        # ezkl_num_rows → EZKL_PROVE fidelity
        r_proved = _make_record(
            config=_make_config(d_model=128, n_heads=8),
            onnx_estimate=1_000_000,
            ezkl_num_rows=2_097_152,
        )
        dataset.add(r_onnx)
        dataset.add(r_proved)

        x_all, _ = dataset.to_xy(min_fidelity=Fidelity.ONNX_ESTIMATE)
        assert x_all.shape[0] == 2

        x_proved, _ = dataset.to_xy(min_fidelity=Fidelity.EZKL_PROVE)
        assert x_proved.shape[0] == 1

    def test_backend_filter(self) -> None:
        dataset = ProxyDataset()
        r_ezkl = _make_record(
            config=_make_config(d_model=64, n_heads=4),
            onnx_estimate=500_000,
            backend="ezkl",
        )
        r_expander = _make_record(
            config=_make_config(d_model=128, n_heads=8),
            onnx_estimate=1_000_000,
            backend="expander",
        )
        dataset.add(r_ezkl)
        dataset.add(r_expander)

        x_ezkl, _ = dataset.to_xy(backend="ezkl")
        assert x_ezkl.shape[0] == 1

        x_all, _ = dataset.to_xy(backend=None)
        assert x_all.shape[0] == 2

    def test_y_values_are_log_labels(self) -> None:
        estimate = 548_864
        dataset = ProxyDataset()
        dataset.add(_make_record(onnx_estimate=estimate))
        _, y_vec = dataset.to_xy()
        assert y_vec[0] == pytest.approx(math.log(estimate))

    def test_x_dtype_is_float64(self) -> None:
        dataset = ProxyDataset()
        dataset.add(_make_record())
        x_mat, _ = dataset.to_xy()
        assert x_mat.dtype == np.float64

    def test_ezkl_fidelity_filter_with_calibration(self) -> None:
        dataset = ProxyDataset()
        r_calib = _make_record(
            config=_make_config(d_model=64, n_heads=4),
            onnx_estimate=None,
            ezkl_calibration={"k": 21},
        )
        r_onnx = _make_record(
            config=_make_config(d_model=128, n_heads=8),
            onnx_estimate=500_000,
        )
        dataset.add(r_calib)
        dataset.add(r_onnx)

        # r_calib has fidelity=EZKL_CALIBRATION but no numeric label (no num_rows /
        # prove_time), so best_label() returns None and it is excluded from to_xy.
        x_calib, _ = dataset.to_xy(min_fidelity=Fidelity.EZKL_CALIBRATION)
        assert x_calib.shape[0] == 0


# ---------------------------------------------------------------------------
# ProxyDataset – configs_proved
# ---------------------------------------------------------------------------


class TestConfigsProved:
    def test_empty_dataset(self) -> None:
        dataset = ProxyDataset()
        assert not dataset.configs_proved()

    def test_only_onnx_not_included_at_prove_level(self) -> None:
        dataset = ProxyDataset()
        r = _make_record(onnx_estimate=500_000)
        dataset.add(r)
        assert not dataset.configs_proved(Fidelity.EZKL_PROVE)

    def test_proved_record_included(self) -> None:
        dataset = ProxyDataset()
        config = _make_config()
        r = _make_record(config=config, ezkl_num_rows=2_097_152)
        dataset.add(r)
        proved = dataset.configs_proved(Fidelity.EZKL_PROVE)
        assert len(proved) == 1
        assert proved[0].d_model == config.d_model

    def test_calibration_fidelity_filter(self) -> None:
        dataset = ProxyDataset()
        r_calib = _make_record(
            config=_make_config(d_model=64, n_heads=4),
            ezkl_calibration={"k": 21, "num_rows": 2_097_152},
        )
        r_onnx = _make_record(
            config=_make_config(d_model=128, n_heads=8),
            onnx_estimate=500_000,
        )
        dataset.add(r_calib)
        dataset.add(r_onnx)

        # At EZKL_CALIBRATION level, only calib record qualifies
        calib_configs = dataset.configs_proved(Fidelity.EZKL_CALIBRATION)
        assert len(calib_configs) == 1

        # At ONNX_ESTIMATE level, both qualify
        onnx_configs = dataset.configs_proved(Fidelity.ONNX_ESTIMATE)
        assert len(onnx_configs) == 2

    def test_proved_record_returned_at_onnx_level(self) -> None:
        """A record with EZKL_PROVE fidelity satisfies ONNX_ESTIMATE threshold."""
        dataset = ProxyDataset()
        dataset.add(_make_record(ezkl_num_rows=2_097_152))
        assert len(dataset.configs_proved(Fidelity.ONNX_ESTIMATE)) == 1


# ---------------------------------------------------------------------------
# ProxyDataset – config_hash
# ---------------------------------------------------------------------------


class TestConfigHash:
    def test_same_config_same_hash(self) -> None:
        c1 = _make_config()
        c2 = _make_config()
        assert ProxyDataset.config_hash(c1) == ProxyDataset.config_hash(c2)

    def test_different_d_model_different_hash(self) -> None:
        c1 = _make_config(d_model=64)
        c2 = _make_config(d_model=128, n_heads=8)
        assert ProxyDataset.config_hash(c1) != ProxyDataset.config_hash(c2)

    def test_different_activation_different_hash(self) -> None:
        c1 = _make_config(activation="relu")
        c2 = _make_config(activation="gelu")
        assert ProxyDataset.config_hash(c1) != ProxyDataset.config_hash(c2)

    def test_hash_is_64_char_hex_string(self) -> None:
        h = ProxyDataset.config_hash(_make_config())
        assert isinstance(h, str)
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_hash_is_deterministic_across_instances(self) -> None:
        """Two independently constructed configs with same params must hash equal."""
        h1 = ProxyDataset.config_hash(
            TransformerConfig(d_model=256, n_heads=8, n_layers=6, seq_len=128)
        )
        h2 = ProxyDataset.config_hash(
            TransformerConfig(d_model=256, n_heads=8, n_layers=6, seq_len=128)
        )
        assert h1 == h2


# ---------------------------------------------------------------------------
# ProxyDataset – from_zkml_bench_results
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.path.exists(ZKML_BENCH_RESULTS),
    reason=f"zkml-bench results file not found: {ZKML_BENCH_RESULTS}",
)
class TestFromZkmlBenchResults:
    def test_parses_without_error(self) -> None:
        dataset = ProxyDataset.from_zkml_bench_results(ZKML_BENCH_RESULTS)
        assert isinstance(dataset, ProxyDataset)

    def test_has_at_least_one_record(self) -> None:
        dataset = ProxyDataset.from_zkml_bench_results(ZKML_BENCH_RESULTS)
        assert len(dataset) >= 1

    def test_all_records_have_best_label(self) -> None:
        dataset = ProxyDataset.from_zkml_bench_results(ZKML_BENCH_RESULTS)
        for record in dataset._records:
            assert record.best_label() is not None, (
                f"Record {record.record_id!r} has no label "
                f"(onnx_estimate={record.onnx_estimate!r}, "
                f"ezkl_num_rows={record.ezkl_num_rows!r})"
            )

    def test_ezkl_num_rows_parsed_correctly(self) -> None:
        """Records with ezkl.num_rows should have that as their primary label."""
        dataset = ProxyDataset.from_zkml_bench_results(ZKML_BENCH_RESULTS)
        proved_records = [r for r in dataset._records if r.ezkl_num_rows is not None]
        assert len(proved_records) >= 1, "Expected at least one ezkl.num_rows record"
        for r in proved_records:
            assert r.ezkl_num_rows > 0
            assert r.fidelity_level() == Fidelity.EZKL_PROVE

    def test_onnx_estimate_parsed_correctly(self) -> None:
        dataset = ProxyDataset.from_zkml_bench_results(ZKML_BENCH_RESULTS)
        onnx_records = [r for r in dataset._records if r.onnx_estimate is not None]
        assert len(onnx_records) >= 1
        for r in onnx_records:
            assert r.onnx_estimate > 0

    def test_to_xy_returns_valid_shapes(self) -> None:
        dataset = ProxyDataset.from_zkml_bench_results(ZKML_BENCH_RESULTS)
        x_mat, y_vec = dataset.to_xy()
        assert x_mat.ndim == 2
        assert y_vec.ndim == 1
        assert x_mat.shape[0] == y_vec.shape[0]
        assert x_mat.shape[1] == len(feature_names())
        assert x_mat.shape[0] >= 1

    def test_no_nan_in_features(self) -> None:
        dataset = ProxyDataset.from_zkml_bench_results(ZKML_BENCH_RESULTS)
        x_mat, y_vec = dataset.to_xy()
        assert not np.any(np.isnan(x_mat)), "Feature matrix contains NaN values"
        assert not np.any(np.isnan(y_vec)), "Label vector contains NaN values"

    def test_deduplication_applied(self) -> None:
        """Parsing the same file twice and re-adding all records must not grow the dataset."""
        d1 = ProxyDataset.from_zkml_bench_results(ZKML_BENCH_RESULTS)
        d2 = ProxyDataset()
        for record in d1._records:
            d2.add(record)
            d2.add(record)  # duplicate
        assert len(d2) == len(d1)

    def test_bare_list_format_supported(self) -> None:
        """from_zkml_bench_results should accept a bare JSON list."""
        with open(ZKML_BENCH_RESULTS, encoding="utf-8") as fh:
            raw = json.load(fh)
        results_list = raw.get("results", [])

        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w", encoding="utf-8"
        ) as f:
            json.dump(results_list, f)
            tmp_path = f.name

        try:
            dataset = ProxyDataset.from_zkml_bench_results(tmp_path)
            assert len(dataset) >= 1
        finally:
            os.unlink(tmp_path)

    def test_onnx_constraint_propagated_to_features(self) -> None:
        """When constraint_estimate is available, features.onnx_constraint_estimate
        must be populated."""
        dataset = ProxyDataset.from_zkml_bench_results(ZKML_BENCH_RESULTS)
        for record in dataset._records:
            if record.onnx_estimate is not None:
                assert record.features.onnx_constraint_estimate == record.onnx_estimate
                break
        else:
            pytest.skip("No records with onnx_estimate in test file")
