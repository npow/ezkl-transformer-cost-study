#!/usr/bin/env python3
"""Aggregate raw artifacts into the numbers reported in the paper.

Reads ``artifacts/ezkl_validation/*.json`` and ``artifacts/causal_study/*.json``
and writes ``artifacts/summary.json``.

EZKL: rank agreement of the analytical model AND trivial baselines, cluster
bootstrap CIs over shapes (the 4 activations at a shape are near-duplicates),
log-log fits, a leave-one-width-out held-out test, activation overheads, the
softmax x FFN decomposition, and logrows.
LM: mean and *sample* s.d. (ddof=1) at the validation-selected checkpoint, plus
final-checkpoint values.
"""
from __future__ import annotations

import glob
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent / "artifacts"
RNG = np.random.default_rng(0)


def _load(pattern):
    return [json.load(open(f)) for f in sorted(glob.glob(str(ROOT / pattern)))]


def _r2(y, pred):
    return float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def ezkl_summary(sub: str = "ezkl_validation") -> dict:
    recs = _load(f"{sub}/d*.json")
    ok = [r for r in recs if r["status"] == "ok"]
    out = {"n_total": len(recs), "n_ok": len(ok), "n_failed": len(recs) - len(ok)}
    if len(ok) < 8:
        return out
    m = np.array([r["num_rows"] for r in ok], float)
    a = np.array([r["analytical_constraints"] for r in ok], float)
    d = np.array([r["d_model"] for r in ok], float)
    L = np.array([r["n_layers"] for r in ok], float)
    T = np.array([r["seq_len"] for r in ok], float)
    acts = np.array([r["activation"] for r in ok])
    shape = [(r["d_model"], r["n_layers"], r["seq_len"]) for r in ok]

    sp = lambda x: float(stats.spearmanr(x, m)[0])  # noqa: E731
    preds = {"analytical": a, "d*L*T": d * L * T, "L*T": L * T, "d^2*L*T": d * d * L * T}
    out["spearman"] = {k: sp(v) for k, v in preds.items()}
    out["spearman_within_activation"] = {
        k: {act: float(stats.spearmanr(v[acts == act], m[acts == act])[0]) for act in sorted(set(acts))}
        for k, v in preds.items()}

    # cluster bootstrap over shapes
    shapes = sorted(set(shape))
    idx_by_shape = {s: [i for i, x in enumerate(shape) if x == s] for s in shapes}
    boots = defaultdict(list)
    for _ in range(2000):
        pick = RNG.integers(0, len(shapes), len(shapes))
        ix = np.concatenate([idx_by_shape[shapes[j]] for j in pick])
        for k in ("analytical", "d*L*T"):
            boots[k].append(stats.spearmanr(preds[k][ix], m[ix])[0])
        boots["diff"].append(boots["analytical"][-1] - boots["d*L*T"][-1])
    out["spearman_cluster_bootstrap_95ci"] = {
        k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in boots.items()}
    out["n_shapes"] = len(shapes)

    # log-log fits
    ly = np.log(m)
    X = np.c_[np.ones(len(m)), np.log(d), np.log(L), np.log(T)]
    b = np.linalg.lstsq(X, ly, rcond=None)[0]
    out["loglog_dLT_fit"] = {"const": float(b[0]), "d": float(b[1]), "L": float(b[2]), "T": float(b[3]),
                             "r2": _r2(ly, X @ b)}
    Xa = np.c_[np.ones(len(m)), np.log(a)]
    ba = np.linalg.lstsq(Xa, ly, rcond=None)[0]
    out["loglog_analytical_fit"] = {"const": float(ba[0]), "slope": float(ba[1]), "r2": _r2(ly, Xa @ ba)}
    ratio = m / a
    out["ratio_measured_over_modelled"] = {"median": float(np.median(ratio)), "min": float(ratio.min()),
                                           "max": float(ratio.max())}
    # per-activation exponents (rows ~ d^a L^b T^c)
    out["loglog_dLT_fit_by_activation"] = {}
    for act in sorted(set(acts)):
        s = acts == act
        bb = np.linalg.lstsq(X[s], ly[s], rcond=None)[0]
        out["loglog_dLT_fit_by_activation"][act] = {"d": float(bb[1]), "L": float(bb[2]), "T": float(bb[3])}

    # leave-one-width-out held-out error of log-rows (mean |relative error| in %)
    def heldout(design):
        errs = []
        for w in (64, 96, 128):
            tr, te = d != w, d == w
            coef = np.linalg.lstsq(design[tr], ly[tr], rcond=None)[0]
            errs.append(np.abs(np.exp(design[te] @ coef) / m[te] - 1).mean() * 100)
        return float(np.mean(errs))
    A = np.c_[np.ones(len(m)), *(acts == k for k in ("gelu", "relu", "poly2"))]
    out["heldout_width_mean_abs_pct_error"] = {
        "analytical_x_const": heldout(np.c_[np.ones(len(m)), np.log(a)]),
        "analytical_x_activation_const": heldout(np.c_[A, np.log(a)]),
        "loglog_dLT_plus_activation": heldout(np.c_[X, A[:, 1:]]),
    }
    # NB: extrapolating to unseen *widths* tests exactly the scale-transfer question.

    # activation overhead relative to the linear variant (same shape)
    by_shape = defaultdict(dict)
    for r in ok:
        by_shape[(r["d_model"], r["n_layers"], r["seq_len"])][r["activation"]] = r["num_rows"]
    rel = defaultdict(list)
    rel_by_width = defaultdict(lambda: defaultdict(list))
    for k, v in by_shape.items():
        if "linear" in v:
            for act, rows in v.items():
                if act != "linear":
                    rel[act].append(rows / v["linear"])
                    rel_by_width[act][k[0]].append(rows / v["linear"])
    out["rows_vs_linear_median"] = {k: float(np.median(v)) for k, v in rel.items()}
    out["rows_vs_linear_by_width_mean"] = {act: {w: float(np.mean(v)) for w, v in ws.items()}
                                           for act, ws in rel_by_width.items()}
    out["logrows_by_activation"] = {act: sorted({r["logrows"] for r in ok if r["activation"] == act})
                                    for act in sorted(set(acts))}
    out["total_assignments_over_num_rows"] = float(np.median(
        [r["total_assignments"] / r["num_rows"] for r in ok if r.get("total_assignments")]))

    # softmax x FFN decomposition
    dec = _load(f"{sub}/ab_*.json")
    dec = [r for r in dec if r["status"] == "ok"]
    if dec:
        tab = defaultdict(dict)
        for r in dec:
            tab[(r["d_model"], r["seq_len"])][(r["attention"], r["ffn"])] = {
                "rows": r["num_rows"], "logrows": r["logrows"], "param_scale": r.get("param_scale")}
        out["decomposition"] = {f"d{w}_T{t}": {f"attn-{a_}/ffn-{f}": v for (a_, f), v in cells.items()}
                                for (w, t), cells in sorted(tab.items())}
        # relative rows vs (linear attention, identity FFN) at each (d, T)
        rel = {}
        for (w, t), cells in sorted(tab.items()):
            base = cells.get(("linear", "linear"))
            if base:
                rel[f"d{w}_T{t}"] = {f"attn-{a_}/ffn-{f}": round(v["rows"] / base["rows"], 4)
                                     for (a_, f), v in cells.items()}
        out["decomposition_relative_rows"] = rel
    mf = ROOT / sub / "maskfill_sensitivity.json"
    if mf.exists():
        out["maskfill_sensitivity"] = json.load(open(mf))
    out.update(_extra_models(ok, acts, m, d, L, T, a))
    sc = defaultdict(lambda: defaultdict(int))
    for r in ok:
        sc[r["activation"]][r.get("param_scale")] += 1
    out["param_scale_counts"] = {k: dict(v) for k, v in sc.items()}
    out["total_assignments_exactly_2x_rows"] = int(sum(
        1 for r in ok if r.get("total_assignments") == 2 * r["num_rows"]))
    return out


def _components(ok):
    """Per-config analytical components (cached in artifacts/model_components.json)."""
    cache = ROOT / "model_components.json"
    data = json.load(open(cache)) if cache.exists() else {}
    need = [r for r in ok if r["name"] not in data]
    if need:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from experiments.operator_attribution import compute_op_breakdown
        from proxy.features import TransformerConfig
        for r in need:
            cfg = TransformerConfig(r["d_model"], 4, r["n_layers"], r["seq_len"], 4.0, r["activation"],
                                    False, True, 512, True)
            b = compute_op_breakdown(cfg)
            data[r["name"]] = {
                "matmul": b.attn_qkv_proj_constraints + b.attn_score_constraints + b.attn_value_constraints
                          + b.attn_out_proj_constraints + b.ffn_up_constraints + b.ffn_down_constraints,
                "softmax": b.softmax_constraints, "activation": b.activation_constraints,
                "layernorm": b.layernorm_constraints, "embedding": b.embedding_constraints,
                "lm_head": 2 * r["seq_len"] * r["d_model"] * 512}  # NOT in the analytical model
        cache.write_text(json.dumps(data, indent=1))
    return [data[r["name"]] for r in ok]


def _extra_models(ok, acts, m, d, L, T, a):
    """Leave-one-axis-out error for all predictors + a calibrated additive (NNLS) competitor."""
    from scipy.optimize import nnls
    out = {}
    ly = np.log(m)
    A = np.c_[(acts == "gelu"), (acts == "relu"), (acts == "poly2")].astype(float)
    one = np.ones(len(m))
    designs = {
        "analytical (log-log, 2 params + activation offsets)": np.c_[one, np.log(a), A],
        "d*L*T (log-log, 2 params + activation offsets)": np.c_[one, np.log(d * L * T), A],
        "power law d^x L^y T^z (+ activation offsets)": np.c_[one, np.log(d), np.log(L), np.log(T), A],
    }
    axes = {"d": d, "L": L, "T": T}

    def loo_ols(X, ax):
        errs = []
        for v in sorted(set(ax)):
            tr, te = ax != v, ax == v
            c = np.linalg.lstsq(X[tr], ly[tr], rcond=None)[0]
            errs.append(np.abs(np.exp(X[te] @ c) / m[te] - 1).mean() * 100)
        return float(np.mean(errs))

    res = {k: {ax: round(loo_ols(X, v), 1) for ax, v in axes.items()} for k, X in designs.items()}
    try:
        comp = _components(ok)
    except Exception as exc:  # zkml-free now, but matplotlib may be missing
        out["nnls_error"] = str(exc)[:120]
        comp = None
    if comp:
        def feats(with_head):
            keys = ["matmul", "softmax", "activation", "layernorm", "embedding"] + (["lm_head"] if with_head else [])
            return np.array([[c[k] for k in keys] for c in comp], float), keys

        def loo_nnls(with_head, ax):
            F, _ = feats(with_head)
            sc = F.max(axis=0)
            Fs = F / sc
            errs = []
            for v in sorted(set(ax)):
                tr, te = ax != v, ax == v
                w, _ = nnls(Fs[tr], m[tr])
                errs.append(np.abs((Fs[te] @ w) / m[te] - 1).mean() * 100)
            return float(np.mean(errs))
        for wh, name in ((False, "additive NNLS over the model's operator categories"),
                         (True, "additive NNLS + output-projection (LM head) term")):
            res[name] = {ax: round(loo_nnls(wh, v), 1) for ax, v in axes.items()}
            F, keys = feats(wh)
            sc = F.max(axis=0)
            w, _ = nnls(F / sc, m)
            pred = (F / sc) @ w
            out.setdefault("nnls_fit", {})[name] = {
                "spearman_in_sample": float(stats.spearmanr(pred, m)[0]),
                "weights_rows_per_constraint": {k: float(wi / si) for k, wi, si in zip(keys, w, sc)}}
            # within-shape ordering of the four variants
            by = defaultdict(dict)
            for i, r in enumerate(ok):
                by[(r["d_model"], r["n_layers"], r["seq_len"])][r["activation"]] = (m[i], pred[i])
            bad = tot = 0
            import itertools
            for v in by.values():
                for p, q in itertools.combinations(v, 2):
                    tot += 1
                    bad += (v[p][0] - v[q][0]) * (v[p][1] - v[q][1]) <= 0
            out["nnls_fit"][name]["within_shape_misordered_pairs"] = f"{int(bad)}/{tot}"
    out["leave_one_axis_out_mean_abs_pct_error"] = res
    by = defaultdict(dict)
    for i, r in enumerate(ok):
        by[(r["d_model"], r["n_layers"], r["seq_len"])][r["activation"]] = (m[i], a[i])
    import itertools
    bad = tot = 0
    for v in by.values():
        for p, q in itertools.combinations(v, 2):
            tot += 1
            bad += (v[p][0] - v[q][0]) * (v[p][1] - v[q][1]) <= 0
    out["analytical_within_shape_misordered_pairs"] = f"{int(bad)}/{tot}"
    return out


def lm_summary() -> dict:
    groups = defaultdict(list)
    for r in _load("causal_study/*.json"):
        if "val_ppl" in r and "seed" in r:
            groups[r["name"] + ("+kd" if r.get("distill_alpha") else "")].append(r)
    sd = lambda v: st.stdev(v) if len(v) > 1 else 0.0  # noqa: E731  (sample s.d.)
    out = {}
    for k, rs in groups.items():
        g = lambda f: [r[f] for r in rs if f in r]  # noqa: E731
        out[k] = {
            "n_seeds": len(rs), "n_params": rs[0].get("n_params"),
            "val_sel_mean": st.mean(g("val_ppl")), "val_sel_sd": sd(g("val_ppl")),
            "test_sel_mean": st.mean(g("test_ppl")), "test_sel_sd": sd(g("test_ppl")),
            "best_step_mean": st.mean(g("best_step")) if g("best_step") else None,
            "val_final_mean": st.mean(g("final_val_ppl")) if g("final_val_ppl") else None,
            "test_final_mean": st.mean(g("final_test_ppl")) if g("final_test_ppl") else None,
        }
    z = ROOT / "causal_study" / "gpt2_pretrained_zero_shot.json"
    if z.exists():
        out["gpt2_pretrained_zero_shot"] = json.load(open(z))
    return out


def extra_stats() -> dict:
    """NNLS bootstrap stability, proving-time baselines, within-shape ordering, paired quality tests."""
    import itertools
    from scipy.optimize import nnls
    out = {}
    R = _load("ezkl_validation/d*.json")
    cache = ROOT / "model_components.json"
    if R and cache.exists():
        comp = json.load(open(cache))
        keys = ["matmul", "softmax", "activation", "layernorm", "embedding"]
        F = np.array([[comp[r["name"]][k] for k in keys] for r in R], float)
        m = np.array([r["num_rows"] for r in R], float)
        sc = F.max(0)
        shape = [(r["d_model"], r["n_layers"], r["seq_len"]) for r in R]
        shapes = sorted(set(shape))
        idx = {s_: [i for i, x in enumerate(shape) if x == s_] for s_ in shapes}
        rng = np.random.default_rng(0)
        W = []
        for _ in range(1000):
            pick = rng.integers(0, len(shapes), len(shapes))
            ix = np.concatenate([idx[shapes[j]] for j in pick])
            W.append(nnls(F[ix] / sc, m[ix])[0] / sc)
        W = np.array(W)
        out["nnls_cluster_bootstrap"] = {k: {"frac_zero": float(np.mean(W[:, j] == 0)),
                                             "median": float(np.median(W[:, j])),
                                             "ci95": [float(np.percentile(W[:, j], 2.5)), float(np.percentile(W[:, j], 97.5))]}
                                         for j, k in enumerate(keys)}
        out["spearman_matmul_vs_layernorm_feature"] = float(stats.spearmanr(F[:, 0], F[:, 3])[0])

        def r2(f):
            w, _ = nnls(f / f.max(0), m)
            p_ = (f / f.max(0)) @ w
            return float(1 - ((m - p_) ** 2).sum() / ((m - m.mean()) ** 2).sum())
        out["nnls_r2"] = {"all": r2(F), "matmul_only": r2(F[:, [0]]), "without_matmul": r2(F[:, 1:])}
    P = _load("ezkl_proving/d*.json")
    P = [p for p in P if p.get("status") == "ok" and p.get("verified")]
    if P:
        t = np.array([p["prove_s"] for p in P])
        rows = np.array([p["num_rows"] for p in P], float)
        k = np.array([p["logrows"] for p in P], float)
        dlt = np.array([p["d_model"] * p["n_layers"] * p["seq_len"] for p in P], float)
        an = np.array([json.load(open(ROOT / "ezkl_validation" / f"{p['name']}.json"))["analytical_constraints"] for p in P], float)
        sp = lambda x: float(stats.spearmanr(x, t)[0])  # noqa: E731
        out["proving_time_spearman"] = {"rows": sp(rows), "d*L*s": sp(dlt), "analytical": sp(an), "logrows": sp(k)}
        by = defaultdict(dict)
        for p in P:
            by[(p["d_model"], p["n_layers"], p["seq_len"])][p["activation"]] = p
        agree = tot = 0
        for v in by.values():
            for a_, b_ in itertools.combinations(v, 2):
                tot += 1
                agree += (v[a_]["num_rows"] - v[b_]["num_rows"]) * (v[a_]["prove_s"] - v[b_]["prove_s"]) > 0
        shp = [(p["d_model"], p["n_layers"], p["seq_len"]) for p in P]
        ush = sorted(set(shp))
        ix_by = {u: [i for i, x in enumerate(shp) if x == u] for u in ush}
        brng = np.random.default_rng(1)
        bs = {"rows": [], "d*L*s": []}
        for _ in range(2000):
            pick = brng.integers(0, len(ush), len(ush))
            ix = np.concatenate([ix_by[ush[j]] for j in pick])
            if len(set(t[ix])) > 1:
                bs["rows"].append(stats.spearmanr(rows[ix], t[ix])[0])
                bs["d*L*s"].append(stats.spearmanr(dlt[ix], t[ix])[0])
        out["proving_time_spearman_cluster_bootstrap_95ci"] = {
            k_: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] for k_, v in bs.items()}
        out["proving_within_shape_pairs_rows_order_time"] = f"{int(agree)}/{tot}"
        rel = defaultdict(list)
        for v in by.values():
            if "linear" in v:
                for a_ in ("relu", "poly2", "gelu"):
                    rel[a_].append(v[a_]["prove_s"] / v["linear"]["prove_s"])
        out["proving_time_vs_linear"] = {a_: {"median": float(np.median(v)), "min": float(min(v)), "max": float(max(v))}
                                         for a_, v in rel.items()}
        for key in ("prove_s", "setup_s", "pk_bytes", "peak_rss_mb", "proof_bytes", "verify_s"):
            y = np.array([p[key] for p in P], float)
            out.setdefault("proving_vs_rows", {})[key] = {
                "spearman": float(stats.spearmanr(rows, y)[0]),
                "loglog_slope": float(stats.linregress(np.log(rows), np.log(y))[0]),
                "min": float(y.min()), "max": float(y.max())}
    # paired-by-seed quality differences at the selected checkpoint (shared seeds)
    def g(n):
        return {r["seed"]: r["val_ppl"] for r in _load(f"causal_study/{n}_seed*.json") if "seed" in r}
    G, Re, Po = g("384d6L_gelu"), g("384d6L_relu"), g("384d6L_poly2")
    if G and Re and Po:
        seeds = sorted(set(G) & set(Re) & set(Po))
        d1 = [G[x] - Re[x] for x in seeds]
        d2 = [Po[x] - Re[x] for x in seeds]
        out["paired_val_ppl"] = {"gelu_minus_relu": {"per_seed": d1, "mean": st.mean(d1), "sd": st.stdev(d1)},
                                 "poly2_minus_relu": {"per_seed": d2, "mean": st.mean(d2), "sd": st.stdev(d2)},
                                 "relu_lowest_in_all_seeds": all(x > 0 for x in d1 + d2),
                                 "sign_test_one_sided_p_3_of_3": 0.125}
    return out


def large_extrapolation() -> dict:
    """Fit every predictor on the 108-config grid, predict the out-of-range (width 256/512) configs."""
    import sys
    from scipy.optimize import nnls
    G = [r for r in _load("ezkl_validation/d*.json") if r["status"] == "ok"]
    Lg = [r for r in _load("ezkl_validation_large/d*.json") if r["status"] == "ok"]
    if not Lg:
        return {}
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from experiments.operator_attribution import compute_op_breakdown
    from proxy.features import TransformerConfig

    def comp(r):
        cfg = TransformerConfig(r["d_model"], 4, r["n_layers"], r["seq_len"], 4.0, r["activation"], False, True, 512, True)
        b = compute_op_breakdown(cfg)
        return {"matmul": b.attn_qkv_proj_constraints + b.attn_score_constraints + b.attn_value_constraints
                + b.attn_out_proj_constraints + b.ffn_up_constraints + b.ffn_down_constraints,
                "softmax": b.softmax_constraints, "activation": b.activation_constraints,
                "layernorm": b.layernorm_constraints, "embedding": b.embedding_constraints,
                "total": b.total_constraints}

    def arrs(R):
        d = np.array([r["d_model"] for r in R], float)
        L = np.array([r["n_layers"] for r in R], float)
        T = np.array([r["seq_len"] for r in R], float)
        acts = np.array([r["activation"] for r in R])
        A = np.c_[(acts == "gelu"), (acts == "relu"), (acts == "poly2")].astype(float)
        C = [comp(r) for r in R]
        m = np.array([r["num_rows"] for r in R], float)
        return d, L, T, A, C, m
    d, L, T, A, C, m = arrs(G)
    d2, L2, T2, A2, C2, m2 = arrs(Lg)
    ones, ones2 = np.ones(len(m)), np.ones(len(m2))
    a = np.array([c["total"] for c in C], float)
    a2 = np.array([c["total"] for c in C2], float)
    ly = np.log(m)
    designs = {
        "analytical (log-log fit)": (np.c_[ones, np.log(a), A], np.c_[ones2, np.log(a2), A2]),
        "d*L*s (log-log fit)": (np.c_[ones, np.log(d * L * T), A], np.c_[ones2, np.log(d2 * L2 * T2), A2]),
        "power law d^x L^y s^z": (np.c_[ones, np.log(d), np.log(L), np.log(T), A],
                                  np.c_[ones2, np.log(d2), np.log(L2), np.log(T2), A2]),
    }
    out = {"n_large": len(Lg), "predictions": {}}
    for k, (X, X2) in designs.items():
        c = np.linalg.lstsq(X, ly, rcond=None)[0]
        pred = np.exp(X2 @ c)
        out["predictions"][k] = {"mape_pct": float(np.mean(np.abs(pred / m2 - 1)) * 100),
                                 "median_ratio_pred_over_measured": float(np.median(pred / m2)),
                                 "spearman": float(stats.spearmanr(pred, m2)[0])}
    keys = ["matmul", "softmax", "activation", "layernorm", "embedding"]
    F = np.array([[c[k] for k in keys] for c in C], float)
    F2 = np.array([[c[k] for k in keys] for c in C2], float)
    sc = F.max(0)
    w, _ = nnls(F / sc, m)
    pred = (F2 / sc) @ w
    out["predictions"]["additive NNLS"] = {"mape_pct": float(np.mean(np.abs(pred / m2 - 1)) * 100),
                                           "median_ratio_pred_over_measured": float(np.median(pred / m2)),
                                           "spearman": float(stats.spearmanr(pred, m2)[0])}
    out["analytical_raw_ratio_measured_over_modelled"] = {
        "median": float(np.median(m2 / a2)), "min": float((m2 / a2).min()), "max": float((m2 / a2).max())}
    # activation overhead at large width
    by = defaultdict(dict)
    for r in Lg:
        by[(r["d_model"], r["n_layers"], r["seq_len"])][r["activation"]] = r["num_rows"]
    out["rows_vs_linear_by_shape"] = {f"{k[0]}/{k[1]}/{k[2]}": {a_: round(v[a_] / v["linear"], 3) for a_ in v if a_ != "linear"}
                                      for k, v in by.items() if "linear" in v}
    out["logrows"] = {r["name"]: r["logrows"] for r in Lg}
    return out


def proving_repeats() -> dict:
    """Combine the original proofs (run 1) with the repeat runs: mean/sd per config, ordering counts."""
    import itertools
    runs = {1: _load("ezkl_proving/d*.json"), 2: _load("ezkl_proving_run2/d*.json"), 3: _load("ezkl_proving_run3/d*.json")}
    runs = {k: [r for r in v if r.get("status") == "ok" and r.get("verified")] for k, v in runs.items() if v}
    if len(runs) < 2:
        return {}
    cfg = defaultdict(list)
    for k, v in runs.items():
        for r in v:
            cfg[r["name"]].append(r)
    out = {"n_runs_per_config": {n: len(v) for n, v in cfg.items()}, "per_config": {}}
    cvs = []
    for n, v in cfg.items():
        t = [r["prove_s"] for r in v]
        rec = {"n": len(t), "mean_prove_s": float(np.mean(t)), "min": float(min(t)), "max": float(max(t)),
               "logrows": v[0]["logrows"], "num_rows": v[0]["num_rows"],
               "pk_gb": float(np.mean([r["pk_bytes"] for r in v]) / 1e9),
               "peak_rss_gb": float(np.mean([r["peak_rss_mb"] for r in v]) / 1024 * 1.048576)}
        if len(t) > 1:
            rec["cv_pct"] = float(np.std(t, ddof=1) / np.mean(t) * 100)
            cvs.append(rec["cv_pct"])
        out["per_config"][n] = rec
    out["cv_pct_median"] = float(np.median(cvs)) if cvs else None
    out["cv_pct_max"] = float(max(cvs)) if cvs else None
    by = defaultdict(dict)
    for n, rec in out["per_config"].items():
        r0 = cfg[n][0]
        by[(r0["d_model"], r0["n_layers"], r0["seq_len"])][r0["activation"]] = rec
    rel = defaultdict(list)
    wins = {"relu_faster_than_poly2": 0, "relu_slower_than_poly2": 0, "ties_within_2pct": 0}
    gelu_slower = {"relu": 0, "poly2": 0, "total_shapes": 0}
    for sh, v in by.items():
        if "linear" in v:
            for a_ in ("relu", "poly2", "gelu"):
                if a_ in v:
                    rel[a_].append(v[a_]["mean_prove_s"] / v["linear"]["mean_prove_s"])
        if "relu" in v and "poly2" in v:
            q = v["relu"]["mean_prove_s"] / v["poly2"]["mean_prove_s"]
            if abs(q - 1) <= 0.02:
                wins["ties_within_2pct"] += 1
            elif q < 1:
                wins["relu_faster_than_poly2"] += 1
            else:
                wins["relu_slower_than_poly2"] += 1
        if "gelu" in v and "relu" in v and "poly2" in v:
            gelu_slower["total_shapes"] += 1
            gelu_slower["relu"] += v["gelu"]["mean_prove_s"] > v["relu"]["mean_prove_s"]
            gelu_slower["poly2"] += v["gelu"]["mean_prove_s"] > v["poly2"]["mean_prove_s"]
    out["mean_time_vs_linear"] = {a_: {"median": float(np.median(v)), "min": float(min(v)), "max": float(max(v))}
                                  for a_, v in rel.items()}
    out["relu_vs_poly2_by_shape"] = wins
    out["gelu_slower_than"] = gelu_slower
    out["table_mean_prove_s"] = {f"{sh[0]}/{sh[1]}/{sh[2]}": {a_: (round(v[a_]["mean_prove_s"], 1), v[a_]["logrows"], v[a_]["n"])
                                                              for a_ in v} for sh, v in sorted(by.items())}
    # rows vs mean proving time
    names = list(out["per_config"])
    rows = np.array([out["per_config"][n]["num_rows"] for n in names], float)
    tm = np.array([out["per_config"][n]["mean_prove_s"] for n in names])
    out["spearman_rows_vs_mean_prove_s"] = float(stats.spearmanr(rows, tm)[0])
    shp = [(cfg[n][0]["d_model"], cfg[n][0]["n_layers"], cfg[n][0]["seq_len"]) for n in names]
    dlt = np.array([a_ * b_ * c_ for a_, b_, c_ in shp], float)
    out["spearman_dLs_vs_mean_prove_s"] = float(stats.spearmanr(dlt, tm)[0])
    ush = sorted(set(shp))
    ixs = {u: [i for i, x in enumerate(shp) if x == u] for u in ush}
    brng = np.random.default_rng(2)
    bs = []
    for _ in range(2000):
        pick = brng.integers(0, len(ush), len(ush))
        ix = np.concatenate([ixs[ush[j]] for j in pick])
        if len(set(tm[ix])) > 1 and len(set(rows[ix])) > 1:
            bs.append(stats.spearmanr(rows[ix], tm[ix])[0])
    out["spearman_rows_vs_mean_prove_s_cluster_ci95"] = [float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))]
    agree = tot = 0
    for u in ush:
        for i, j in itertools.combinations(ixs[u], 2):
            tot += 1
            agree += (rows[i] - rows[j]) * (tm[i] - tm[j]) > 0
    out["within_shape_pairs_rows_order_mean_time"] = f"{int(agree)}/{tot}"
    ratios = []
    for n, v in cfg.items():
        if len(v) == 2:
            ratios.append(v[1]["prove_s"] / v[0]["prove_s"])
    out["run2_over_run1_prove_time"] = {"min": float(min(ratios)), "median": float(np.median(ratios)), "max": float(max(ratios))}
    return out


if __name__ == "__main__":
    summary = {"ezkl_validation": ezkl_summary(), "lm": lm_summary()}
    if (ROOT / "ezkl_validation_scale11").exists():
        summary["ezkl_validation_scale11"] = ezkl_summary("ezkl_validation_scale11")
    pr = _load("ezkl_proving/d*.json")
    if pr:
        summary["ezkl_proving"] = pr
    summary["extra_stats"] = extra_stats()
    summary["proving_repeats"] = proving_repeats()
    if (ROOT / "ezkl_validation_large").exists():
        summary["ezkl_validation_large"] = _load("ezkl_validation_large/d*.json")
        summary["large_extrapolation"] = large_extrapolation()
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
