"""Embedding-artifact evals T-EMB-1..4 (EVAL.md section 5, T-EMB table).

Every runner takes an embeddings matrix ``emb`` of shape ``(n_nodes, d)``
aligned with the graph's node order plus per-node label arrays derived
from ``nodes.parquet``. Pre-registered thresholds (``eval_v1``):

- T-EMB-1  cell-type recovery: kNN (k=10) leave-one-out top-1 accuracy
  ``>= 2x`` the majority-baseline rate AND ``>=`` a degree-matched
  spectral baseline on the same graph ``+ 0.05``. The baseline is a
  normalized-adjacency spectral embedding (``scipy.sparse.linalg.eigsh``,
  deterministic starting vector) of the SAME embedding dimension -
  "degree-matched" is read as matched degrees of freedom under an
  identical kNN protocol on the identical graph.
- T-EMB-2  NT-class linear probe (7 NT classes): test accuracy
  ``>= majority rate + 0.15``. Probe = multinomial logistic regression
  implemented in numpy, deterministic stratified 70/30 split.
- T-EMB-3  dimorphism probe (isomorphic | dimorphic | male-specific):
  balanced accuracy ``>= 0.50`` (chance ~0.33).
- T-EMB-4  NBLAST agreement: mean fraction of each type's top-10
  embedding neighbors inside the type's NBLAST top-50 ``>= 3x`` the
  chance overlap. Uses ``evals/nblast_neighbors.parquet`` (columns
  ``query_id, match_id, score``) when present; otherwise falls back to a
  connectivity-neighborhood proxy (shared undirected 2-hop type
  neighborhoods computed from the bundle) and the artifact records
  ``nblast_source = "connectivity_2hop_proxy"``.

Unlabeled rows (empty type) are excluded from every metric - never
silently scored. Missing optional inputs produce ``pass: null`` with a
recorded reason, never a silent pass.

Usage:
    python -m flygpt.evals.emb_evals --embeddings emb.npz --graph-dir data/graph
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from flygpt.schema import GraphBundle, rng_from_seed, write_result

K_KNN = 10
MAJORITY_FACTOR = 2.0
SPECTRAL_MARGIN = 0.05
NT_MAJORITY_MARGIN = 0.15
BALANCED_ACC_MIN = 0.50
TOP_EMB = 10
TOP_REF = 50
CHANCE_FACTOR = 3.0

_UNTYPED = ""


def _emit(
    results_dir: Path | None,
    test_id: str,
    checks: dict[str, Any],
    *,
    inputs: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    passed: bool | None = None,
    extra: dict[str, Any] | None = None,
) -> dict:
    """Write ``results/<test_id>.json`` when a dir is given; else just
    return the envelope-shaped dict (tests inject artifacts directly)."""
    if results_dir is not None:
        return write_result(
            results_dir, test_id, checks,
            inputs=inputs, metrics=metrics, passed=passed, extra=extra,
        )
    art = {
        "artifact": test_id,
        "inputs": inputs or {},
        "checks": checks,
        "metrics": metrics or {},
        "pass": passed,
    }
    if extra:
        art.update(extra)
    return art


# ---------------------------------------------------------------------------
# Small deterministic numerics (numpy/scipy only)
# ---------------------------------------------------------------------------


def _str_labels(labels: Any) -> np.ndarray:
    """Label column -> object array with None/NaN mapped to ""."""
    arr = np.asarray(labels, dtype=object).ravel()
    return np.array(["" if v is None else str(v) for v in arr], dtype=object)


def _cosine_sim(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.where(n == 0.0, 1.0, n)
    Xn = X / n
    return Xn @ Xn.T


def _knn_loo_accuracy(X: np.ndarray, labels: np.ndarray, k: int = K_KNN) -> dict:
    """kNN leave-one-out top-1 accuracy with deterministic tie-breaks.

    Cosine similarity; neighbors sorted by descending similarity with a
    stable sort (index order breaks ties). A tied vote is won by the label
    whose highest-similarity neighbor is closest.
    """
    labels = _str_labels(labels)
    n = X.shape[0]
    if n < 2:
        return {"acc": None, "n": int(n), "reason": "fewer than 2 labeled rows"}
    kk = int(min(k, n - 1))
    sim = _cosine_sim(X)
    np.fill_diagonal(sim, -np.inf)
    order = np.argsort(-sim, axis=1, kind="stable")[:, :kk]
    correct = 0
    for i in range(n):
        nbr = labels[order[i]]
        uniq, counts = np.unique(nbr, return_counts=True)
        tied = uniq[counts == counts.max()]
        pred = (
            tied[0]
            if tied.size == 1
            else min(tied, key=lambda lab: int(np.flatnonzero(nbr == lab)[0]))
        )
        correct += int(pred == labels[i])
    return {"acc": correct / n, "n": int(n), "k": kk}


def _majority_label_rate(labels: np.ndarray) -> float:
    labels = _str_labels(labels)
    labels = labels[labels != _UNTYPED]
    if labels.size == 0:
        return 0.0
    _, counts = np.unique(labels, return_counts=True)
    return float(counts.max() / labels.size)


def _spectral_embedding(bundle: GraphBundle, d: int) -> np.ndarray:
    """Normalized-adjacency spectral embedding of the bundle graph.

    Symmetrized weighted adjacency ``A + A.T``, normalized
    ``D^-1/2 A D^-1/2``, top-``d`` eigenvectors. Deterministic: ``eigsh``
    gets a hash-derived starting vector; tiny graphs fall back to dense
    ``eigh``. Zero-padded to ``d`` columns when fewer are available.
    """
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    n = bundle.n_nodes
    pre = np.repeat(np.arange(n), np.diff(bundle.out_indptr))
    a = sp.csr_matrix(
        (
            bundle.out_weight.astype(np.float64),
            (pre, bundle.out_indices.astype(np.int64)),
        ),
        shape=(n, n),
    )
    a = (a + a.T).tocsr()
    deg = np.asarray(a.sum(axis=1)).ravel()
    dinv = np.where(deg > 0.0, 1.0 / np.sqrt(np.where(deg > 0.0, deg, 1.0)), 0.0)
    norm = sp.diags(dinv)
    s = norm @ a @ norm

    k = min(d, n - 2)
    vecs = np.zeros((n, 0))
    if k >= 1:
        if n > 4 and k < n - 1:
            v0 = rng_from_seed("T-EMB-1", "spectral_v0", n).standard_normal(n)
            try:
                vals, vecs = spla.eigsh(s, k=k, which="LA", v0=v0)
                vecs = vecs[:, np.argsort(-vals)]
            except Exception:
                vecs = np.zeros((n, 0))
        if vecs.shape[1] < k:
            vals, vecs = np.linalg.eigh(s.toarray())
            vecs = vecs[:, np.argsort(-vals)[:k]]
    if vecs.shape[1] < d:
        vecs = np.hstack([vecs, np.zeros((n, d - vecs.shape[1]))])
    return vecs[:, :d]


# ---------------------------------------------------------------------------
# Linear probe (T-EMB-2 / T-EMB-3): multinomial logistic regression
# ---------------------------------------------------------------------------


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _standardize(x_tr: np.ndarray, x_te: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = x_tr.mean(axis=0)
    sd = x_tr.std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (x_tr - mu) / sd, (x_te - mu) / sd


def _fit_logistic(
    x: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    *,
    l2: float = 1e-4,
    lr: float = 0.3,
    iters: int = 2000,
) -> tuple[np.ndarray, np.ndarray]:
    """Full-batch softmax regression, deterministic (fixed hyperparams)."""
    n, d = x.shape
    w = np.zeros((d, n_classes))
    b = np.zeros(n_classes)
    onehot = np.zeros((n, n_classes))
    onehot[np.arange(n), y] = 1.0
    for _ in range(iters):
        p = _softmax(x @ w + b)
        g = (p - onehot) / n
        w -= lr * (x.T @ g + l2 * w)
        b -= lr * g.sum(axis=0)
    return w, b


def _stratified_split(
    labels: np.ndarray, frac: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Per-class seeded split; classes with <2 members are dropped."""
    train: list[int] = []
    test: list[int] = []
    dropped: list[str] = []
    for lab in np.unique(labels):
        idx = np.flatnonzero(labels == lab)
        idx = idx[rng.permutation(idx.size)]
        if idx.size < 2:
            dropped.append(str(lab))
            continue
        n_tr = int(round(frac * idx.size))
        n_tr = min(max(n_tr, 1), idx.size - 1)
        train.extend(idx[:n_tr].tolist())
        test.extend(idx[n_tr:].tolist())
    return np.asarray(train), np.asarray(test), dropped


def _balanced_accuracy_idx(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    recalls = [
        float((y_pred[y_true == c] == c).mean())
        for c in range(n_classes)
        if int((y_true == c).sum()) > 0
    ]
    return float(np.mean(recalls)) if recalls else 0.0


def _balanced_accuracy(y_true: Any, y_pred: Any) -> float:
    """Mean per-class recall over arbitrary label values."""
    yt = _str_labels(y_true)
    yp = _str_labels(y_pred)
    classes = np.unique(np.concatenate([yt, yp]))
    cmap = {c: i for i, c in enumerate(classes)}
    yi = np.array([cmap[v] for v in yt])
    pi = np.array([cmap[v] for v in yp])
    return _balanced_accuracy_idx(yi, pi, classes.size)


def _run_probe(
    emb: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int,
    train_frac: float = 0.7,
) -> dict:
    """Stratified-split multinomial probe. Returns metrics or a reason."""
    keep = labels != _UNTYPED
    x = emb[keep]
    y = labels[keep]
    classes = np.unique(y)
    if classes.size < 2:
        return {"reason": f"fewer than 2 classes with labels ({classes.size})"}
    rng = rng_from_seed("emb_probe", int(seed), float(train_frac), tuple(classes.tolist()))
    tr, te, dropped = _stratified_split(y, train_frac, rng)
    if tr.size == 0 or te.size == 0:
        return {"reason": "stratified split produced an empty side"}
    cmap = {c: i for i, c in enumerate(classes)}
    ytr = np.array([cmap[v] for v in y[tr]])
    yte = np.array([cmap[v] for v in y[te]])
    x_tr, x_te = _standardize(x[tr], x[te])
    w, b = _fit_logistic(x_tr, ytr, classes.size)
    pred = (x_te @ w + b).argmax(axis=1)
    return {
        "acc": float((pred == yte).mean()),
        "balanced_acc": _balanced_accuracy_idx(yte, pred, classes.size),
        "majority_rate": float(np.bincount(yte, minlength=classes.size).max() / yte.size),
        "n_train": int(tr.size),
        "n_test": int(te.size),
        "classes": [str(c) for c in classes],
        "dropped_classes": dropped,
        "y_test_idx": yte,
        "y_pred_idx": pred,
    }


# ---------------------------------------------------------------------------
# T-EMB-4 helpers: type-level embeddings + connectivity proxy
# ---------------------------------------------------------------------------


def _type_mean_embeddings(emb: np.ndarray, labels: np.ndarray) -> tuple[list[str], np.ndarray]:
    """Mean embedding per non-empty type; rows aligned with sorted types."""
    keep = labels != _UNTYPED
    types = sorted(set(labels[keep].tolist()))
    mat = np.stack([emb[labels == t].mean(axis=0) for t in types])
    return types, mat


def _top_neighbors(sim: np.ndarray, k: int) -> np.ndarray:
    """Per-row top-k indices by descending similarity (stable ties)."""
    s = sim.copy()
    np.fill_diagonal(s, -np.inf)
    return np.argsort(-s, axis=1, kind="stable")[:, :k]


def _twohop_type_profiles(bundle: GraphBundle, labels: np.ndarray) -> np.ndarray:
    """Per-node count profile over *types* reachable in <=2 undirected hops."""
    n = bundle.n_nodes
    adj = [
        np.union1d(bundle.out_neighbors(v), bundle.in_neighbors(v))
        for v in range(n)
    ]
    types = sorted(set(labels[labels != _UNTYPED].tolist()))
    tidx = {t: i for i, t in enumerate(types)}
    prof = np.zeros((n, len(types)))
    for v in range(n):
        if adj[v].size:
            seen = np.unique(np.concatenate([adj[v], *[adj[u] for u in adj[v]]]))
            seen = seen[seen != v]
        else:
            seen = adj[v]
        for u in seen:
            t = labels[u]
            if t != _UNTYPED:
                prof[v, tidx[t]] += 1.0
    # pool node profiles into type profiles
    tprof = np.zeros((len(types), len(types)))
    for v in range(n):
        tv = labels[v]
        if tv != _UNTYPED:
            tprof[tidx[tv]] += prof[v]
    return tprof


def _proxy_top_sets(
    bundle: GraphBundle, labels: np.ndarray, types: list[str], top_ref: int
) -> dict[str, set[str]]:
    """Connectivity proxy for NBLAST top-``top_ref`` per type.

    Two types are "morphologically near" in the proxy when their pooled
    undirected 2-hop type-neighborhood profiles have high cosine
    similarity. Fixed-size ranking: the zero-similarity tail still fills
    the top-``top_ref`` set (matching the top-50 convention).
    """
    tprof = _twohop_type_profiles(bundle, labels)
    sim = _cosine_sim(tprof)
    order = _top_neighbors(sim, min(top_ref, len(types) - 1))
    return {t: {types[j] for j in order[i] if j != i} for i, t in enumerate(types)}


def _nblast_top_sets(
    nblast_path: Path, types: list[str], top_ref: int
) -> tuple[dict[str, set[str]], int]:
    """NBLAST parquet -> top-``top_ref`` match sets per type."""
    df = pl.read_parquet(nblast_path)
    required = {"query_id", "match_id", "score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{nblast_path} missing columns {sorted(missing)}")
    df = df.with_columns(
        pl.col("query_id").cast(pl.String), pl.col("match_id").cast(pl.String)
    )
    sets: dict[str, set[str]] = {}
    n_present = 0
    for t in types:
        rows = (
            df.filter(pl.col("query_id") == t)
            .sort("score", descending=True)
            .head(top_ref)
        )
        s = set(rows["match_id"].to_list())
        sets[t] = s
        n_present += int(bool(s))
    return sets, n_present


def run_t4_nblast_agreement(
    emb: np.ndarray,
    labels: Any,
    *,
    nblast_path: Path | None = None,
    bundle: GraphBundle | None = None,
    top_emb: int = TOP_EMB,
    top_ref: int = TOP_REF,
    results_dir: Path | None = None,
) -> dict:
    """T-EMB-4: embedding neighbors vs NBLAST (or connectivity proxy)."""
    emb = np.asarray(emb, dtype=np.float64)
    labels = _str_labels(labels)
    types, mat = _type_mean_embeddings(emb, labels)

    reason = None
    if len(types) <= top_emb:
        reason = f"only {len(types)} typed classes - need > {top_emb} for top-{top_emb}"
    checks: dict[str, Any] = {}
    metrics: dict[str, Any] = {}
    artifact: dict[str, Any] | None = None

    if reason is None:
        emb_top = _top_neighbors(_cosine_sim(mat), min(top_emb, len(types) - 1))
        emb_sets = {t: {types[j] for j in emb_top[i] if j != i} for i, t in enumerate(types)}

        source = None
        ref_sets: dict[str, set[str]] = {}
        n_table_types = 0
        if nblast_path is not None and Path(nblast_path).exists():
            ref_sets, n_table_types = _nblast_top_sets(Path(nblast_path), types, top_ref)
            source = "nblast_parquet"
        elif bundle is not None:
            ref_sets = _proxy_top_sets(bundle, labels, types, top_ref)
            source = "connectivity_2hop_proxy"
        else:
            reason = "no NBLAST table and no graph bundle for the proxy"

        if source is not None:
            per_type = {}
            frac_sum = 0.0
            chance_sum = 0.0
            n_used = 0
            for t in types:
                s = ref_sets.get(t, set())
                if not s:
                    per_type[t] = {"hits": 0, "ref_size": 0, "skipped": True}
                    continue
                hits = len(emb_sets[t] & s)
                per_type[t] = {"hits": int(hits), "ref_size": int(len(s))}
                frac_sum += hits / len(emb_sets[t])
                chance_sum += len(s) / (len(types) - 1)
                n_used += 1
            if n_used == 0:
                reason = "no type has a non-empty reference neighbor set"
            else:
                observed = frac_sum / n_used
                chance = chance_sum / n_used
                ratio = observed / chance if chance > 0 else float("inf")
                metrics.update(
                    {
                        "observed_fraction": observed,
                        "chance_fraction": chance,
                        "agreement_ratio": ratio,
                        "n_types_scored": n_used,
                        "n_types_total": len(types),
                        "nblast_source": source,
                        "n_types_in_nblast_table": n_table_types,
                        "top_emb": top_emb,
                        "top_ref": top_ref,
                    }
                )
                checks["observed_ge_3x_chance"] = bool(
                    observed >= CHANCE_FACTOR * chance
                )
                checks["chance_nonzero"] = bool(chance > 0)
                artifact = {"per_type": per_type}

    if reason is not None:
        checks["status"] = f"NA: {reason}"
        passed = None
    else:
        passed = bool(all(v for v in checks.values()))

    inputs = {
        "n_rows": int(emb.shape[0]),
        "n_types": len(types),
        "nblast_path": str(nblast_path) if nblast_path else None,
        "proxy_used": metrics.get("nblast_source") == "connectivity_2hop_proxy",
    }
    return _emit(
        results_dir,
        "T-EMB-4",
        checks,
        inputs=inputs,
        metrics=metrics,
        passed=passed,
        extra=artifact,
    )


# ---------------------------------------------------------------------------
# T-EMB-1..3 runners
# ---------------------------------------------------------------------------


def run_t1_type_recovery(
    emb: np.ndarray,
    labels: Any,
    *,
    bundle: GraphBundle | None = None,
    k: int = K_KNN,
    results_dir: Path | None = None,
) -> dict:
    """T-EMB-1: kNN type recovery vs 2x majority and spectral baseline."""
    emb = np.asarray(emb, dtype=np.float64)
    labels = _str_labels(labels)
    keep = labels != _UNTYPED

    knn = _knn_loo_accuracy(emb[keep], labels[keep], k)
    maj = _majority_label_rate(labels)
    metrics: dict[str, Any] = {
        "knn_acc": knn["acc"],
        "majority_rate": maj,
        "k": knn.get("k", k),
        "n_eval": knn["n"],
        "n_excluded_untyped": int((~keep).sum()),
    }
    checks: dict[str, Any] = {}
    passed: bool | None = True

    if knn["acc"] is None:
        checks["status"] = f"NA: {knn['reason']}"
        passed = None
    else:
        checks["knn_acc_ge_2x_majority"] = bool(knn["acc"] >= MAJORITY_FACTOR * maj)
        if bundle is not None:
            spec = _spectral_embedding(bundle, emb.shape[1])
            sknn = _knn_loo_accuracy(spec[keep], labels[keep], k)
            metrics["spectral_acc"] = sknn["acc"]
            metrics["spectral_dim"] = emb.shape[1]
            if sknn["acc"] is None:
                checks["knn_acc_ge_spectral_plus_margin"] = "NA: spectral kNN undefined"
                passed = None
            else:
                checks["knn_acc_ge_spectral_plus_margin"] = bool(
                    knn["acc"] >= sknn["acc"] + SPECTRAL_MARGIN
                )
        else:
            checks["knn_acc_ge_spectral_plus_margin"] = (
                "NA: no graph bundle - spectral baseline not computed"
            )
            passed = None

        if passed is not None:
            passed = bool(all(v is True for v in checks.values()))

    return _emit(
        results_dir,
        "T-EMB-1",
        checks,
        inputs={"n_rows": int(emb.shape[0]), "dim": int(emb.shape[1])},
        metrics=metrics,
        passed=passed,
    )


def run_t2_nt_probe(
    emb: np.ndarray,
    nt_labels: Any,
    *,
    seed: int = 0,
    train_frac: float = 0.7,
    results_dir: Path | None = None,
) -> dict:
    """T-EMB-2: NT-class linear probe, accuracy >= majority + 0.15."""
    emb = np.asarray(emb, dtype=np.float64)
    labels = _str_labels(nt_labels)
    probe = _run_probe(emb, labels, seed=seed, train_frac=train_frac)

    checks: dict[str, Any] = {}
    metrics: dict[str, Any] = {}
    if "reason" in probe:
        checks["status"] = f"NA: {probe['reason']}"
        passed = None
    else:
        margin = probe["acc"] - probe["majority_rate"]
        metrics = {
            "probe_acc": probe["acc"],
            "majority_rate": probe["majority_rate"],
            "margin": margin,
            "n_train": probe["n_train"],
            "n_test": probe["n_test"],
            "n_classes": len(probe["classes"]),
            "classes": probe["classes"],
            "dropped_classes": probe["dropped_classes"],
        }
        checks["acc_ge_majority_plus_0.15"] = bool(margin >= NT_MAJORITY_MARGIN)
        passed = bool(all(checks.values()))

    return _emit(
        results_dir,
        "T-EMB-2",
        checks,
        inputs={"n_rows": int(emb.shape[0]), "train_frac": train_frac, "seed": seed},
        metrics=metrics,
        passed=passed,
    )


def run_t3_dimorphism_probe(
    emb: np.ndarray,
    dimorphism_labels: Any,
    *,
    seed: int = 0,
    train_frac: float = 0.7,
    results_dir: Path | None = None,
) -> dict:
    """T-EMB-3: iso/dimorphic/male-specific probe, balanced acc >= 0.50."""
    emb = np.asarray(emb, dtype=np.float64)
    labels = _str_labels(dimorphism_labels)
    probe = _run_probe(emb, labels, seed=seed, train_frac=train_frac)

    checks: dict[str, Any] = {}
    metrics: dict[str, Any] = {}
    if "reason" in probe:
        checks["status"] = f"NA: {probe['reason']}"
        passed = None
    else:
        metrics = {
            "balanced_acc": probe["balanced_acc"],
            "probe_acc": probe["acc"],
            "majority_rate": probe["majority_rate"],
            "n_train": probe["n_train"],
            "n_test": probe["n_test"],
            "classes": probe["classes"],
            "dropped_classes": probe["dropped_classes"],
        }
        checks["balanced_acc_ge_0.50"] = bool(probe["balanced_acc"] >= BALANCED_ACC_MIN)
        passed = bool(all(checks.values()))

    return _emit(
        results_dir,
        "T-EMB-3",
        checks,
        inputs={"n_rows": int(emb.shape[0]), "train_frac": train_frac, "seed": seed},
        metrics=metrics,
        passed=passed,
    )


# ---------------------------------------------------------------------------
# Artifact loading + CLI
# ---------------------------------------------------------------------------


def _load_embeddings(path: Path) -> dict[str, np.ndarray]:
    """Load an embeddings artifact.

    ``.npz``: must contain ``emb`` (n,d); optional label arrays ``type``,
    ``nt``, ``dimorphism``. ``.parquet``: either a list column ``emb`` or
    columns ``emb_0..emb_{d-1}``; remaining columns are label candidates.
    """
    if path.suffix == ".npz":
        z = np.load(path, allow_pickle=False)
        return {k: z[k] for k in z.files}
    if path.suffix == ".parquet":
        df = pl.read_parquet(path)
        out: dict[str, np.ndarray] = {}
        emb_cols = [c for c in df.columns if c.startswith("emb_")]
        if "emb" in df.columns:
            out["emb"] = np.stack(df["emb"].to_numpy())
        elif emb_cols:
            out["emb"] = df.select(sorted(emb_cols)).to_numpy()
        for c in df.columns:
            if c != "emb" and not c.startswith("emb_"):
                out[c] = df[c].to_numpy()
        return out
    raise ValueError(f"unsupported embeddings artifact: {path}")


# ---------------------------------------------------------------------------
# Dispatcher entry points (flygpt.evals.RUNNERS names). Required inputs
# arrive via kwargs; absent inputs record a documented NA - never a crash.
# ---------------------------------------------------------------------------


def _load_bundle_kw(bundle: Any, graph_dir: Any, cutoff: int) -> GraphBundle | None:
    if bundle is not None:
        return bundle
    if graph_dir is not None:
        from flygpt.schema import load_graph_bundle

        return load_graph_bundle(Path(graph_dir), cutoff=int(cutoff))
    return None


def run_t_emb_1(
    results_dir: Path,
    *,
    emb: Any = None,
    labels: Any = None,
    bundle: GraphBundle | None = None,
    graph_dir: Path | None = None,
    cutoff: int = 5,
    k: int = K_KNN,
    **_ignored: Any,
) -> dict:
    """Dispatcher entry for T-EMB-1 (see ``run_t1_type_recovery``)."""
    if emb is None or labels is None:
        return _emit(
            results_dir, "T-EMB-1",
            {"status": "NA: no embeddings matrix / type labels injected"},
            passed=None,
        )
    return run_t1_type_recovery(
        emb, labels,
        bundle=_load_bundle_kw(bundle, graph_dir, cutoff), k=k,
        results_dir=results_dir,
    )


def run_t_emb_2(
    results_dir: Path,
    *,
    emb: Any = None,
    nt_labels: Any = None,
    seed: int = 0,
    **_ignored: Any,
) -> dict:
    """Dispatcher entry for T-EMB-2 (see ``run_t2_nt_probe``)."""
    if emb is None or nt_labels is None:
        return _emit(
            results_dir, "T-EMB-2",
            {"status": "NA: no embeddings matrix / NT labels injected"},
            passed=None,
        )
    return run_t2_nt_probe(emb, nt_labels, seed=seed, results_dir=results_dir)


def run_t_emb_3(
    results_dir: Path,
    *,
    emb: Any = None,
    dimorphism_labels: Any = None,
    seed: int = 0,
    **_ignored: Any,
) -> dict:
    """Dispatcher entry for T-EMB-3 (see ``run_t3_dimorphism_probe``)."""
    if emb is None or dimorphism_labels is None:
        return _emit(
            results_dir, "T-EMB-3",
            {"status": "NA: no embeddings matrix / dimorphism labels injected"},
            passed=None,
        )
    return run_t3_dimorphism_probe(
        emb, dimorphism_labels, seed=seed, results_dir=results_dir
    )


def run_t_emb_4(
    results_dir: Path,
    *,
    emb: Any = None,
    labels: Any = None,
    bundle: GraphBundle | None = None,
    graph_dir: Path | None = None,
    cutoff: int = 5,
    nblast_path: Path | None = None,
    **_ignored: Any,
) -> dict:
    """Dispatcher entry for T-EMB-4 (see ``run_t4_nblast_agreement``)."""
    if emb is None or labels is None:
        return _emit(
            results_dir, "T-EMB-4",
            {"status": "NA: no embeddings matrix / type labels injected"},
            passed=None,
        )
    nb = nblast_path
    if nb is None:
        default_nb = Path("evals/nblast_neighbors.parquet")
        nb = default_nb if default_nb.exists() else None
    return run_t4_nblast_agreement(
        emb, labels,
        nblast_path=nb,
        bundle=_load_bundle_kw(bundle, graph_dir, cutoff),
        results_dir=results_dir,
    )


def run_all(
    data: dict[str, np.ndarray],
    *,
    bundle: GraphBundle | None = None,
    nblast_path: Path | None = None,
    results_dir: Path | None = None,
) -> dict[str, dict]:
    """Run every T-EMB-* eval for which the artifact carries labels."""
    emb = data["emb"]
    out: dict[str, dict] = {}
    if "type" in data:
        out["T-EMB-1"] = run_t1_type_recovery(
            emb, data["type"], bundle=bundle, results_dir=results_dir
        )
        out["T-EMB-4"] = run_t4_nblast_agreement(
            emb,
            data["type"],
            nblast_path=nblast_path,
            bundle=bundle,
            results_dir=results_dir,
        )
    if "nt" in data:
        out["T-EMB-2"] = run_t2_nt_probe(emb, data["nt"], results_dir=results_dir)
    if "dimorphism" in data:
        out["T-EMB-3"] = run_t3_dimorphism_probe(
            emb, data["dimorphism"], results_dir=results_dir
        )
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--embeddings", type=Path, required=True,
                    help=".npz (emb + type/nt/dimorphism arrays) or .parquet")
    ap.add_argument("--graph-dir", type=Path, default=None,
                    help="graph bundle dir for the spectral baseline + proxy")
    ap.add_argument("--cutoff", type=int, default=5)
    ap.add_argument("--nblast", type=Path, default=None,
                    help="default: evals/nblast_neighbors.parquet when present")
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    args = ap.parse_args(argv)

    data = _load_embeddings(args.embeddings)
    bundle = None
    if args.graph_dir is not None:
        from flygpt.schema import load_graph_bundle

        bundle = load_graph_bundle(args.graph_dir, cutoff=args.cutoff)
    nblast = args.nblast
    if nblast is None:
        default_nb = Path("evals/nblast_neighbors.parquet")
        nblast = default_nb if default_nb.exists() else None

    out = run_all(data, bundle=bundle, nblast_path=nblast, results_dir=args.results_dir)
    if not out:
        print("no label arrays found in the embeddings artifact - nothing run")
        return 1
    for test_id, art in out.items():
        print(f"{test_id}: pass={art['pass']} checks={art['checks']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
