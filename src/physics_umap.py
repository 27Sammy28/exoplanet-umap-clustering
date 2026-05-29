"""Evaluation, PVP, and PAGER scoring utilities."""
from __future__ import annotations

import numpy as np
import pandas as pd

TAU = 0.25
LAMBDA = 1.0
GAMMA = 1.0
ETA = 1.0


def physics_penalty(raw: pd.DataFrame, tau: float = TAU) -> np.ndarray:
    """Compute the flux-based PVP-style soft physics penalty."""

    flux = pd.to_numeric(raw.get("P_FLUX"), errors="coerce").to_numpy(float)
    if {"S_HZ_CON_MIN", "S_HZ_CON_MAX"}.issubset(raw.columns):
        inner_flux = 1.0 / (pd.to_numeric(raw["S_HZ_CON_MIN"], errors="coerce").to_numpy(float) ** 2)
        outer_flux = 1.0 / (pd.to_numeric(raw["S_HZ_CON_MAX"], errors="coerce").to_numpy(float) ** 2)
        margin = np.minimum((flux - outer_flux) / outer_flux, (inner_flux - flux) / inner_flux)
    else:
        margin = np.minimum((flux - 0.10) / 0.10, (2.00 - flux) / 2.00)
    margin = np.where(np.isfinite(margin), margin, -1.0)
    penalty = 1.0 / (1.0 + np.exp(margin / tau))
    if "P_HABZONE_CON" in raw:
        hz_con = pd.to_numeric(raw["P_HABZONE_CON"], errors="coerce").fillna(0).to_numpy(int)
        penalty = np.where(hz_con == 1, np.minimum(penalty, 0.10), penalty)
    return penalty


def entropy_penalty(probabilities: np.ndarray) -> np.ndarray:
    """Binary entropy uncertainty penalty in bits."""

    q = np.clip(probabilities, 1e-9, 1 - 1e-9)
    return -(q * np.log(q) + (1 - q) * np.log(1 - q)) / np.log(2)


def pager_score(
    p_runs: np.ndarray,
    raw_features: pd.DataFrame,
    completeness: np.ndarray,
    lambda_value: float = LAMBDA,
    gamma: float = GAMMA,
    eta: float = ETA,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute PAGER score and its evidence, physics, and stability components."""

    pbar = p_runs.mean(axis=0)
    pvar = p_runs.var(axis=0)
    v = physics_penalty(raw_features)
    u = entropy_penalty(pbar) + eta * pvar
    score = pbar * np.exp(-lambda_value * v) * np.exp(-gamma * u) * completeness
    return score, pbar, v, u


def rank_runs(
    p_runs: np.ndarray,
    raw_features: pd.DataFrame,
    completeness: np.ndarray,
    lambda_value: float = LAMBDA,
    audit_tie_break: bool = False,
    hard_audit: bool = False,
) -> np.ndarray:
    """Rank candidates for each repeated evidence run."""

    ranks = np.full_like(p_runs, np.nan, dtype=float)
    v = physics_penalty(raw_features)
    for run_index, p_seed in enumerate(p_runs):
        u_seed = entropy_penalty(p_seed)
        score = p_seed * np.exp(-lambda_value * v) * np.exp(-GAMMA * u_seed) * completeness
        if hard_audit:
            score = np.where(v <= 0.5, score, 0.0)
        order = np.lexsort((v, -score)) if audit_tie_break else np.argsort(-score)
        ranks[run_index, order] = np.arange(1, len(order) + 1)
    return ranks


def topk_overlap(ranks: np.ndarray, consensus_order: np.ndarray, k: int) -> tuple[float, float]:
    """Mean and sample SD of overlap between repeated-run and consensus top-k."""

    consensus = set(consensus_order[:k].tolist())
    overlaps = []
    for run_rank in ranks:
        run_top = set(np.argsort(run_rank)[:k].tolist())
        overlaps.append(len(consensus & run_top) / k)
    return float(np.mean(overlaps)), float(np.std(overlaps, ddof=1))


def mean_neighbour_jaccard(neighbour_runs: list[list[set[int]]], top_indices: np.ndarray) -> float:
    """Mean pairwise Jaccard overlap of neighbour sets for selected candidates."""

    values = []
    for idx in top_indices:
        sets = [run[idx] for run in neighbour_runs]
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                union = sets[i] | sets[j]
                values.append(len(sets[i] & sets[j]) / len(union) if union else 1.0)
    return float(np.mean(values)) if values else float("nan")


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Compute precision among the top-k scored records."""

    order = np.argsort(-scores)[:k]
    return float(np.asarray(y_true)[order].mean()) if k else float("nan")




import umap
from src.physics_distance import physics_distance_matrix


def run_physics_umap(X, feature_names, lambda_phys=0.5):

    print("Computing physics-aware distance matrix...")

    D = physics_distance_matrix(X, feature_names, lambda_phys)

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        metric="precomputed",
        random_state=42
    )

    embedding = reducer.fit_transform(D)

    return embedding
