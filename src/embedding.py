"""Embedding and nearest-neighbour evidence utilities for PAGER.

This module intentionally avoids heavy dependencies. The PCA helper uses NumPy SVD,
and the repeated KNN evidence function matches the dependency-light manuscript
reproducibility scripts.
"""
from __future__ import annotations

import numpy as np


def fit_standardizer(x_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return column means and standard deviations with zero-variance protection."""

    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale[scale == 0] = 1.0
    return mean, scale


def standardize(x: np.ndarray, mean: np.ndarray | None = None, scale: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardize an array and return transformed data plus fitted parameters."""

    if mean is None or scale is None:
        mean, scale = fit_standardizer(x)
    return (x - mean) / scale, mean, scale


def pca_embedding(x: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Compute a lightweight PCA embedding using singular value decomposition."""

    x_scaled, _, _ = standardize(np.asarray(x, dtype=float))
    _, _, vt = np.linalg.svd(x_scaled, full_matrices=False)
    return x_scaled @ vt[:n_components].T


def neighbour_vote_scores(
    x_train_all: np.ndarray,
    y_train_all: np.ndarray,
    x_query_all: np.ndarray,
    train_indices: np.ndarray | None = None,
    query_indices: np.ndarray | None = None,
    seed: int = 0,
    k: int = 7,
    leave_self_out: bool = False,
) -> tuple[np.ndarray, list[set[int]]]:
    """Return repeated KNN PHL-positive vote scores and neighbour index sets."""

    rng = np.random.default_rng(seed)
    train_indices = np.arange(len(x_train_all)) if train_indices is None else train_indices
    train_mask = rng.random(len(x_train_all)) < 0.8
    positives = y_train_all[train_mask].sum()
    negatives = train_mask.sum() - positives
    if positives < k or negatives < k:
        train_mask = np.ones(len(x_train_all), dtype=bool)

    weights = rng.normal(1.0, 0.015, size=x_train_all.shape[1])
    x_train_raw = x_train_all[train_mask]
    y_train = y_train_all[train_mask]
    active_indices = train_indices[train_mask]
    mean, scale = fit_standardizer(x_train_raw)
    x_train = ((x_train_raw - mean) / scale) * weights
    x_query = ((x_query_all - mean) / scale) * weights

    scores = np.zeros(len(x_query), dtype=float)
    neighbourhoods: list[set[int]] = []
    kth = min(k, len(x_train) - 1)
    for row in range(len(x_query)):
        diff = x_train - x_query[row]
        distances = np.einsum("ij,ij->i", diff, diff)
        if leave_self_out and query_indices is not None:
            distances[active_indices == query_indices[row]] = np.inf
        neighbours = np.argpartition(distances, kth)[:k]
        scores[row] = y_train[neighbours].mean()
        neighbourhoods.append(set(active_indices[neighbours].tolist()))
    return scores, neighbourhoods


def repeated_neighbour_evidence(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_query: np.ndarray,
    seeds: list[int] | range = range(20),
    k: int = 7,
    train_indices: np.ndarray | None = None,
    query_indices: np.ndarray | None = None,
    leave_self_out: bool = False,
) -> tuple[np.ndarray, list[list[set[int]]]]:
    """Run neighbour evidence over many random seeds."""

    runs = []
    neighbour_runs = []
    for seed in seeds:
        scores, neighbours = neighbour_vote_scores(
            x_train,
            y_train,
            x_query,
            train_indices=train_indices,
            query_indices=query_indices,
            seed=seed,
            k=k,
            leave_self_out=leave_self_out,
        )
        runs.append(scores)
        neighbour_runs.append(neighbours)
    return np.vstack(runs), neighbour_runs

import numpy as np
import umap
from sklearn.metrics import silhouette_score, davies_bouldin_score


def run_umap(X, seed=0, n_neighbors=15, min_dist=0.1, return_embedding=False):
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        random_state=seed
    )

    embedding = reducer.fit_transform(X)

    # fake clustering proxy (replace later if you have labels)
    labels = np.random.randint(0, 3, size=len(X))

    metrics = {
        "silhouette": float(silhouette_score(embedding, labels)),
        "davies_bouldin": float(davies_bouldin_score(embedding, labels))
    }

    if return_embedding:
        metrics["embedding"] = embedding

    return metrics
