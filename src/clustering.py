"""Small clustering helpers for catalogue-space diagnostics."""
from __future__ import annotations

import numpy as np


def kmeans(
    x: np.ndarray,
    n_clusters: int = 2,
    seed: int = 0,
    max_iter: int = 300,
    tol: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Run dependency-light k-means and return labels plus centroids."""

    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    if n_clusters < 1 or n_clusters > len(x):
        raise ValueError("n_clusters must be between 1 and the number of rows")
    centroids = x[rng.choice(len(x), size=n_clusters, replace=False)].copy()
    labels = np.zeros(len(x), dtype=int)

    for _ in range(max_iter):
        distances = ((x[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        new_labels = distances.argmin(axis=1)
        new_centroids = centroids.copy()
        for cluster in range(n_clusters):
            members = x[new_labels == cluster]
            if len(members):
                new_centroids[cluster] = members.mean(axis=0)
        shift = np.sqrt(((new_centroids - centroids) ** 2).sum())
        labels = new_labels
        centroids = new_centroids
        if shift < tol:
            break
    return labels, centroids


def cluster_purity(cluster_labels: np.ndarray, true_labels: np.ndarray) -> float:
    """Compute majority-label purity for diagnostic clusters."""

    cluster_labels = np.asarray(cluster_labels)
    true_labels = np.asarray(true_labels)
    total = 0
    for cluster in np.unique(cluster_labels):
        values = true_labels[cluster_labels == cluster]
        if values.size == 0:
            continue
        _, counts = np.unique(values, return_counts=True)
        total += counts.max()
    return float(total / len(true_labels)) if len(true_labels) else float("nan")


def nearest_centroid_labels(x: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Assign each row to the nearest supplied centroid."""

    distances = ((x[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    return distances.argmin(axis=1)
