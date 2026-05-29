from sklearn.metrics import silhouette_score, davies_bouldin_score
import numpy as np

def evaluate_clusters(X, labels):
    """
    Compute standard clustering evaluation metrics.

    Returns a dictionary suitable for paper reporting.
    """

    results = {}

    # Safety check: avoid crashes on single-cluster cases
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return {
            "silhouette": -1,
            "davies_bouldin": np.inf
        }

    results["silhouette"] = float(silhouette_score(X, labels))
    results["davies_bouldin"] = float(davies_bouldin_score(X, labels))

    return results
import numpy as np


def physics_consistency_score(X, embedding, feature_names, k=10):
    """
    Measures whether neighbors in embedding space
    are physically consistent in feature space.

    Higher = better physics preservation.
    """

    n = embedding.shape[0]

    # pairwise distances in embedding space
    emb_dist = np.linalg.norm(
        embedding[:, None, :] - embedding[None, :, :],
        axis=-1
    )

    # physical distances in feature space
    feat_dist = np.linalg.norm(
        X[:, None, :] - X[None, :, :],
        axis=-1
    )

    consistency_scores = []

    for i in range(n):

        # nearest neighbors in embedding space
        nn_idx = np.argsort(emb_dist[i])[:k+1][1:]

        # compute correlation between embedding-neighborhood and physics similarity
        emb_sim = emb_dist[i, nn_idx]
        phys_sim = feat_dist[i, nn_idx]

        # rank correlation proxy (simple, stable)
        score = np.corrcoef(emb_sim, phys_sim)[0, 1]

        if np.isnan(score):
            continue

        consistency_scores.append(score)

    return float(np.nanmean(consistency_scores))
