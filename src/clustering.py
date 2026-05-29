"""Small clusterin
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
    ""Compute majority-label purity for diagnostic clusters."""

   
from sklearn.cluster import KMeans

def run_kmeans(X, n_clusters=3):
    """
    Run KMeans clustering on embedding space.
    Returns cluster labels.
    """
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = model.fit_predict(X)
    return labels
