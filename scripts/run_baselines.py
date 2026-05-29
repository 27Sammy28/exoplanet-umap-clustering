from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def run_pca(X, n_components=2):
    return PCA(n_components=n_components, random_state=42).fit_transform(X)

def run_tsne(X, n_components=2):
    return TSNE(
        n_components=n_components,
        perplexity=30,
        random_state=42
    ).fit_transform(X)
