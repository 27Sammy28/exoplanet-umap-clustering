import matplotlib.pyplot as plt
import umap

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from src.preprocess import load_data
from src.clustering import run_kmeans

X = load_data()

methods = {
    "UMAP": umap.UMAP(n_components=2, random_state=42).fit_transform(X),
    "PCA": PCA(n_components=2).fit_transform(X),
    "t-SNE": TSNE(n_components=2, random_state=42).fit_transform(X),
}

for name, embedding in methods.items():

    labels = run_kmeans(embedding)

    plt.figure(figsize=(8,6))

    plt.scatter(
        embedding[:,0],
        embedding[:,1],
        c=labels,
        s=10
    )

    plt.title(f"{name} Embedding of Exoplanet Dataset")

    plt.xlabel("Component 1")
    plt.ylabel("Component 2")

    plt.tight_layout()

    plt.savefig(f"results/figures/{name.lower()}_embedding.png")

    print(f"Saved: {name}")
