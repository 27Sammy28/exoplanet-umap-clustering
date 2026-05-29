import matplotlib.pyplot as plt

from src.preprocess import load_data, PHL_FEATURES
from src.embedding import run_umap
from src.baselines import run_pca, run_tsne
from src.physics_umap import run_physics_umap
from src.clustering import run_kmeans


def plot(ax, emb, labels, title):
    ax.scatter(emb[:, 0], emb[:, 1], c=labels, s=8, cmap="viridis")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


print("Loading data...")
X = load_data()

print("Computing embeddings...")

emb_umap = run_umap(X)
emb_pca = run_pca(X)
emb_tsne = run_tsne(X)
emb_phys = run_physics_umap(X, PHL_FEATURES)

labels_umap = run_kmeans(emb_umap)
labels_pca = run_kmeans(emb_pca)
labels_tsne = run_kmeans(emb_tsne)
labels_phys = run_kmeans(emb_phys)

fig, axs = plt.subplots(2, 2, figsize=(10, 10))

plot(axs[0, 0], emb_umap, labels_umap, "UMAP")
plot(axs[0, 1], emb_pca, labels_pca, "PCA")
plot(axs[1, 0], emb_tsne, labels_tsne, "t-SNE")
plot(axs[1, 1], emb_phys, labels_phys, "Physics-UMAP")

plt.tight_layout()
plt.savefig("results/figures/paper_embedding_grid.png", dpi=300)

print("Saved: results/figures/paper_embedding_grid.png")
