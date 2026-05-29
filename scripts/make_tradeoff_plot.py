import matplotlib.pyplot as plt

from src.preprocess import load_data, PHL_FEATURES
from src.embedding import run_umap
from src.physics_umap import run_physics_umap
from src.baselines import run_pca, run_tsne
from src.clustering import run_kmeans
from src.evaluation import evaluate_clusters, physics_consistency_score


print("Loading data...")
X = load_data()

methods = {}

print("Running embeddings...")

methods["UMAP"] = run_umap(X)
methods["PCA"] = run_pca(X)
methods["t-SNE"] = run_tsne(X)
methods["Physics-UMAP"] = run_physics_umap(X, PHL_FEATURES)

results = []

print("Evaluating...")

for name, emb in methods.items():

    labels = run_kmeans(emb)
    metrics = evaluate_clusters(emb, labels)
    phys = physics_consistency_score(X, emb, PHL_FEATURES)

    results.append({
        "method": name,
        "silhouette": metrics["silhouette"],
        "physics": phys
    })

# -------------------------
# PLOT TRADE-OFF
# -------------------------

plt.figure(figsize=(7, 5))

for r in results:
    plt.scatter(
        r["silhouette"],
        r["physics"],
        s=120
    )
    plt.text(
        r["silhouette"] + 0.002,
        r["physics"] + 0.002,
        r["method"]
    )

plt.xlabel("Clustering Quality (Silhouette Score)")
plt.ylabel("Physics Consistency Score")
plt.title("Trade-off: Structure vs Physics Constraint")

plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("results/figures/tradeoff_plot.png", dpi=300)

print("Saved: results/figures/tradeoff_plot.png")
