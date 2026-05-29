import matplotlib.pyplot as plt
import numpy as np

from src.preprocess import load_data, PHL_FEATURES
from src.embedding import run_umap
from src.baselines import run_pca, run_tsne
from src.physics_umap import run_physics_umap
from src.clustering import run_kmeans
from src.evaluation import evaluate_clusters, physics_consistency_score


# -------------------------
# Load data
# -------------------------
X = load_data()

methods = {
    "UMAP": run_umap(X),
    "PCA": run_pca(X),
    "t-SNE": run_tsne(X),
    "Physics-UMAP": run_physics_umap(X, PHL_FEATURES)
}

results = []

# -------------------------
# Evaluate
# -------------------------
for name, emb in methods.items():
    labels = run_kmeans(emb)
    metrics = evaluate_clusters(emb, labels)
    phys = physics_consistency_score(X, emb, PHL_FEATURES)

    results.append({
        "name": name,
        "silhouette": metrics["silhouette"],
        "physics": phys
    })

# -------------------------
# Extract arrays
# -------------------------
names = [r["name"] for r in results]
S = np.array([r["silhouette"] for r in results])
P = np.array([r["physics"] for r in results])

# -------------------------
# Pareto frontier (maximize both)
# -------------------------
def is_dominated(i):
    return any(
        (S[j] >= S[i] and P[j] >= P[i]) and (S[j] > S[i] or P[j] > P[i])
        for j in range(len(S)) if j != i
    )

frontier = [i for i in range(len(S)) if not is_dominated(i)]

# -------------------------
# Plot (NeurIPS style)
# -------------------------
plt.figure(figsize=(6.5, 5.2))

# background points
plt.scatter(S, P, s=90, alpha=0.6)

# labels
for i, name in enumerate(names):
    plt.text(S[i] + 0.002, P[i] + 0.002, name, fontsize=10)

# highlight Pareto frontier
for i in frontier:
    plt.scatter(S[i], P[i], s=180, edgecolors="black", linewidths=2)

# connect frontier (sorted by silhouette)
frontier_sorted = sorted(frontier, key=lambda i: S[i])
plt.plot(S[frontier_sorted], P[frontier_sorted], linestyle="--", linewidth=1)

# styling (NeurIPS clean look)
plt.xlabel("Clustering Quality (Silhouette)", fontsize=11)
plt.ylabel("Physics Consistency", fontsize=11)
plt.title("Pareto Trade-off: Structure vs Physics Constraints", fontsize=12)

plt.grid(True, alpha=0.2)
plt.tight_layout()

plt.savefig("results/figures/pareto_frontier.png", dpi=300, bbox_inches="tight")

print("Saved: results/figures/pareto_frontier.png")
