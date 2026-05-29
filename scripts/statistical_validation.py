import numpy as np
from scipy.stats import ttest_rel

from src.preprocess import load_data, PHL_FEATURES
from src.embedding import run_umap
from src.physics_umap import run_physics_umap
from src.clustering import run_kmeans
from src.evaluation import evaluate_clusters, physics_consistency_score


# -------------------------
# Settings
# -------------------------
N_RUNS = 10
X = load_data()


def run_method(method_fn, is_physics=False):
    sil_scores = []
    phys_scores = []

    for seed in range(N_RUNS):

        np.random.seed(seed)

        if is_physics:
            emb = method_fn(X, PHL_FEATURES)
        else:
            emb = method_fn(X)

        labels = run_kmeans(emb)

        metrics = evaluate_clusters(emb, labels)
        sil_scores.append(metrics["silhouette"])

        phys_scores.append(physics_consistency_score(X, emb, PHL_FEATURES))

    return np.array(sil_scores), np.array(phys_scores)


# -------------------------
# Run experiments
# -------------------------
print("Running UMAP...")
umap_sil, umap_phys = run_method(run_umap)

print("Running Physics-UMAP...")
phys_sil, phys_phys = run_method(run_physics_umap, is_physics=True)


# -------------------------
# Statistics
# -------------------------
def summarize(name, sil, phys):
    print(f"\n{name}")
    print(f"Silhouette: {sil.mean():.4f} ± {sil.std():.4f}")
    print(f"Physics:    {phys.mean():.4f} ± {phys.std():.4f}")


summarize("UMAP", umap_sil, umap_phys)
summarize("Physics-UMAP", phys_sil, phys_phys)


# -------------------------
# Significance tests
# -------------------------
sil_t, sil_p = ttest_rel(umap_sil, phys_sil)
phys_t, phys_p = ttest_rel(umap_phys, phys_phys)

print("\nStatistical Tests:")
print(f"Silhouette difference p-value: {sil_p:.6f}")
print(f"Physics consistency p-value:   {phys_p:.6f}")
