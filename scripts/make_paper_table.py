import pandas as pd

from src.preprocess import load_data, PHL_FEATURES
from src.embedding import run_umap
from src.baselines import run_pca, run_tsne
from src.physics_umap import run_physics_umap
from src.clustering import run_kmeans
from src.evaluation import evaluate_clusters
from src.evaluation import evaluate_clusters, physics_consistency_score

X = load_data()

methods = {
    "UMAP": run_umap(X),
    "PCA": run_pca(X),
    "t-SNE": run_tsne(X),
    "Physics-UMAP": run_physics_umap(X, PHL_FEATURES)
}

rows = []

for name, emb in methods.items():

    labels = run_kmeans(emb)
    metrics = evaluate_clusters(emb, labels)

phys_score = physics_consistency_score(X, emb, PHL_FEATURES)

rows.append({
    "Method": name,
    "Silhouette": metrics["silhouette"],
    "Davies-Bouldin": metrics["davies_bouldin"],
    "Physics-Consistency": phys_score
})



    
df = pd.DataFrame(rows)

df = df.sort_values("Silhouette", ascending=False)

print(df)

df.to_csv("results/paper_table.csv", index=False)

with open("results/paper_table.tex", "w") as f:
    f.write(df.to_latex(index=False, float_format="%.4f"))

print("Saved table to results/")
