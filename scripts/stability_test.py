import numpy as np
import pandas as pd
import umap

from src.preprocess import load_data
from src.clustering import run_kmeans
from src.evaluation import evaluate_clusters

X = load_data()

results = []

for seed in range(10):

    reducer = umap.UMAP(
        n_components=2,
        random_state=seed
    )

    embedding = reducer.fit_transform(X)

    labels = run_kmeans(embedding)

    metrics = evaluate_clusters(embedding, labels)

    results.append(metrics)

df = pd.DataFrame(results)

print("\n=== UMAP Stability Results ===\n")
print(df)

print("\nMean Metrics:\n")
print(df.mean())

df.to_csv("results/umap_stability.csv", index=False)
