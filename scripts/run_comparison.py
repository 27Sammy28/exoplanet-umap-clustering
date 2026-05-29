import json
import os
import numpy as np

from src.preprocess import load_data
from src.embedding import run_umap
from src.baselines import run_pca, run_tsne
from src.clustering import run_kmeans
from src.evaluation import evaluate_clusters

os.makedirs("results/metrics", exist_ok=True)

def save_results(name, metrics):
    path = f"results/metrics/{name}_results.json"
    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)

def run_method(name, embedding, X):
    labels = run_kmeans(embedding)
    metrics = evaluate_clusters(embedding, labels)
    print(f"{name}:", metrics)
    save_results(name, metrics)

def main():
    print("Loading data...")
    X = load_data()

    print("\nRunning UMAP...")
    umap_emb = run_umap(X)
    run_method("umap", umap_emb, X)

    print("\nRunning PCA...")
    pca_emb = run_pca(X)
    run_method("pca", pca_emb, X)

    print("\nRunning t-SNE...")
    tsne_emb = run_tsne(X)
    run_method("tsne", tsne_emb, X)

    print("\nDONE: All experiments completed.")

if __name__ == "__main__":
    main()
