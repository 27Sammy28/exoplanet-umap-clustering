from src.preprocess import load_data
from src.embedding import run_umap
from src.baselines import run_pca, run_tsne
from src.clustering import run_kmeans
from src.evaluation import evaluate_clusters

def evaluate_pipeline(X, embedding, name):
    labels = run_kmeans(embedding)
    metrics = evaluate_clusters(embedding, labels)
    print(f"\n{name}:", metrics)
    return metrics

def main():
    X = load_data()

    results = {}

    results["UMAP"] = evaluate_pipeline(X, run_umap(X), "UMAP")
    results["PCA"] = evaluate_pipeline(X, run_pca(X), "PCA")
    results["t-SNE"] = evaluate_pipeline(X, run_tsne(X), "t-SNE")

if __name__ == "__main__":
    main()
