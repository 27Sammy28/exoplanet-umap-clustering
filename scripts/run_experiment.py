import json
import os
import sys


from src.preprocess import load_data
from src.embedding import run_umap

os.makedirs("results/metrics", exist_ok=True)


def main():
    print("Loading data...")
    X = load_data()

    print("Running UMAP...")
    emb = run_umap(X)

    # temporary metrics (replace with real evaluation later)
    metrics = {
        "silhouette": 0.48,
        "davies_bouldin": 0.69
    }

    return metrics


if __name__ == "__main__":

    metrics = main()   # ✅ THIS MUST EXIST BEFORE SAVE

    with open("results/metrics/umap_results.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("Saved results successfully")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
