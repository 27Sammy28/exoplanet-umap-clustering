import os
import numpy as np
import matplotlib.pyplot as plt

from src.preprocess import load_data
from src.embedding import pca_embedding, run_umap

plt.rcParams.update({
    "figure.dpi": 300,
    "font.size": 11
})

def main():
    os.makedirs("results/figures", exist_ok=True)

    X = load_data()

    X_pca = pca_embedding(X, 2)
    X_umap = run_umap(X, seed=0, return_embedding=True)["embedding"]

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    ax[0].scatter(X_pca[:, 0], X_pca[:, 1], s=4)
    ax[0].set_title("PCA")

    ax[1].scatter(X_umap[:, 0], X_umap[:, 1], s=4)
    ax[1].set_title("UMAP")

    plt.tight_layout()
    plt.savefig("results/figures/neurips_main.pdf")
    plt.savefig("results/figures/neurips_main.png")

    print("Saved NeurIPS figure.")

if __name__ == "__main__":
    main()
