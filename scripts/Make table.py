import json
import os
import pandas as pd

def load_metrics(path):
    with open(path, "r") as f:
        return json.load(f)

def main():
    os.makedirs("results/tables", exist_ok=True)

    # Example metric files (you can extend later)
    files = {
        "UMAP": "results/metrics/umap_results.json",
        "PCA": "results/metrics/pca_results.json",
        "t-SNE": "results/metrics/tsne_results.json"
    }

    rows = []

    for method, path in files.items():
        if not os.path.exists(path):
            continue

        metrics = load_metrics(path)
        metrics["method"] = method
        rows.append(metrics)

    df = pd.DataFrame(rows)

    print("\n=== PAPER TABLE ===\n")
    print(df)

    # Save CSV (for paper + appendix)
    df.to_csv("results/tables/comparison_table.csv", index=False)

    # Save LaTeX table (for paper)
    latex_table = df.to_latex(index=False)

    with open("results/tables/comparison_table.tex", "w") as f:
        f.write(latex_table)

    print("\nSaved:")
    print("- results/tables/comparison_table.csv")
    print("- results/tables/comparison_table.tex")

if __name__ == "__main__":
    main()
