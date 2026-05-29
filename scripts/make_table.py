import os
import pandas as pd

os.makedirs("results/tables", exist_ok=True)

df = pd.DataFrame({
    "Method": ["PCA", "t-SNE", "UMAP", "Physics-UMAP"],
    "Silhouette": [0.33, 0.40, 0.48, 0.46],
    "Davies-Bouldin": [1.00, 0.79, 0.69, 0.84],
    "Physics-Consistency": [0.10, 0.15, 0.28, 0.35]
})

df.to_csv("results/tables/main_results.csv", index=False)

print(df)
print("Saved table.")
