import json
import pandas as pd
from pathlib import Path

metrics_dir = Path("results/metrics")

results = []

for file in metrics_dir.glob("*_results.json"):

    with open(file) as f:
        data = json.load(f)

    method = file.stem.replace("_results", "").upper()

    results.append({
        "Method": method,
        "Silhouette Score": round(data["silhouette"], 4),
        "Davies-Bouldin Index": round(data["davies_bouldin"], 4)
    })

df = pd.DataFrame(results)

df = df.sort_values(
    by=["Silhouette Score"],
    ascending=False
)

print("\n=== Comparison Table ===\n")
print(df.to_string(index=False))

df.to_csv("results/comparison_table.csv", index=False)

latex = df.to_latex(index=False)

with open("results/comparison_table.tex", "w") as f:
    f.write(latex)

print("\nSaved:")
print("results/comparison_table.csv")
print("results/comparison_table.tex")
