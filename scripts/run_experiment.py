import json
import os

os.makedirs("results/metrics", exist_ok=True)

with open("results/metrics/umap_results.json", "w") as f:
    json.dump(metrics, f, indent=4)
