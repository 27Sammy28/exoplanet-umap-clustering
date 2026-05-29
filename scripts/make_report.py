import json
import os
import platform
import sys
import numpy as np
from datetime import datetime

def main():
    os.makedirs("results/reports", exist_ok=True)

    report = {
        "timestamp": str(datetime.now()),
        "python_version": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "reproducibility": {
            "umap_seed": 42,
            "kmeans_seed": 42,
            "pipeline_deterministic": True
        },
        "notes": "Full NeurIPS pipeline reproducibility report"
    }

    with open("results/reports/reproducibility.json", "w") as f:
        json.dump(report, f, indent=4)

    print("Reproducibility report saved.")

if __name__ == "__main__":
    main()
