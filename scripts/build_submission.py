import os
import shutil
import subprocess
import pandas as pd


def run(cmd):
    subprocess.run(cmd, shell=True, check=True)


def main():
    print("📦 Building submission package...")

    # Create folders
    os.makedirs("submission_package/figures", exist_ok=True)
    os.makedirs("submission_package/tables", exist_ok=True)
    os.makedirs("submission_package/metrics", exist_ok=True)
    os.makedirs("submission_package/reports", exist_ok=True)

    # ---- BUILD TABLE AUTOMATICALLY ----
    csv_path = "results/paper_table.csv"       
    tex_path = "submission_package/tables/results_table.tex"

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df.to_latex(tex_path, index=False)

    elif os.path.exists("results/tables/results_table.tex"):
        shutil.copy(
            "results/tables/results_table.tex",
            tex_path
        )

    else:
        raise FileNotFoundError(
            "No table file found (CSV or TEX)"
        )

    # Copy other folders safely
    for folder in ["figures", "tables", "metrics", "logs"]:
        src = f"results/{folder}"
        dst = f"submission_package/{folder}"

        if os.path.exists(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)

    # Copy reproducibility report
    if os.path.exists("results/reports"):
        shutil.copytree(
            "results/reports",
            "submission_package/reports",
            dirs_exist_ok=True
        )

    # Compile LaTeX
    run(
        "pdflatex -output-directory=submission_package latex/main.tex"
    )

    # Rename output PDF
    if os.path.exists("submission_package/main.pdf"):
        os.rename(
            "submission_package/main.pdf",
            "submission_package/paper.pdf"
        )

    print("📄 Paper built successfully")


if __name__ == "__main__":
    main()
