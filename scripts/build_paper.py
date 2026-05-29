import os
import shutil
import subprocess

def run(cmd):
    subprocess.run(cmd, shell=True, check=True)

def main():
    os.makedirs("submission_package/figures", exist_ok=True)
    os.makedirs("submission_package/tables", exist_ok=True)
    os.makedirs("submission_package/results", exist_ok=True)

    # Copy figures
    for f in os.listdir("results/figures"):
        shutil.copy(f"results/figures/{f}", "submission_package/figures/")

    # Copy tables
    for f in os.listdir("results/tables"):
        shutil.copy(f"results/tables/{f}", "submission_package/tables/")

    # Copy metrics
    if os.path.exists("results/metrics"):
        shutil.copytree("results/metrics", "submission_package/results", dirs_exist_ok=True)

    # Compile LaTeX
    run("pdflatex -output-directory=submission_package latex/main.tex")

    # Rename PDF
    os.rename(
        "submission_package/main.pdf",
        "submission_package/paper.pdf"
    )

    print("Paper built successfully.")

if __name__ == "__main__":
    main()
