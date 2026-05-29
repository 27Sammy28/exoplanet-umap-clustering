import os

os.makedirs("submission_package/tables", exist_ok=True)

df.to_latex("submission_package/tables/results_table.tex", index=False)
