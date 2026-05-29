submit:
	PYTHONPATH=. python scripts/run_experiment.py
	PYTHONPATH=. python scripts/make_table.py
	PYTHONPATH=. python scripts/make_neurips_figures.py
	PYTHONPATH=. python scripts/make_report.py
	PYTHONPATH=. python scripts/build_submission.py
	PYTHONPATH=. python scripts/make_zip.py

paper:
	PYTHONPATH=. python scripts/build_submission.py
	PYTHONPATH=. python scripts/make_zip.py
