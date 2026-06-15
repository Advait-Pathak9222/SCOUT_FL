.PHONY: test microbenchmark synthetic aircomp fl-synthetic-small fl-quick clean

PY = python3

test:
	$(PY) -m pytest scout_fl/tests -q

microbenchmark:
	$(PY) -m scout_fl.experiments.run_microbenchmark --config scout_fl/configs/microbenchmark.yaml

synthetic:
	$(PY) -m scout_fl.experiments.run_synthetic --config scout_fl/configs/synthetic_small.yaml

aircomp:
	$(PY) -m scout_fl.experiments.run_aircomp --config scout_fl/configs/synthetic_small.yaml

fl-synthetic-small:
	$(PY) -m scout_fl.experiments.run_fl_synthetic --config scout_fl/configs/fl_synthetic_small.yaml

fl-quick:
	$(PY) -m scout_fl.experiments.run_fl_synthetic --config scout_fl/configs/fl_synthetic_small.yaml --quick

clean:
	rm -rf outputs/*/ scout_fl/**/__pycache__ scout_fl/__pycache__
