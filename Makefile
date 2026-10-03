.PHONY: install test generate baselines clean

install:
	pip install -e ".[dev]"

test:
	pytest -q

# Build a fresh, deterministic task set (public + private dirs under ./tasks)
generate:
	python -m failurelab.cli generate --n 36 --seed 0 --out tasks

# Run the reference baselines and print the scorecard
baselines:
	python -m failurelab.cli baselines --tasks tasks --out runs

clean:
	rm -rf tasks runs reports
