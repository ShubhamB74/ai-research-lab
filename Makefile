.PHONY: install test generate baselines selftest analyze clean

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

# Scripted model through the full LLM harness (sandbox, tool loop, transcripts); needs no API key
selftest:
	python -m failurelab.cli selftest --tasks tasks --policy careful --max-tasks 12

# Pre-registered analysis; usage: make analyze RUNS="runs_llm/modelA runs_llm/modelB"
analyze:
	python -m failurelab.cli analyze --tasks tasks --runs $(RUNS) --out report

clean:
	rm -rf tasks runs reports runs_selftest runs_smoke runs_llm
