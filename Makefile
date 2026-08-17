.DEFAULT_GOAL := check
.PHONY: check test fast lint fmt types cov clean demo demo-portal demo-numbers demo-stage demo-check

## Everything the build gate runs.
check: lint types test

## Full test suite, including anything marked slow.
test:
	uv run pytest

## The inner-loop suite — must stay under 90 seconds.
fast:
	uv run pytest -m "not slow"

lint:
	uv run ruff check .
	uv run ruff format --check .

## Apply formatting and safe autofixes.
fmt:
	uv run ruff check --fix .
	uv run ruff format .

types:
	uv run mypy

cov:
	uv run pytest --cov=src --cov-report=term-missing --cov-report=html

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache .hypothesis htmlcov .coverage
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

## Stage the machine, serve the portal, and open it.
demo:
	uv run python demo/drive.py

## The portal alone, against whatever state is already there.
demo-portal:
	uv run python -m demo.portal

## The demonstration alone, in a temporary directory. Prints measured numbers.
demo-numbers:
	uv run python demo/e2e.py

## Get this machine ready to record, from a clean slate. Run between takes.
demo-stage:
	uv run python demo/stage.py

## Preflight only. The last thing to run before hitting record.
demo-check:
	uv run python demo/stage.py --check
