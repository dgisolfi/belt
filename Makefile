install:
	python -m pip install -e ".[dev]"

lint:
	python -m black belt
	python -m isort belt
	python -m ruff check belt tests

clean:
	rm -rf .pytest_cache .ruff_cache build dist outputs *.egg-info

.PHONY: install lint clean 