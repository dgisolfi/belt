install:
	python -m pip install -e ".[dev]"

lint:
	python -m black belt
	python -m isort belt
	python -m ruff check belt --fix

train:
	python -m belt.supervised.softmax --config configs/supervised_softmax.yaml

clean:
	rm -rf .pytest_cache .ruff_cache build dist out *.egg-info data

.PHONY: install lint train clean 