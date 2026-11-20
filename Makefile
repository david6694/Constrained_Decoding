.PHONY: install run lint debug clean

install:
	uv sync

run:
	uv run python -m src

lint:
	flake8 .
	mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

debug:
	uv run python -m pdb -m src

clean:
	rm -rf .venv __pycache__ src/__pycache__ .mypy_cache
