.PHONY: data calibrate test

# Regenerate the synthetic dataset in data/generated/ from config/ (no raw data needed).
data:
	uv run python -m greenhome.data.generate --seed 42

# Rebuild config/ calibration tables from data/raw/ (needs the raw downloads).
calibrate:
	uv run python -m greenhome.data.calibrate

test:
	uv run pytest
