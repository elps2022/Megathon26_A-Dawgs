.PHONY: data calibrate train test

# Regenerate the synthetic dataset in data/generated/ from config/ (no raw data needed).
data:
	uv run python -m greenhome.data.generate --seed 42

# Rebuild config/ calibration tables from data/raw/ (needs the raw downloads).
calibrate:
	uv run python -m greenhome.data.calibrate

test:
	uv run pytest

# Energy model first cut on NEED: baselines vs LightGBM, 5-fold CV + val (test split untouched).
train:
	uv run python -m greenhome.models.energy_ml
