.PHONY: data calibrate train test

# Regenerate the synthetic dataset in data/generated/ from config/ (no raw data needed).
data:
	uv run python -m greenhome.data.generate --seed 42

# Rebuild config/ calibration tables from data/raw/ (needs the raw downloads).
calibrate:
	uv run python -m greenhome.data.calibrate

test:
	@if command -v uv >/dev/null 2>&1; then uv run pytest; else PYTHONPATH="src:." PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/pytest; fi

app:
	@PYTHONPATH="src:." .venv/bin/streamlit run app/main.py

api:
	@PYTHONPATH="src:." .venv/bin/uvicorn api.main:app --reload --port 8000

# Energy model first cut on NEED: baselines vs LightGBM, 5-fold CV + val (test split untouched).
train:
	@PYTHONPATH="src:." .venv/bin/python -m greenhome.models.energy_ml

