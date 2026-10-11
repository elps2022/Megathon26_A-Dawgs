.PHONY: data calibrate build epc real sales value labels features train eval-test test eval app api

# Regenerate the synthetic dataset in data/generated/ from config/ (no raw data needed).
data:
	@if command -v uv >/dev/null 2>&1; then uv run python -m greenhome.data.generate --seed 42 && uv run python -m greenhome.data.features --interim; else PYTHONPATH="src:." .venv/bin/python -m greenhome.data.generate --seed 42 && PYTHONPATH="src:." .venv/bin/python -m greenhome.data.features --interim; fi

# Rebuild config/ calibration tables from data/raw/ (needs the raw downloads).
calibrate:
	@if command -v uv >/dev/null 2>&1; then uv run python -m greenhome.data.calibrate; else PYTHONPATH="src:." .venv/bin/python -m greenhome.data.calibrate; fi

test:
	@if command -v uv >/dev/null 2>&1; then uv run pytest; else PYTHONPATH="src:." PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/pytest; fi

# Run comprehensive 4-tier evaluation matrix (models, grounding shield, invariants).
eval:
	@PYTHONPATH="src:." .venv/bin/python scripts/run_eval.py

app:
	@PYTHONPATH="src:." .venv/bin/streamlit run app/main.py

api:
	@PYTHONPATH="src:." .venv/bin/uvicorn api.main:app --reload --port 8000

# Energy models on NEED: baselines, V2, V1, mode 1 (5-fold CV + val, test untouched), then save the shipped
# V1/V2 artifacts to src/greenhome/models/artifacts/ (needs data/raw/need and make labels).
train:
	@PYTHONPATH="src:." .venv/bin/python -m greenhome.models.energy_ml
	@PYTHONPATH="src:." .venv/bin/python -m greenhome.models.energy_greybox
	@PYTHONPATH="src:." .venv/bin/python -m greenhome.models.energy_bills
	@PYTHONPATH="src:." .venv/bin/python -m greenhome.models.energy

# The single scoring of the NEED test split (after make train; every model choice must be final first).
eval-test:
	@PYTHONPATH="src:." .venv/bin/python eval/eval_energy.py

# Real property book from EPC certificates (needs data/raw/epc, onspd, imd). Writes data/processed/*.parquet.
epc:
	@PYTHONPATH="src:." .venv/bin/python -m greenhome.data.epc_ingest

# Synthetic parts for the real EPC homes (bills, customer_id, smart controls, LTV). Run after `make epc`.
real:
	@PYTHONPATH="src:." .venv/bin/python -m greenhome.data.synthetic_parts --seed 42

# Price Paid sales matched to the real EPC homes, with pre-sale EPC features. Run after `make epc`.
sales:
	@PYTHONPATH="src:." .venv/bin/python -m greenhome.data.sales_join

# current_value_estimate for the real homes (last sale x UK HPI) + loan balances. Run after `make real` and `make sales`.
value:
	@PYTHONPATH="src:." .venv/bin/python -m greenhome.data.valuation

# The whole real data layer in one go: epc -> real -> sales -> value -> labels -> features (~2.5 min, needs data/raw/).
build:
	@PYTHONPATH="src:." .venv/bin/python -m greenhome.data.build --seed 42

# Observed per-home retrofit before/after labels from NEED, with the fixed NEED split (held-out savings validation).
labels:
	@PYTHONPATH="src:." .venv/bin/python -m greenhome.data.need_labels

# Stage 2 features for the real homes (parity model inputs, weather-normalised use, intensities). After `make real`.
features:
	@PYTHONPATH="src:." .venv/bin/python -m greenhome.data.features
