# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Hackathon entry for Megathon X, Lloyds Problem Statement 2: a Green Home Score and Retrofit Recommendation Engine. Three-person team. `docs/ARCHITECTURE.md` is the source of truth for design, interfaces (section 4) and who owns what (section 6). Read it before making non-trivial changes. If you change an interface, update that file in the same change.

## Hackathon timeline (IST)

- Mid submission 1: Sun 11 Oct 2026, 22:30
- Mid submission 2: Mon 12 Oct 2026, 04:30
- Final submission: Mon 12 Oct 2026, 11:30

Scope must fit these deadlines. Prefer cutting a feature over missing a submission. Keep something runnable at each deadline.

## Data sources (public-first, minimal)

Real: EPC domestic certificates (3 local authorities), NEED anonymised 50k sample, Land Registry Price Paid, UK HPI, Met Office monthly regional temperature (for heating degree days). Synthetic only where no public data exists: monthly bills per EPC home (NEED-calibrated), `customer_id`, smart-controls level. Raw downloads live in `data/raw/` and are never committed (EPC/PPD address fields are under Royal Mail terms).

## Git rules

- Never commit, amend, or push without the user's explicit approval for that specific commit. Stage and show the diff, then ask.

## Status and commands

Early scaffolding: only the data layer exists (`src/greenhome/data/`), and `tests/` is empty. The target module layout is in `docs/ARCHITECTURE.md` section 5. Everything flows into one `PropertyReport` (section 4), built by `build_report(property, retrofits=None, ...)`. The dashboard, API, portfolio view and LLM all consume that report and never recompute it.

The environment is managed with `uv` (`.venv/`, package installed editable via hatchling):

- Setup: `uv sync` (the dev group adds pytest)
- Synthetic data: `make data` → `data/generated/*.csv` (reads only `config/`, seed 42, ~6 s). File schemas and caveats are in `docs/DATA.md`. Never use `data/generated/_hidden/` as features.
- Calibration tables: `make calibrate` rebuilds `config/need_calibration.yaml`, `degree_days.csv`, `price_calibration.yaml` and `hpi_index.csv` from `data/raw/`. Only this step reads raw data. Its outputs are aggregates and are committed.
- All tests: `make test` / `uv run pytest`. Single test: `uv run pytest tests/test_x.py::test_name`

FastAPI, Streamlit and the LLM SDK are planned but not yet in `pyproject.toml`; add them there when you first need them. `make train`, `make eval` and `make app` don't exist yet.

Raw data quirks (`data/raw/`): the PPD CSVs (`pp-YYYY-filtered.csv`) have no header row, so use the standard Land Registry column order. NEED column metadata is `need/metadata.ods`, read with pandas plus `odfpy`. Met Office files are regional `Tmean_*.txt` series. EPC certificates have not been downloaded yet.

## Architecture invariants (graded by judges, do not break)

- **The LLM never computes numbers.** Score, savings, carbon and value come from `src/greenhome/` code. The LLM only receives tool output (`get_property_report`, `run_scenario`, memory tools) and explains it. A what-if must rerun `build_report(property, retrofits=[...])` before the LLM answers. A grounding checker verifies every number in an LLM answer appears in tool JSON.
- **Per-fuel everywhere.** Track electricity and gas separately end to end. Carbon = Σ ΔE_f × EF_f. An increase in use is a negative saving (heat pumps raise electricity).
- **Purchased energy = grid import + gas.** Solar reduces grid import, not underlying demand.
- **Score weights are fixed** (0.25, 0.20, 0.15, 0.15, 0.10, 0.10, 0.05) and every sub-score mapping must be monotonic. Mappings live in `config/score_mappings.yaml`, not hard-coded.
- **Retrofit interactions:** the savings simulator applies retrofits in priority order (fabric, controls, heating, generation). Each acts on the demand left after the previous one. Per-retrofit savings are marginal in that order and sum to the package total.
- **Missing data is never 0.** Annualise monthly energy with seasonal weights. Report "unavailable" plus a flag when coverage is too low.
- **No leakage.** Split train/val/test by `customer_id`. Never use post-retrofit or post-sale fields as prediction-time features. Keep `current_value_estimate` separate from historical sale `price`.
- **Every constant has a source.** Catalogue impacts, costs, emission factors (DESNZ) and synthetic-data rules carry a `source` field in `config/`.
- **Property value is an indicative range,** predicted before vs after for the same property and date. Never present it as causal.
