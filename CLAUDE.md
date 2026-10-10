# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Hackathon entry for Megathon X, Lloyds Problem Statement 2: a Green Home Score and Retrofit Recommendation Engine. Three-person team. `docs/ARCHITECTURE.md` is the source of truth for design, interfaces (section 4) and who owns what (section 6). Read it before making non-trivial changes. If you change an interface, update that file in the same change.

## Git rules

- Never commit, amend, or push without the user's explicit approval for that specific commit. Stage and show the diff, then ask.

## Status

Repo is at the scaffolding stage: no code, build, or test commands exist yet. Planned stack: Python 3.11, pandas, LightGBM, scikit-learn, Pydantic, FastAPI, Streamlit dashboard. Planned entry points are `make data`, `make train`, `make eval`, `make app`. Update this section with real commands once they exist.

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
