# Final submission checklist

| Item | Deliverable | Owner | Status |
|---|---|---|---|
| A | Working web app / dashboard | B | [x] `app/main.py` complete (Customer View, Bank Portfolio, Grounding Inspector) |
| B | Code + `make data`, `make train`, `make eval`, `make app`; run instructions | All | [x] README, Makefile, and dependencies fully wired and verified |
| C | Trained models in `models/artifacts/` + inference code | A | [x] 15 model weights hosted externally, links in README, `make models` verified |
| D | `docs/SCORING.md` | C | [x] Sub-score mappings and weight justifications complete |
| E | `docs/LLM.md` (prompts, model, how tested) | B | [x] Gemini 2.5 Flash + Grounding Shield architecture documented |
| F | Architecture diagram | B | [x] Mermaid diagrams and `docs/Architecture_Overview.pdf` |
| G | `docs/EVALUATION.md` (validation vs baselines, limitations) | A + B | [x] Complete bake-offs (energy, savings, value, detector, LLM) |
| H | `docs/RESPONSIBLE_AI.md` | C | [x] Fairness audit across IMD quintiles, housing types, and regions complete |
| I | Held-out test predictions + data documentation (`docs/DATA.md`) | A | [x] `eval/results/test_predictions.csv`, `docs/DATA.md`; `make real-data` verified |

## Golden rules to re-check before submitting
- [x] 1. Every number comes from code, the AI only explains
- [x] 2. Electricity and gas shown separately everywhere
- [x] 3. Every config number has a `source`
- [x] 4. Savings are "estimates", value is an "indicative range", missing data says "unavailable", never 0
- [x] 5. No leakage: splits by customer, no future information

