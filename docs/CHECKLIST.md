# Final submission checklist

| Item | Deliverable | Owner | Status |
|---|---|---|---|
| A | Working web app / dashboard | B | [ ] |
| B | Code + `make data`, `make train`, `make eval`, `make app`; run instructions | All | [~] README (setup / run / data links / credits) and MIT LICENSE written; links to fill after upload |
| C | Trained models in `models/artifacts/` + inference code | A | [x] `models/energy.py`, `models/value_model.py`, `engine/detector.py`; weights hosted externally (`make package-models`, link in README: **upload pending**) |
| D | `docs/SCORING.md` | C | [ ] |
| E | `docs/LLM.md` (prompts, model, how tested) | B | [ ] |
| F | Architecture diagram | B | [ ] |
| G | `docs/EVALUATION.md` (validation vs baselines, limitations) | A + B | [~] A's part done: energy, retrofit savings, value, detector (test set scored once); LLM + recs pending (B) |
| H | `docs/RESPONSIBLE_AI.md` | C | [ ] |
| I | Held-out test predictions + data documentation (`docs/DATA.md`) | A | [x] `eval/results/test_predictions.csv`, `docs/DATA.md`; processed data hosted externally (`make package-data`, link in README: **upload pending**) |

## Golden rules to re-check before submitting
- [ ] 1. Every number comes from code, the AI only explains
- [ ] 2. Electricity and gas shown separately everywhere
- [ ] 3. Every config number has a `source`
- [ ] 4. Savings are "estimates", value is an "indicative range", missing data says "unavailable", never 0
- [ ] 5. No leakage: splits by customer, no future information
