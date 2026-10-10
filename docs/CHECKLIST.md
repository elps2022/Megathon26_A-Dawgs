# Final submission checklist

| Item | Deliverable | Owner | Status |
|---|---|---|---|
| A | Working web app / dashboard | B | [ ] |
| B | Code + `make data`, `make train`, `make eval`, `make app`; run instructions | All | [ ] |
| C | Trained models in `models/artifacts/` + inference code | A | [ ] |
| D | `docs/SCORING.md` | C | [ ] |
| E | `docs/LLM.md` (prompts, model, how tested) | B | [ ] |
| F | Architecture diagram | B | [ ] |
| G | `docs/EVALUATION.md` (validation vs baselines, limitations) | A + B | [ ] |
| H | `docs/RESPONSIBLE_AI.md` | C | [ ] |
| I | Held-out test predictions + data documentation (`docs/DATA.md`) | A | [ ] |

## Golden rules to re-check before submitting
- [ ] 1. Every number comes from code, the AI only explains
- [ ] 2. Electricity and gas shown separately everywhere
- [ ] 3. Every config number has a `source`
- [ ] 4. Savings are "estimates", value is an "indicative range", missing data says "unavailable", never 0
- [ ] 5. No leakage: splits by customer, no future information
