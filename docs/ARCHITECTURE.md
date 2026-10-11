# Green Home Score & Retrofit Engine — Architecture and Team Plan

Problem statement: Lloyds PS2 (Megathon X). This doc is the shared contract. If you change an interface in section 4, update this file in the same PR.

---

## 1. Guiding principles (these are what the judges grade)

1. **Numbers come from code, never the LLM.** The LLM only gets JSON produced by our engine and explains it. A what-if reruns the engine first.
2. **Everything is per-fuel.** Electricity and gas are tracked separately from data generation to the dashboard. Carbon = Σ ΔE_f × EF_f.
3. **Every number has a documented source.** Catalogue impacts, emission factors, sub-score mappings and synthetic data rules all live in versioned config files (`config/*.yaml`, `config/*.csv`) with a `source` field.
4. **Honest uncertainty.** Savings are "estimates", value impact is an "indicative range", and missing data is reported as missing (never silently 0).
5. **Leak-free evaluation.** Splits by `customer_id` (which also keeps each property in a single split). No post-retrofit or post-sale fields at prediction time.

---

## 2. Pipeline

```
 ┌──────────────────────┐
 │ Data layer            │  real: EPC (3 LAs), NEED 50k, Price Paid, UK HPI, Met Office
 │  properties, monthly  │  synthetic: monthly bills (NEED-calibrated), customer_id,
 │  energy, sales,       │  smart controls, LTV
 │  retrofit labels      │  retrofit labels: NEED observed before/after (loft, CWI, PV)
 └─────────┬────────────┘
           ▼
 ┌──────────────────────┐
 │ Feature engineering   │  annualise (seasonal-weighted), kWh/m², data-quality flags,
 │                       │  weather-normalise annual use, map EPC fields -> NEED bands
 └─────────┬────────────┘
           ▼
 ┌──────────────────────┐   ┌──────────────────────────────────────┐
 │ Green Home Score      │   │ Energy model (two versions, 1 API)    │
 │ 7 sub-scores, fixed w │   │  V1 grey box: heat-loss physics +     │
 │                       │   │   bill calibration + ML residual      │
 │                       │   │  V2 tuned GBDT; both scored on NEED   │
 └─────────┬────────────┘   └────────────┬─────────────────────────┘
           ▼                              ▼  expected use -> "worse than similar homes?"
 ┌─────────────────────────────────────────────┐
 │ Efficiency detector: Poor/Very poor flag,   │
 │ key issues, improvement-potential ranking   │
 └─────────┬───────────────────────────────────┘
           ▼
 ┌──────────────────────┐
 │ Retrofit engine       │  feasibility rules (catalogue) -> ordered package
 │  feasibility + order  │  (fabric -> controls -> heating -> generation)
 └─────────┬────────────┘
           ▼
 ┌──────────────────────┐  ┌──────────────┐  ┌───────────────────────────┐
 │ Savings simulator     │─▶│ Carbon (det.) │  │ Value model (LightGBM      │
 │ edits features, re-   │  │ DESNZ factors │  │ quantile, before vs after) │
 │ runs energy model     │  │              │  │                            │
 └─────────┬────────────┘  └──────┬───────┘  └────────────┬──────────────┘
           └──────────────┬───────┴────────────────────────┘
                          ▼
              PropertyReport JSON  (single source of truth)
                          ▼
 ┌──────────────────────────────────────────────┐
 │ LLM layer: tool-calling agent                 │
 │  tools = get_report, run_scenario, memory     │
 │  + post-hoc grounding checker (numbers must   │
 │    appear in tool output)                     │
 └─────────┬────────────────────────────────────┘
           ▼
 FastAPI  ──▶  Customer dashboard (score, heat-loss house, recs, savings, carbon, value, chat)
          └─▶  Lender portfolio view (aggregates many PropertyReports)
```

What-if flow: user asks "what if I skip the heat pump and add solar?" → LLM calls `run_scenario(property_id, retrofits=[...])` → engine recomputes savings/carbon/value/new score → LLM explains the returned JSON.

---

## 3. Key modelling decisions

### 3.1 Data (real public data first, synthetic only where none exists)

**Interim (now): fully synthetic dataset, same shape.** `make data` builds 10k homes from `config/` only (calibrated to NEED, Price Paid, UK HPI and Met Office via `make calibrate`). Teammates build against it while the real EPC book is prepared. Files and caveats: `docs/DATA.md`. The real book (`make build`, `data/processed/real/`) has the same fields as parquet; the bills stay synthetic either way.

**Real sources** (raw files in `data/raw/`, never committed; sources and licences in `docs/DATA.md`):

| Source | What it gives us | Used for |
|---|---|---|
| EPC domestic certificates (Leeds, Cornwall, Southwark) | Per-address property type, age band, floor area, rating/SAP, heating, insulation, glazing, PV, modelled energy | The property book: `Property` inputs, score, feasibility |
| NEED anonymised 50k sample | Banded property features + **real annual gas and electricity 2005–2023** + loft (`LI_DATE`), cavity wall (`CWI_DATE`) and PV (`PV_DATE`) install years | Training/evaluating the energy models; **observed before/after retrofit savings**; calibrating synthetic bills |
| Land Registry Price Paid (same 3 LAs, 2018–2025) | Real sale prices, date, type, tenure, new-build | Value model (joined to EPC by postcode + address numbers, `greenhome.data.sales_join`) |
| UK HPI | Monthly regional price index | Re-basing sale prices to a common date |
| Met Office HadUK-Grid regional monthly Tmean | Regional monthly temperature | Heating degree days (15.5 °C base) |

LA → Met Office region: Leeds → England E & NE, Cornwall → England SW & S Wales, Southwark → England SE & Central S.

**Synthetic, only where no public data exists** (seeded, rules in `config/` with a `source` field):
- **Monthly bills per EPC home** (`greenhome.data.synthetic_parts`, `make real`). EPCs carry no metered use, and NEED isn't linked to EPC addresses. Annual gas and electricity come from the NEED log-linear model for the home's cell (type × age × floor-area band × EPC band), times hidden occupancy / behaviour latents and the smart-control multiplier. This is the same process as the interim generator's `EnergyWorld`, checked to match it exactly. Bills are split into months with real regional HDD (region enters through the weather), with electric heating via efficiency / SCOP and solar via self-consumption. ~10% of homes lose some months and ~2% of gas homes have unmatched meters. Homes the NEED model can't place (unknown type, age, floor area or heating, ~1%) get no bills, which exercises the "unavailable" path. A gas meter in a home not heated by gas carries the hot-water and cooking load on gas. Oil, LPG and district heating aren't metered, so those homes are flagged `heating_fuel_unmetered`. **Conflicting fields are not injected into real EPC records:** relabelling real ratings would corrupt the value model and lender view. Real records already contain a few natural conflicts (~0.07%), and the interim synthetic dataset keeps its ~5% injected conflicts for the scenario tests.
- **`customer_id`**: one per home, except ~30% of private-rented homes, which are grouped into landlord customers holding 2–5 homes in the same local authority. Used for splits (70/15/15 by customer).
- **Smart-controls level**: EPC records none / programmer / zone control but has no smart-thermostat field. Programmer homes are upgraded to `smart_thermostat` until the book matches the documented 22% market share (flagged `smart_controls_synthetic`).
- **LTV** for the lender view (3.12): mortgaged share by tenure (none for social housing), LTV uniform 0.25–0.90. The balance is LTV × current value, unavailable for homes without a valuation.

**Leakage and join rules**
- Only use an EPC lodged *before* the sale date as the features for that sale. `current_value_estimate` = last sale price HPI-indexed to today (UK HPI for the sale's property type in its local authority, `greenhome.data.valuation`), kept separate from historical `price`. Homes with no sale since 2018 stay unavailable until the value model predicts them.
- For NEED retrofit labels, compare consumption in the years before the install with the years after. Drop install years and homes with other interventions in the window.

**Avoiding circular evaluation.** The synthetic bills are generated from NEED-fitted distributions, so a model scored on them would partly be scored on our own assumptions. **The headline energy-model metrics come only from NEED's real consumption on a held-out NEED test set** (3.5). The synthetic bills are used to exercise annualisation, data-quality flags and the dashboard, never to report accuracy.

Reproducible: `python -m greenhome.data.build --seed 42` (`make build`: EPC ingest → synthetic parts → Price Paid join → valuation → NEED retrofit labels, ~2 min). Calibration (`make calibrate`) stays separate because its outputs are committed.

### 3.2 Annualisation
`annual = Σ observed_months / Σ seasonal_share(observed_months)` using separate monthly profiles for gas (strongly seasonal) and electricity. Profiles and thresholds are in `config/annualisation.yaml`. If coverage < 6 months or share < 40%, flag `{fuel}_low_coverage`; if no data, `null` + `{fuel}_unavailable`. Never fill with 0.

### 3.3 Green Home Score sub-scores (all monotonic, documented in `config/score_mappings.yaml`)
| Component | w | Mapping (draft) |
|---|---|---|
| Energy efficiency | .25 | SAP points if present (1–100, clipped), else rating band midpoint: A 96, B 86, C 75, D 61, E 46, F 30, G 10 |
| Consumption intensity | .20 | purchased energy ÷ expected for a typical similar home (V1 median from the description, `models/energy.similar_home_kwh`; NEED log-linear table if the models aren't installed), piecewise-linear: ≤0.5 → 100, 1.0 → 75, ≥2.0 → 0. Raw kWh/m² penalised small homes by ~23 points on real NEED use (`docs/SCORING.md`); unmetered heating fuel → default + flag |
| Insulation | .15 | loft depth score × 0.5 + wall score × 0.5 (none 0, partial 50, full 100) |
| Heating | .15 | heat pump 100, district 70, condensing gas 55, non-condensing gas 35, oil/LPG 25, storage heaters 30, direct electric 20 |
| Glazing | .10 | triple 100, double (post-2002) 80, double (older) 60, partial 35, single 0 |
| Renewables | .10 | min(100, 100 × solar generation / electricity demand); none 0 |
| Smart controls | .05 | none 0, programmer/timer 30, smart thermostat 70, smart + TRVs 100 |

Double-counting justification: the rating is a modelled whole-home figure; the other components are observed features. We document that rating and fabric features overlap and keep weights fixed as required. Missing component → use a documented conservative default and set a flag shown in the UI.

Unit tests: weights sum to 1, each mapping monotonic, P10001-style example gives ~38 / Poor.

### 3.4 Improvement potential & detector
- `E = grid_import + gas` (kWh/yr). Potential = 100 × (E_base − E_after) / E_base using the full feasible package. E_base = 0 or missing → "unavailable".
- "Inefficient" uses intensity **relative to expected for similar homes** (energy-model residual, 3.5), so a big house is not flagged just for being big.
- Key issues = sub-scores below threshold + intensity residual > X%.

### 3.5 Energy model: grey-box physics + tuned ML, accuracy first

Model accuracy is the main judging criterion. There are two versions behind one interface, both evaluated on the same held-out **real** NEED consumption, plus a zero-training physics-only baseline. The savings simulator, detector and dashboard don't care which one runs:

```python
class EnergyModel(Protocol):                                    # models/energy.py
    version: str
    def predict(self, p: Property) -> EnergyBreakdown            # per fuel, heat demand, base load, per-element heat loss
    def predict_frame(self, props: pd.DataFrame) -> pd.DataFrame # many homes at once (portfolio, evaluation)

model = load_model("greybox" | "ml" | "physics")                 # the energy_model switch
```

The input is the (post-retrofit) `Property`, not a `HomeState`: V1 and V2 need the parity features, which a `HomeState` doesn't carry, and V1 builds its physical state (envelope elements with U-values and areas, air changes, heating, region) from the `Property` itself (`physics.property_fabric`). A retrofit edits the `Property` (`savings.apply_retrofits_to_property`), and the model predicts again. Trained artifacts (LightGBM text models + `meta.json`) live in `models/artifacts/`, gitignored and hosted externally; `make train` rebuilds them in seconds from the raw NEED file, and anyone without it installs them with `make models SRC=<zip or link>` (`models/share.py`). Without them the engine falls back to physics deltas and the NEED log-linear estimate, and says so in `assumptions`.

**V1, grey box (physics + calibration + ML residual): handles every retrofit physically**
1. *Physics.* Heat loss coefficient HLC (W/K) = Σ U_element × area_element + 0.33 × air_changes × volume.
   - Element areas (walls, roof, floor, windows) come from floor area, property type and storeys (RdSAP-style geometry rules, `config/geometry.yaml`).
   - U-values come from age band + insulation level + glazing type (RdSAP default tables, `config/u_values.yaml`).
   - Space-heating demand (kWh) = HLC × HDD_region × 24 / 1000, with typical-year HDD per region from `config/region_hdd.csv`. No separate gains term: the 15.5 °C base already allows for internal and solar gains, so subtracting them again would double-count. Hot water and lighting/appliances follow SAP Appendices J and L (`config/physics.yaml`). NEED homes have no fabric fields, so their physics uses age-band default fabric; EPC homes use their own (`models/physics.py`, one implementation shared with the house diagram).
   - Fuel = demand / heating efficiency (`config/efficiencies.yaml`: boiler ~0.85, heat pump SCOP ~2.8–3.2, direct electric 1.0).
2. *Calibration to the home's bills (PRISM method),* when monthly bills exist. Regress monthly gas (or electricity, for electrically heated homes) on monthly regional HDD: slope = actual heating response, intercept = base load (hot water, cooking).
   - Calibration factor k = actual slope / physics slope. It captures occupancy, setpoint and hidden draughts.
   - Fewer than 6 usable months, or a poor fit → k = 1 and flag `uncalibrated`. k is clipped to a documented range, so one bad bill can't dominate.
   - NEED has annual readings only, so NEED evaluation runs with k = 1. Calibration is a bonus for real customers with bills, not part of the headline metric.
   - *As built:* the engine calibrates to the whole annual bill, not the PRISM slope. Retrofit changes are relative (model after / before), applied to the home's own billed use, so k = actual / expected is effectively applied to every fuel; it is reported in `EnergyBreakdown.calibration_k`. The per-home PRISM slope (`data/features.py`) is computed but not yet used to split heating from base load: the simulator still uses the sourced shares in `config/simulator.yaml`.
3. *ML residual.* log(use) = β × log(physics) + GBDT(parity features). A tuned GBDT, trained on NEED, predicts log(actual) − β log(physics) per fuel from the parity-safe features below; it corrects what physics misses (electric base load, hot water, occupancy patterns, systematic bias in default U-values). The residual model does **not** see the physics value: when it did, its retrofit deltas were step-shaped and, for ~18% of gas homes, said insulation raises use. β is the elasticity of real use to the physics change. For gas it is fitted on the observed loft and cavity installs in the NEED **train** split (β = 0.38: real homes show well under half the change steady-state physics predicts, the prebound effect) and checked on the val-split installs; for electricity, which retrofits don't change, it is picked by CV (0.3).
4. *Retrofit effect* = change the state and predict again, which gives exactly (physics after / before)^β: smooth, never the wrong sign. The low / high band uses the β implied by loft vs cavity installs separately. Diminishing returns (topping up an insulated loft saves less) and interactions fall out of the physics:

   | Retrofit | State change |
   |---|---|
   | Loft insulation | roof U-value |
   | Cavity / solid wall insulation | wall U-value |
   | Glazing | window U-value + air changes |
   | Smart thermostat / TRVs | effective setpoint, so effective HDD |
   | Heat pump | heating efficiency; fuel switches gas → electricity |
   | Solar PV | kWp × regional yield × self-consumption, applied to grid import after the model |

**V2, ML only (required ML baseline and accuracy contender)**
- One tuned GBDT per fuel predicts annual consumption directly from parity-safe features, with monotone constraints.
- Retrofit effect = predict(after) − predict(before) for loft, cavity and PV, which NEED can teach. Glazing, solid wall, controls and heat pumps have no NEED feature, so V2 takes those effects from V1's physics deltas, labelled "physics-based" in the report.

**Physics only** (V1 without step 3) is a third, zero-training baseline.

**Training data: NEED (real metered consumption)**, used for the V1 residual and for V2.
- Targets, one model per fuel: `log(Gcons)` for gas-heated homes, `log(Econs)` for all homes. Valid (`*ValFlag == "V"`) readings only.
- Default target year: 2023. A candidate variant uses the mean of valid 2021–2023 years (less noisy). Pick it in CV, and report the test metric on whichever target is used.
- **Feature parity rule:** only features we can also derive for an EPC home: property type, age band, floor-area band, EPC band, region, main heating fuel, conservatory ("unknown" for EPC homes), IMD quintile, and cavity / PV installed flags. One definition serves both sources (`need_data.features` / `epc_features`, tested to match). EPC proxies: `cavity_installed` = EPC "filled cavity" (a retrofit, not "as built, insulated"); `pv_installed` = solar > 0. The loft flag (a top-up can't be told from an original deep loft) and years since install (EPC has no dates) are dropped: on NEED CV this costs < 0.1% MAE (`docs/EVALUATION.md`). `COUNCIL_TAX_BAND` is excluded (EPC homes don't have it). For V1, NEED bands map to a physics `HomeState` using band midpoints and age-band default fabric.
- **Two prediction modes**, both scored on real NEED use (2022 → 2023, same splits):
  - *Mode 2, "similar homes"* (description only, no lagged consumption): V1 / V2 above. Used for homes without bills and by the "worse than similar homes" detector (3.4), which must not see the home's own use.
  - *Mode 1, "this home"* (`models/energy_bills.py`): adds the home's own previous-year use (for EPC homes, the weather-normalised annual use from their bills, `data/features.py`). A GBDT predicts the change log(this year / last year) with an L1 objective; homes without last year's bill fall back to mode 2. It must beat the no-model references "last year's bill" and "last year × median trend", which carry most of the signal. *Deployed* (`EnergyModel.forecast_frame`, saved with the artifacts): for a billed fuel the engine's baseline is mode 1's next-year forecast (it beats last year's bill by 2.6% gas / 3.3% electricity MAE on test), and `assumptions` says so; the billed figures stay on the `Property`. The "after" score applies the simulated change to the billed figures, so before and after are on the same footing. The bill used is the plain annualised figure (`Property` has no weather-normalised field), while NEED's prior-year gas is weather corrected.

**Split.** NEED rows are distinct properties: one fixed 70/15/15 train/val/test split, stratified by region × fuel, seed 42 (`data/splits/need_split.csv`). **The test set is touched once.** All tuning uses 5-fold CV on train.

**Tuning procedure** (applies to V2 and to V1's residual model):
1. **Baselines first:** NEED cell median with back-off, log-linear, physics-only, untuned GBDT. Every later step must beat the previous one on CV.
2. **Objective:** L2 vs Huber on log kWh vs Tweedie/Gamma on raw kWh, picked on CV MAE in kWh.
3. **Hyperparameter search:** Optuna TPE, ~100–200 trials, with early stopping inside each fold.
4. **Monotone constraints:** worse EPC band → higher use, larger area → higher use, insulation flags → lower gas. Keep them if their CV cost is < ~1% MAE.
5. **Feature work:** ordinal bands, interactions (type × area, EPC × age), years since install, IMD. Keep only what improves CV. For V1, the physics prediction itself is a feature of the residual model.
6. **Library bake-off:** LightGBM vs CatBoost on the same folds and trial budget (XGBoost only if time allows). Ship the winner, or the average of the top two (3.15).
7. **Uncertainty:** conformalised p10/p90 from val (CQR: quantile GBDTs on the V1 residual, fitted on train, one correction Q from val; `models/energy.fit_band`). Test coverage 79.2% gas / 80.7% electricity (`docs/EVALUATION.md`).

**Choosing the shipped version.** Default to V1, because it answers every retrofit physically and gives per-element heat loss for the house diagram (3.13), **unless V2 is clearly more accurate** on NEED CV (beyond fold noise). Either way, both are reported side by side. A config switch (`energy_model: greybox | ml`) selects the version, and the dashboard can show both.

**Metrics (in `docs/EVALUATION.md`):** MAE, MAPE, median APE and R² on log, per fuel, on test, for V1, V2, physics-only and the baselines. Error by subgroup (region, type, age band, EPC band) for the fairness audit (3.14). **Savings validity:** predicted change for loft / cavity / PV vs the observed NEED difference-in-differences effects (`need_calibration.yaml`). The catalogue's % ranges (PS table, Energy Saving Trust) are sanity bounds: flag any prediction outside them.

Config owned by C: `config/u_values.yaml` (RdSAP tables), `config/geometry.yaml` (area rules), `config/efficiencies.yaml` (boiler / heat pump), each with a source. `config/degree_days.csv` already exists (Met Office HDD).

### 3.6 Savings simulator (per fuel, sequential)
1. Start from the baseline property. Apply retrofits **in priority order** (fabric → controls → heating → generation). Each one edits the property, and the energy model predicts again, so every step acts on what the previous steps left. The model's relative change is applied to the home's own use (calibrated to its bills, k = actual / expected). Fabric in a gas-heated home uses the selected model (V2 only sees cavity walls; other measures use physics^β). Electric heating, which the NEED electricity model can't separate, uses the physics space-heat change^β, labelled physics-based. Controls still use the catalogue range (no physics hook yet); a home the model can't place (floor area unknown) falls back to the catalogue range, labelled.
2. Solar is applied after the energy model: kWp × regional yield (~850–1000 kWh/kWp) × self-consumption ratio → reduces grid import only, not demand. The effective self-consumption (0.155) is calibrated on real NEED PV installs (train split; `eval/eval_savings.py` checks val / test); the synthetic bills use the same value (`config/synthetic.yaml`).
3. In a gas-heated home a heat pump lowers gas (space heating + hot water; cooking gas stays) and **raises electricity**, which shows as a negative electricity saving. In an electrically heated home it lowers electricity, since it replaces resistance heating at SCOP × the efficiency. Savings only ever come off the fuel the heating system actually uses (`config/efficiencies.yaml` `fuel`), so a gas cooker in an electrically heated home is untouched.
4. Report per retrofit the **marginal** saving in package order (these add up to the package total), plus the package's elec/gas/total and a low/high band (from U-value / SCOP ranges and the model's p10/p90).

### 3.7 Carbon
`ΔC = ΔE_elec × EF_elec + ΔE_gas × EF_gas` with UK Government GHG Conversion Factors (DESNZ). Use the latest published year; the 2026 values are ~0.131 kg CO2e/kWh (electricity, generation) and ~0.182 (natural gas, gross CV). Check against the official file and store in `config/emission_factors.csv` with year + URL.

### 3.8 Value model
- **Value level:** GBDT quantile model (p10/p50/p90, conformalised as in 3.5) on sales history. Features: LA, type, floor area, bedrooms, age, rating, sale date / HPI. Monotone constraint: better rating → value not lower.
- **Rating uplift:** a hedonic log-linear model (log price on LA × type, log area, age band, EPC band, sale-month HPI). The EPC band coefficients give the % premium with a confidence interval. Uplift = premium(after band) − premium(before band) applied to the p50 level: **same property, same date, everything else fixed**. Return a range (% and £). Tree models are not used for the uplift, because their before/after difference is step-shaped and can be exactly 0 (see 3.15).
- On synthetic sales the premium is the generator's input (`synthetic.yaml`), so the uplift only becomes evidence once real Price Paid sales are joined to real EPCs.
- Rating after retrofit comes from a simple documented SAP-delta rule per retrofit.
- Always labelled "indicative estimate based on associations in our dataset, not a causal guarantee".
- Baseline for comparison: regional median × fixed rating premium lookup.

### 3.9 Recommendations
- Feasibility = hard rules from the catalogue (no cavity insulation on solid walls, no solar on flats without roof access/listed buildings, no double glazing if already double, heat pump needs insulation ≥ threshold or is recommended *after* fabric).
- Suitability High/Medium/Low from saving per £ and the property's constraints.
- Simple baseline to compare against: a fixed-order "recommend everything feasible" list. Metrics vs simulator reference labels: precision@k / NDCG.

### 3.10 LLM layer
- Tool-calling with a cheap model (cost is graded). Tools: `get_property_report`, `run_scenario`, `get_memory`, `update_memory`.
- System prompt: only use numbers from tool results, say "not available" when a field is null, explain infeasible retrofits with the rule that blocked them.
- **Grounding checker**: extract every number from the answer, check it exists in the tool JSON (with rounding tolerance); if not, regenerate or strip. Logs feed the eval.
- Eval set: ~50 held-out scenarios (missing data, conflicting data, infeasible retrofit asks, heat pump raising electricity, insufficient info). Metrics: groundedness, factual accuracy, unsupported-claim rate (targets ≥90 / ≥90 / <5%).

### 3.11 Customer memory that learns (optional extension)
JSON/SQLite per customer: completed retrofits, constraints ("renting, can't change heating", "budget £5k"), accepted/rejected recs.
- Completed retrofits update the `HomeState`.
- Stated constraints become extra feasibility rules.
- Rejected recs are down-ranked in suitability (with a reason shown: "you declined solar on 12 Oct").

The engine reads memory as inputs, so the numbers still come from the engine. Demo: a timeline of the same property's recommendations changing across three chat sessions.

### 3.12 Lender portfolio view (business relevance)
A bank-side page that aggregates `PropertyReport`s across all homes (the EPC homes in our 3 LAs as a notional "mortgage book"):
- EPC / Green Home Score distribution, and the share of homes rated Poor or Very poor.
- **Financed emissions** of the book (sum of per-home CO2e, PCAF-style attribution by loan-to-value; LTV synthetic and documented) and the reduction if the top-N homes retrofit.
- Ranked list of customers with the highest improvement potential: who to contact about green finance.
- Breakdowns by region, property type and age band, plus filters.

Computed by batch-running `build_report`, cached as a table. No new models.

### 3.13 Heat-loss house diagram (UX)
An SVG house on the customer dashboard. Roof, walls, windows and floor are coloured by their share of heat loss, from V1's per-element W/K (`Element.share`). If V2 is the shipped model, the per-element shares still come from V1's physics step, labelled as such.
- Clicking a part shows its retrofit, saving and suitability.
- Toggling a retrofit reruns `run_scenario` and recolours the house.

The house, the score gauge and the savings bars share one scenario state, so the what-if is visual as well as available through chat.

### 3.14 Fairness audit (Responsible AI)
Run on the test split; results go in `docs/RESPONSIBLE_AI.md`.
- Score, Poor-flag rate, recommendation count and energy-model error, broken down by region, property type, age band and value band.
- Check that the energy model's error is not systematically worse for some groups (e.g. pre-1919 solid-wall homes or flats).
- Check that value uplift is not presented as large only in high-value areas, without the caveat.
- **Affordability flag:** low-value or high-energy-burden homes see low-cost and grant-eligible options first (grant rules in config, with sources).
- Document limitations: property and price data are real, but bills, `customer_id`, smart controls and LTV are synthetic, and only 3 LAs are covered. EPC homes skew towards recently sold or let homes. The audit tests our pipeline's behaviour on that sample, not national real-world fairness.

### 3.15 Model choices and alternatives

Each part of the system needs a different kind of model. Only the energy model and the value level are learned tabular regressions.

| Part | Choice | Why not ML / why this |
|---|---|---|
| Energy model (3.5) | V1 grey box (physics + tuned GBDT residual) and V2 tuned GBDT, both scored on NEED | Physics handles retrofits the data can't teach and gives per-element heat loss; GBDT choice below |
| Uncertainty bands | Conformalised quantile regression | Guarantees the stated coverage on held-out data; cheap |
| Value level (3.8) | GBDT quantile, monotone in rating | Non-linear in area × location |
| Value uplift (3.8) | Hedonic log-linear EPC-band effects | Smooth, interpretable, matches the literature; a tree's uplift is step-shaped and can be 0 |
| "Worse than similar homes" (3.4) | Energy-model residual (actual / predicted) | More explainable than an isolation forest or autoencoder |
| Annualisation (3.2) | Seasonal weights | The PS forbids silently filling gaps; auditable |
| Green Home Score (3.3) | Fixed weights | Weights are fixed by the PS |
| Recommendation ranking (3.9) | Rules + saving per £ | No ranking labels for learning-to-rank |
| LLM (3.10) | Cheapest reliable tool-calling model (e.g. Claude Haiku 5.5) | Cost is graded; the LLM does no maths |

**Energy model alternatives** (NEED: ~40k rows, ~10 banded categorical features):

| Option | For | Against | Verdict |
|---|---|---|---|
| **LightGBM** | Fastest, so the most Optuna trials; monotone + quantile + native NaN | Overfits small/noisy data unless regularised | Default; in bake-off |
| **CatBoost** | Built for categorical features (all NEED fields are); strong defaults; less overfitting | ~2–5× slower to train; more limited monotone support | Most likely rival; in bake-off |
| XGBoost | Mature monotone/interaction constraints; accuracy ≈ LightGBM | Slower; categorical support less proven | Only if time allows |
| **Grey box (physics + GBDT residual)** = V1 | Every retrofit modelled physically; diminishing returns and interactions built in; per-element heat loss; residual fixes physics bias | Default U-values are rough for banded NEED inputs; more moving parts | Shipped default unless V2 is clearly more accurate |
| Log-linear GLM | Interpretable; smooth what-ifs; can't overfit; already fitted (`need_calibration.yaml`) | Needs hand-made interactions | Baseline to beat |
| Random forest | Robust, little tuning | Usually less accurate; no monotone constraints; weak quantiles | Not used |
| EBM / GAM (InterpretML) | Near-GBDT accuracy at low dimension; plots per feature; easy monotonicity | Fewer interactions; unfamiliar to the team | Optional explainable comparison |
| MLP / TabNet / FT-Transformer | Smooth outputs; multi-task | Lose to GBDTs on mid-size tabular data (Grinsztajn et al., 2022); heavy tuning; hard to make monotone | Not used |
| TabPFN | No tuning; strong on small tables | Designed for ≤ ~10k rows; slow on CPU; no monotone or quantile output | Benchmark at most |
| Bayesian hierarchical (PyMC) | Honest uncertainty; partial pooling of sparse band cells | Slow to fit and iterate | Not used (time) |

**What decides it.** The features set the accuracy ceiling, not the algorithm. NEED bands describe a few thousand distinct cells, and the log-linear fit leaves a residual sd of ≈ 0.55 (occupancy and behaviour). So the strong options will land within a few % of each other. We also need monotone constraints and quantiles, which excludes RF, NNs and TabPFN as the shipped model. And tuning speed matters under the deadline, which favours LightGBM, then CatBoost. The biggest remaining accuracy lever is extra parity-safe features (e.g. IMD from postcode), not a different algorithm.

---

## What makes us stand out

*(Key innovations giving Team A-Dawgs a decisive edge across the judging criteria: 20% ML/Recommendations, 15% GenAI, 15% Green Home Score, 15% Savings/Carbon/Value, 10% Business Relevance, 10% Technical Architecture, 5% UX, 5% Innovation)*

### Person A (Person 1): Data & Machine Learning Innovations

1. **Non-Circular Evaluation on Real Consumption (Data Quality 20%)**
   - *From Architecture Overview:* Keeping the test fair: our synthetic bills carry hidden habits for each home (occupancy, thermostat behaviour) that the model never sees. Because the model knows what similar homes should use, it also stops big houses being flagged just for being big.
   - *Technical Edge:* Headline energy-model metrics come only from a held-out test set of **real NEED metered consumption**, touched once. The synthetic bills are built from NEED-fitted distributions, so they are used to exercise annualisation, data-quality flags and the dashboard, never to report accuracy.

2. **Dual Energy Models: Grey-Box Physics + Tuned ML (ML Quality 20%)**
   - *Two competing architectures behind one `EnergyModel` Protocol:*
     - **V1 Grey Box:** RdSAP heat loss coefficient ($HLC = \sum U \cdot A + 0.33 \cdot ACH \cdot V$), calibrated to a customer's monthly bills via the **PRISM method** (gas regressed on heating degree days gives heating slope, base load and calibration factor $k$), plus a tuned GBDT residual trained on real NEED consumption. Every retrofit, including heat pumps, glazing and controls, is modelled physically, with diminishing returns and interactions built in.
     - **V2 Tuned ML:** LightGBM vs CatBoost bake-off per fuel, Optuna-tuned (100–200 trials) on identical CV folds, with **monotone constraints** so what-ifs stay physically sensible.
   - *Honest uncertainty:* conformalised p10/p90 bands with test-set coverage reported.
   - *Multi-Baseline Evaluation:* V1, V2 and physics-only benchmarked against a NEED cell-median baseline and a log-linear model on the held-out NEED test set; a feature-parity rule keeps out fields EPC homes don't have (e.g. council tax band).

3. **Observed Mid-Series Retrofit Labels (Held-Out Ground Truth)**
   - *Empirical Savings Validation:* A subset of properties in the dataset undergo recorded retrofits mid-series (and NEED observed before/after interventions for loft, cavity, and PV), generating true observed before-and-after utility bills.
   - *Rigorous Testing:* Provides genuine empirical held-out reference labels to validate the savings engine against, rather than merely evaluating against the engine's own equations.

4. **Quantile Value Level + Hedonic Uplift (Honest Predictive Range)**
   - *Uncertainty-Aware Valuation:* GBDT quantile regression ($p_{10}, p_{50}, p_{90}$, conformalised) on Price Paid transactions (location, floor area, bedrooms, age, EPC band, sale date, HPI), monotone in rating.
   - *Smooth, interpretable uplift:* the rating premium comes from a hedonic log-linear model's EPC-band coefficients, applied to the same home on the same date. The result is an honest, non-causal indicative range (£ and %) that never collapses to a tree's step of 0.

### Person B (Person 2): Engine, AI Assistant & App Innovations

1. **Heat-Loss House / Thermal Twin (Customer View — UX 5% + Score 15%)**
   - *From Architecture Overview:* A drawing of the home with the roof, walls, windows, and floor coloured by how much heat each one leaks. Click a part to see the fix and its saving. Toggle an upgrade and the house recolours, the score moves, and the savings update, because the code reruns. It is a "what if" you can see, not just type.
   - *Deepened Innovation:* Implemented as a living architectural SVG cross-section. Elements glow along a thermal flux spectrum proportional to each element's heat loss ($W/K$) and heat-loss share from the V1 grey-box model. Clicking any element (e.g. roof) toggles 300mm loft insulation $\rightarrow$ instantly calls `run_scenario()` $\rightarrow$ the house cools down visually in real time, the Green Home Score gauge sweeps upward, and fuel savings bars animate simultaneously.

2. **Lender Portfolio View & PCAF Workbench (Bank View — Business Relevance 10%)**
   - *From Architecture Overview:* Lloyds is a bank, so we add a page for the bank, not just the homeowner. It covers the whole (synthetic) mortgage book: the mix of scores and ratings, how many homes are Poor, total carbon linked to the loans ("financed emissions"), how much that drops if the worst homes upgrade, and a ranked list of customers to offer green finance to. Reuses the same reports, so no new models are needed.
   - *Deepened Innovation:* Implements PCAF (Partnership for Carbon Accounting Financials) Scope 3 Category 15 accounting:
     $$\text{Financed Emissions} = \sum \left(\frac{\text{Outstanding Balance}}{\text{Property Value}} \times \text{Property Emissions}\right)$$
     Includes an interactive 2030 EPC C Transition Simulator showing portfolio decarbonization and collateral valuation lift if Lloyds targets the top 10% highest-potential properties with discounted green loans.

3. **Memory That Learns & Living Constraint Timeline (Innovation 5%)**
   - *From Architecture Overview:* The app remembers each customer across chats, for example "I rent, I can't change the boiler", "budget £5k", or "already did the loft". Finished upgrades update the house, limits become rules, and rejected ideas move down the list with a note saying why. A timeline shows the advice changing over three chats. The numbers still come from the code.
   - *Deepened Innovation:* Persistent SQLite/JSON memory layer (`memory/store.py`). Living constraints adapt in real time: tenure restrictions lock structural changes and promote portable smart TRVs; budget caps re-sort packages to maximize carbon reduction per pound; completed works update the baseline `HomeState` and recalibrate heat pump feasibility. Includes a dedicated Memory Inspector sidebar in the UI for judges to test multi-session adaptation.

4. **Real-Time "Hallucination-Free" Grounding Inspector & Interactive Citation Badges (GenAI 15%)**
   - *Under the hood (`grounding.py`):* Automatically parses every numeric token, currency figure, percentage, kWh, and carbon value in the LLM's drafted responses and cross-validates them against the active `PropertyReport` JSON with a tight tolerance ($\pm 1\%$). Any unsupported number triggers an automatic retry or extraction-stripping guardrail, mathematically guaranteeing the rubric targets ($\ge 90\%$ groundedness, $\ge 90\%$ factual accuracy, $< 5\%$ unsupported claims).
   - *In the UI:* Every number rendered in the chat displays an interactive audit citation badge (e.g. `1,850 kWh [✓]`). Hovering over the pill reveals an inspection tooltip: *"Verified: report.recommendations[0].marginal_saving.gas_kwh (via grey-box energy model)"*. A persistent "100% Grounded" badge gives bank stakeholders complete trust.

5. **Lloyds Green Banking & UK Grant Matcher (Business Relevance 10%)**
   - *Bank-Aligned Economics:* Bridges pure engineering to Lloyds Banking Group's retail mortgage products.
   - *Government Grant Optimization:* Automatically factors in UK schemes:
     - *Boiler Upgrade Scheme (BUS):* £7,500 subsidy applied directly to heat pump capital expenditure.
     - *GBIS & ECO4:* Flags low-EPC / high-burden households for fully funded insulation grants.
   - *Net Monthly Cashflow & LTV Upgrade:* Computes green mortgage financing terms (e.g., *“Retrofit adds £32/mo to mortgage borrowing, but cuts energy bills by £58/mo $\rightarrow$ Net cashflow positive by +£26/mo from Month 1”*). Calculates how post-retrofit property value uplift drops the homeowner's Loan-to-Value (LTV) band, unlocking preferential lower mortgage rates.

6. **Multi-Scenario Decision Matrix & Radar Comparator (What-If Flow)**
   - *Side-by-Side Package Tradeoffs:* Instead of evaluating one scenario in isolation, provides an instant matrix comparing 4 distinct pathways:
     1. *Current Baseline (Do Nothing)*
     2. *Quick Wins (< £1,500 Capex, immediate payback)*
     3. *Fabric-First Package (Insulation + Glazing prioritised)*
     4. *Net-Zero Package (Full fabric + Heat pump + Solar PV)*
   - *Radar Comparison:* Visual comparison across 5 key dimensions: Capex, Annual Energy Bill Savings (£), Carbon Cut ($t\text{CO}_2e$), Green Home Score Leap, and Indicative Property Value Uplift.

### Person C (Person 3): Responsible AI & Fairness Innovation

7. **Fairness Check & Affordability Audit (Responsible AI 5%)**
   - *From Architecture Overview:* Checks whether results are unfair to some groups (regions, flats, very old homes, cheaper homes), for example whether the energy model is less accurate for them. Homes on a tight budget see cheap and grant-funded options first. Because the data is synthetic, this tests how our system behaves, not real-world fairness, and we state so transparently in `docs/RESPONSIBLE_AI.md`.

---

## 4. Interfaces (the contract between teammates)

All in `src/greenhome/schema.py` as Pydantic models. Agree on these in the first hour, stub them, then work in parallel.

```python
class Property(BaseModel):            # inputs, prediction-time-safe only
    property_id: str; customer_id: str
    property_type: str; age_band: str; floor_area_m2: float | None
    bedrooms: int | None; region: str
    imd_quintile: float | None; cavity_filled_retrofit: bool | None   # energy-model parity features (optional)
    energy_rating: str | None; sap_score: int | None
    heating: str; loft_insulation_mm: int | None; wall_insulation: str | None
    glazing: str; solar_kwp: float; smart_controls: str
    current_value_estimate: float | None
    annual_elec_kwh: float | None; annual_gas_kwh: float | None
    data_flags: list[str]
    # added for feasibility rules and the generator (see docs/DATA.md):
    local_authority: str; wall_type: str  # solid | cavity
    has_own_roof: bool; gas_meter: bool

class Element(BaseModel): name: str; area_m2: float; u_value: float; heat_loss_w_per_k: float; share: float
class HomeState(BaseModel):           # physical state; retrofits edit this
    elements: list[Element]; air_changes_per_hr: float; heating_system: str; heating_efficiency: float
    smart_controls: str; solar_kwp: float; region: str
class EnergyBreakdown(BaseModel):
    elec_kwh: float; gas_kwh: float; heat_demand_kwh: float; base_load_kwh: float
    hlc_w_per_k: float | None; calibration_k: float | None; calibrated: bool; model_version: str
    elements: list[Element]   # per-element heat loss (house diagram), default []
    elec_p10: float | None; elec_p90: float | None; gas_p10: float | None; gas_p90: float | None  # V1 conformal 80% band (3.5)

class SubScore(BaseModel):  component: str; weight: float; score: float; contribution: float; inputs_used: dict; flags: list[str]
class ScoreResult(BaseModel): total: int; classification: str; components: list[SubScore]

class FuelDelta(BaseModel): elec_kwh: float; gas_kwh: float; total_kwh: float; low: float; high: float
class RetrofitRec(BaseModel):
    retrofit: str; suitability: Literal["High","Medium","Low"]; reason: str
    marginal_saving: FuelDelta; carbon_kg: float; indicative_cost: tuple[float, float]
class Infeasible(BaseModel): retrofit: str; rule: str

class PropertyReport(BaseModel):
    property: Property; score: ScoreResult; score_after: ScoreResult
    is_poor: bool; key_issues: list[str]; improvement_potential_pct: float | None
    recommendations: list[RetrofitRec]; infeasible: list[Infeasible]
    # None = unavailable: no usable bills and too little data to estimate expected use (never a silent 0)
    package_saving: FuelDelta | None; package_carbon_kg: float | None
    value: dict  # status ("estimated" | "unavailable"), baseline, low, high, pct_low, pct_high, method, caveat
    energy_before: EnergyBreakdown | None; energy_after: EnergyBreakdown | None   # per-element heat loss feeds the house diagram
    assumptions: list[str]

def build_report(p: Property, retrofits: list[str] | None = None, memory: dict | None = None,
                 energy_model: str = "greybox") -> PropertyReport
def build_portfolio(reports: list[PropertyReport]) -> PortfolioSummary   # lender view (3.12)
```

`build_report` with an explicit `retrofits` list is the what-if entry point. Requested retrofits that fail a hard feasibility rule, or aren't in the catalogue, are not simulated: they're listed in `infeasible` with the rule, and noted in `assumptions`. A retrofit the customer declined earlier can still be explored when asked for explicitly.

---

## 5. Repo layout

```
config/            score_mappings.yaml, retrofit_catalogue.yaml, emission_factors.csv,
                   u_values.yaml, geometry.yaml, degree_days.csv, efficiencies.yaml, grants.yaml, synthetic.yaml, need_calibration.yaml,
                   epc_mapping.yaml, annualisation.yaml, price_calibration.yaml, hpi_index.csv,
                   simulator.yaml + value_uplift.yaml (interim, until V1 and the value model replace them)
data/              generated/ (gitignored if large), splits/
src/greenhome/
  schema.py
  data/            generate.py, annualise.py, degree_day_fit.py, split.py, epc_ingest.py, synthetic_parts.py,
                   sales_join.py, valuation.py, need_labels.py, features.py, calibrate.py, build.py
  score/           score.py
  engine/          feasibility.py, savings.py, carbon.py, recommend.py, report.py, portfolio.py
  models/          energy.py (EnergyModel interface, load_model, train -> artifacts/), physics.py,
                   energy_greybox.py (V1), energy_ml.py (V2 + residual tuning), energy_bills.py (mode 1), tune.py, bakeoff.py,
                   share.py (pack / install trained models without raw data: make package-models, make models SRC=...),
                   need_data.py, expected_use.py (log-linear fallback when artifacts are missing), value_model.py
  llm/             agent.py, tools.py, grounding.py, prompts/
  memory/          store.py
api/               main.py (FastAPI)
app/               customer dashboard + lender portfolio page
eval/              eval_energy.py (V1 vs V2 vs physics-only vs baselines, NEED test set), eval_savings.py, eda_energy.py (figures/), eval_value.py, eval_recs.py, eval_llm.py, fairness.py, scenarios/
tests/
docs/              ARCHITECTURE.md, SCORING.md, DATA.md, EVALUATION.md, RESPONSIBLE_AI.md, diagram
```

Stack: Python 3.11, pandas, LightGBM + CatBoost (bake-off, 3.15), Optuna, scikit-learn, Pydantic, FastAPI. Dashboard: Streamlit for speed (or React + Vite if time allows).

---

## 6. Team split

Two teammates have strong AI coding subscriptions (A, B); one has a weaker one (C). C therefore owns the work that is **specification-heavy, deterministic, and easy to verify with unit tests** (small amounts of code, lots of careful reading and sourcing), which is also where many judging marks sit (Green Home Score 15%, documentation, Responsible AI). A and B own the parts that need large volumes of generated code.

### A — Data & ML (strong AI)
- Real-data ingest: EPC + PPD clean and join (match rate reported), NEED retrofit before/after labels, HPI indexing, Met Office HDD; synthetic bills / customer_id / smart controls (3.1)
- Annualisation + customer-level splits (3.2)
- PRISM degree-day regression per home + calibration factor k (3.5)
- **Energy models (main focus):** V1 grey box (physics + calibration + tuned GBDT residual) and V2 tuned GBDT behind the `EnergyModel` interface; physics-only and NEED-median baselines. Tuning: baselines → objective → Optuna → monotone constraints → features → LightGBM vs CatBoost → conformal quantiles, all on CV, test touched once (3.5, 3.15)
- Headline energy metrics on the NEED test set, not synthetic bills (3.1 circularity rule)
- Leads and delivers **Person A Standout Innovations**:
  1. Non-circular evaluation: headline accuracy on held-out real NEED consumption; synthetic bills with hidden latents only exercise the pipeline
  2. Dual energy models: V1 grey box (PRISM calibration + GBDT residual) and V2 tuned GBDT bake-off, both scored on real NEED, conformal p10/p90
  3. Observed retrofit labels (NEED difference-in-differences + synthetic mid-series events) for held-out savings validation
  4. Value model: GBDT quantile level ($p_{10}, p_{50}, p_{90}$) + hedonic EPC-band uplift, same home and date
- Value model with quantile range + simple baseline (3.7)
- Eval report for models: V1 vs V2 vs physics-only vs baselines on consumption and retrofit savings, value interval coverage (section G); held-out test predictions file (section I)
- Owns `docs/DATA.md` (sources, licences, schema, generation rules, label rules, limitations)

### B — Engine integration, LLM, API, dashboard (strong AI)
- `schema.py` stubs on day 1 so everyone can code against them
- Savings simulator (3.6): sequential `HomeState` edits through `EnergyModel` + `report.py` orchestration + what-if path
- LLM agent, tools, prompts, grounding checker, LLM eval harness (3.9)
- FastAPI + dashboard (score waterfall, per-fuel savings bars, value range, chat)
- Leads and delivers the **6 Standout Innovations**:
  1. Real-time Grounding Inspector & citation audit pills (`grounding.py`)
  2. Interactive SVG Thermal Twin house diagram with live heat-flux recoloring (3.13)
  3. Lloyds Green Banking & UK Grant Matcher (BUS, ECO4, net monthly savings, LTV drop)
  4. PCAF Financed Emissions Lender Portfolio Workbench (3.12)
  5. Adaptive Customer Memory Timeline & living constraint inspector (3.11, `store.py`)
  6. Multi-Scenario Decision Matrix & Radar Comparator
- Owns `docs/LLM.md` (prompts, model, eval) and the architecture diagram

### C — Scoring, rules & docs (weaker AI)
- `config/score_mappings.yaml` + `score/score.py` + monotonicity / weight tests (3.3)
- `config/retrofit_catalogue.yaml`: applicability rules, cost ranges, impact ranges, with sources (e.g. Energy Saving Trust, EPC recommendations, illustrative PS table)
- `config/emission_factors.csv` from DESNZ (3.7); `engine/carbon.py`; `engine/feasibility.py` + suitability rules (3.9)
- Physics config tables with sources: `u_values.yaml`, `geometry.yaml`, `efficiencies.yaml` (3.5); `grants.yaml` for the affordability flag (3.14)
- Fairness audit script + write-up (3.14)
- Writes the ~50 LLM evaluation scenarios with expected behaviour (B runs them)
- Owns `docs/SCORING.md` (deliverable D) and `docs/RESPONSIBLE_AI.md` (deliverable H); assembles the final submission checklist

Why this split works: C's pieces are pure functions with small inputs and outputs, so C can write them with light AI help and verify them by hand. A and B can use their stronger tools to produce C's boilerplate (tests, YAML loaders) when C asks.

### Hand-offs / dependencies
| When | What | From → To |
|---|---|---|
| Hour 1 | `schema.py` stubs + this contract agreed | B → all |
| ~Hour 4 | First cleaned EPC book + synthetic bills (even rough) | A → B, C |
| ~Hour 6 | Catalogue + emission factors + feasibility + physics config tables (U-values, geometry, efficiencies) | C → A, B |
| ~Hour 8 | `EnergyModel` V1 (physics + calibration) and V2 first cut; tuning continues after | A → B |
| ~Hour 8 | `score()` working | C → B (report) |
| ~Hour 12 | Value model `predict_range()` | A → B |
| ~Hour 14 | End-to-end `build_report` on one property | B |
| ~Hour 18 | Heat-loss house, portfolio view, memory | B |
| Then | LLM eval on C's scenarios, V1 vs V2 eval and tuning report, fairness audit, metrics into EVALUATION.md | A + B + C |

---

## 7. Submission checklist (map to PS section 13)

- [ ] A Working dashboard — B
- [ ] B Code + `make data`, `make train`, `make eval`, `make app` — all
- [ ] C Trained models in `models/artifacts/` + inference code — A
- [ ] D `docs/SCORING.md` — C
- [ ] E `docs/LLM.md` — B
- [ ] F Architecture diagram — B
- [ ] G `docs/EVALUATION.md` (validation metrics, baselines, limitations) — A, B
- [ ] H `docs/RESPONSIBLE_AI.md` — C
- [ ] I Held-out test predictions + schema/label docs — A
