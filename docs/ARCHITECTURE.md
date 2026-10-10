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
 │ Green Home Score      │   │ Energy model (accuracy is the focus)  │
 │ 7 sub-scores, fixed w │   │  tuned LightGBM per fuel, trained on  │
 │                       │   │  NEED real consumption, + documented  │
 │                       │   │  rule layer for unlearnable retrofits │
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

**Interim (now): fully synthetic dataset, same shape.** `make data` builds 10k homes from `config/` only (calibrated to NEED, Price Paid, UK HPI and Met Office via `make calibrate`). Teammates build against it while the real EPC book is prepared. Files and caveats: `docs/DATA.md`. Swapping in real EPC homes later changes `properties.csv` only; the bills stay synthetic either way.

**Real sources** (raw files in `data/raw/`, never committed; sources and licences in `docs/DATA.md`):

| Source | What it gives us | Used for |
|---|---|---|
| EPC domestic certificates (Leeds, Cornwall, Southwark) | Per-address property type, age band, floor area, rating/SAP, heating, insulation, glazing, PV, modelled energy | The property book: `Property` inputs, score, feasibility |
| NEED anonymised 50k sample | Banded property features + **real annual gas and electricity 2005–2023** + loft (`LI_DATE`), cavity wall (`CWI_DATE`) and PV (`PV_DATE`) install years | Training/evaluating the energy models; **observed before/after retrofit savings**; calibrating synthetic bills |
| Land Registry Price Paid (same 3 LAs, 2018–2022) | Real sale prices, date, type, tenure, new-build | Value model (joined to EPC by postcode + address) |
| UK HPI | Monthly regional price index | Re-basing sale prices to a common date |
| Met Office HadUK-Grid regional monthly Tmean | Regional monthly temperature | Heating degree days (15.5 °C base) |

LA → Met Office region: Leeds → England E & NE, Cornwall → England SW & S Wales, Southwark → England SE & Central S.

**Synthetic, only where no public data exists** (seeded, rules in `config/` with a `source` field):
- **Monthly bills per EPC home.** EPCs carry no metered use, and NEED isn't linked to EPC addresses. Annual gas and electricity are drawn from the NEED distribution for the home's matching cell (type × age × floor-area band × EPC band × region), scaled by a per-home noise factor. They're then split into months with the HDD-driven seasonal profile (gas strongly seasonal, electricity mildly). ~10% of homes lose some months and ~5% get conflicting fields, to exercise the scenario tests.
- **`customer_id`**: one per EPC address (a few multi-property customers), used for splits.
- **Smart-controls level**: not in EPC; drawn from documented market-share assumptions.
- **LTV** for the lender view (3.12).

**Leakage and join rules**
- Only use an EPC lodged *before* the sale date as the features for that sale. `current_value_estimate` = last sale price HPI-indexed to today, kept separate from historical `price`.
- For NEED retrofit labels, compare consumption in the years before the install with the years after. Drop install years and homes with other interventions in the window.

**Avoiding circular evaluation.** The synthetic bills are generated from NEED-fitted distributions, so a model scored on them would partly be scored on our own assumptions. **The headline energy-model metrics come only from NEED's real consumption on a held-out NEED test set** (3.5). The synthetic bills are used to exercise annualisation, data-quality flags and the dashboard, never to report accuracy.

Reproducible: `python -m greenhome.data.build --seed 42` (clean + join real data, then generate the synthetic parts).

### 3.2 Annualisation
`annual = Σ observed_months / Σ seasonal_share(observed_months)` using separate monthly profiles for gas (strongly seasonal) and electricity. If coverage < 6 months or share < 40%, mark `annual_estimate_quality = "low"`; if no data, `null` + "unavailable". Never fill with 0.

### 3.3 Green Home Score sub-scores (all monotonic, documented in `config/score_mappings.yaml`)
| Component | w | Mapping (draft) |
|---|---|---|
| Energy efficiency | .25 | SAP points if present (1–100, clipped), else rating band midpoint: A 96, B 86, C 75, D 61, E 46, F 30, G 10 |
| Consumption intensity | .20 | purchased kWh/m²/yr, piecewise-linear: ≤50 → 100, ≥350 → 0 |
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

### 3.5 Energy model: tuned ML, accuracy first

Model accuracy is the main judging criterion, so the energy model is a single, heavily tuned ML model with no physics model. One interface keeps the savings simulator, detector and dashboard independent of its internals:

```python
class EnergyModel(Protocol):
    def predict(self, x: HomeFeatures) -> EnergyBreakdown   # elec_kwh, gas_kwh, p10/p90 per fuel
```

`HomeFeatures` is the model-visible description of the home. A retrofit edits these features, and the model predicts again.

**Training data: NEED (real metered consumption).**
- Targets, one model per fuel: `log(Gcons)` for gas-heated homes, and `log(Econs)` for all homes, split by `MAIN_HEAT_FUEL`. Keep only valid readings (`*ValFlag == "V"`).
- Default target year: 2023. A candidate variant uses the mean of valid 2021–2023 years, which is less noisy. Pick it in CV, and report the test metric on whichever target definition is used.
- **Feature parity rule:** only use features we can also derive for an EPC home at prediction time. These are property type, age band, floor-area band, EPC band, region, main heating fuel, conservatory, and loft / cavity / PV flags with years since install. IMD band can be added if we download the ONS postcode → IMD lookup. `COUNCIL_TAX_BAND` is not available for EPC homes, so it is excluded from the shipped model. Report it only as an "upper bound" experiment.
- **No lagged consumption** (e.g. Gcons2022 to predict 2023) in the shipped model. The model must estimate "expected use for similar homes" (3.4). A lagged-use variant can be reported separately and labelled as such.

**Split.** NEED rows are distinct properties. Use a fixed 70/15/15 train/val/test split, stratified by region × fuel, seed 42. **The test set is touched once**, for the final numbers. All tuning uses 5-fold CV on train, with val for early stopping.

**Tuning procedure (where the effort goes):**
1. **Baselines first:** the NEED cell median (type × area × age × EPC), the log-linear model (as in `need_calibration.yaml`), and untuned LightGBM. Every later step must beat the previous one on CV.
2. **Objective:** compare L2 on log kWh, Huber on log kWh, and Tweedie/Gamma on raw kWh. Pick on CV MAE in kWh, which is what users see.
3. **Hyperparameter search:** Optuna TPE, ~100–200 trials, over `num_leaves`, `max_depth`, `min_child_samples`, `learning_rate`, `feature_fraction`, `bagging_fraction/freq`, `lambda_l1/l2` and `min_split_gain`, with early stopping on the fold's val part.
4. **Monotone constraints:** worse EPC band → higher use, larger area → higher use, insulation flags → lower gas. Measure their CV cost. Keep them if the cost is small (< ~1% MAE), because they make what-ifs physically sensible.
5. **Feature work:** ordinal encodings of bands, interactions (type × area, EPC × age) and years since install. Keep a feature only if it improves CV.
6. **Library bake-off:** LightGBM vs CatBoost, on the same CV folds and with the same Optuna trial budget. XGBoost joins only if time allows. Ship the winner on CV MAE, or an average of the top two if that is better. Then average across seeds. See 3.15 for why these candidates.
7. **Uncertainty:** conformalised quantile regression, i.e. quantile models (p10/p90) widened by a split-conformal correction computed on val. Report coverage on test, target ~80%.

**Metrics (reported in `docs/EVALUATION.md`):**
- MAE, MAPE, median APE and R² on log, per fuel, on test, against all baselines.
- Error by subgroup (region, type, age band, EPC band) for the fairness audit (3.14).
- **Savings validity:** predicted change from toggling loft / cavity / PV flags vs the observed NEED difference-in-differences effects (`need_calibration.yaml`).

**Rule layer, for retrofits NEED cannot teach.** NEED has no glazing, smart-controls, solid-wall or heating-system detail. Those what-ifs apply documented multipliers to the model's prediction:

| Retrofit | Effect | Source |
|---|---|---|
| Loft insulation, cavity wall | **Learned** (NEED flags), checked against DiD | NEED |
| Solar PV | Learned flag, or kWp × regional yield × self-consumption on grid import | NEED / PVGIS |
| Solid wall insulation, glazing | Fractional cut to heating gas | Catalogue range (EST) |
| Smart thermostat / TRVs | Fractional cut to heating gas | Catalogue range (EST) |
| Heat pump | Gas heating share × boiler eff / SCOP moves to electricity | `config/efficiencies.yaml` |

These are labelled "rule-based" in the report, so judges can see which numbers are learned and which are assumed.

**Using bills for EPC homes.** When a home has annualised bills, report actual vs predicted (the residual drives "worse than similar homes"). Optionally rescale what-if predictions by a shrunk actual/predicted ratio, validated on NEED by holding out the target year.

Config owned by C: `config/efficiencies.yaml` (boiler efficiency, heat-pump SCOP range) and the rule-layer effect ranges in `retrofit_catalogue.yaml`, each with a source. `config/degree_days.csv` already exists (weather normalisation).

### 3.6 Savings simulator (per fuel, sequential)
1. Start from the baseline `HomeFeatures`. Apply retrofits **in priority order** (fabric → controls → heating → generation). Each one edits the features, or applies its rule multiplier, and `EnergyModel.predict` runs again, so every step acts on what the previous steps left.
2. Solar is applied after the energy model: kWp × regional yield (~850–1000 kWh/kWp) × self-consumption ratio → reduces grid import only, not demand.
3. A heat pump lowers gas and **raises electricity**; this shows as a negative electricity saving.
4. Report per retrofit the **marginal** saving in package order (these add up to the package total), plus the package's elec/gas/total and a low/high band (from the model's p10/p90 and the rule-layer ranges).

### 3.7 Carbon
`ΔC = ΔE_elec × EF_elec + ΔE_gas × EF_gas` with UK Government GHG Conversion Factors (DESNZ). Use the latest published year; the 2024 values are ~0.207 kg CO2e/kWh (electricity, generation) and ~0.183 (natural gas, gross CV). Check against the official file and store in `config/emission_factors.csv` with year + URL.

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
- Completed retrofits update the `HomeFeatures`.
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
An SVG house on the customer dashboard. Roof, walls, windows and floor are coloured by typical heat-loss shares for the property type (Energy Saving Trust figures, in config with a source), scaled by the home's insulation and glazing sub-scores. Labelled "typical, not measured".
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
| Energy model (3.5) | GBDT, LightGBM vs CatBoost bake-off | See below |
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
| Log-linear GLM | Interpretable; smooth what-ifs; can't overfit; already fitted (`need_calibration.yaml`) | Needs hand-made interactions | Baseline to beat |
| Random forest | Robust, little tuning | Usually less accurate; no monotone constraints; weak quantiles | Not used |
| EBM / GAM (InterpretML) | Near-GBDT accuracy at low dimension; plots per feature; easy monotonicity | Fewer interactions; unfamiliar to the team | Optional explainable comparison |
| MLP / TabNet / FT-Transformer | Smooth outputs; multi-task | Lose to GBDTs on mid-size tabular data (Grinsztajn et al., 2022); heavy tuning; hard to make monotone | Not used |
| TabPFN | No tuning; strong on small tables | Designed for ≤ ~10k rows; slow on CPU; no monotone or quantile output | Benchmark at most |
| Bayesian hierarchical (PyMC) | Honest uncertainty; partial pooling of sparse band cells | Slow to fit and iterate | Not used (time) |

**What decides it.** The features set the accuracy ceiling, not the algorithm. NEED bands describe a few thousand distinct cells, and the log-linear fit leaves a residual sd of ≈ 0.55 (occupancy and behaviour). So the strong options will land within a few % of each other. We also need monotone constraints and quantiles, which excludes RF, NNs and TabPFN as the shipped model. And tuning speed matters under the deadline, which favours LightGBM, then CatBoost. The biggest remaining accuracy lever is extra parity-safe features (e.g. IMD from postcode), not a different algorithm.

---

## 4. Interfaces (the contract between teammates)

All in `src/greenhome/schema.py` as Pydantic models. Agree on these in the first hour, stub them, then work in parallel.

```python
class Property(BaseModel):            # inputs, prediction-time-safe only
    property_id: str; customer_id: str
    property_type: str; age_band: str; floor_area_m2: float | None
    bedrooms: int | None; region: str
    energy_rating: str | None; sap_score: int | None
    heating: str; loft_insulation_mm: int | None; wall_insulation: str | None
    glazing: str; solar_kwp: float; smart_controls: str
    current_value_estimate: float | None
    annual_elec_kwh: float | None; annual_gas_kwh: float | None
    data_flags: list[str]
    # added for feasibility rules and the generator (see docs/DATA.md):
    local_authority: str; wall_type: str  # solid | cavity
    has_own_roof: bool; gas_meter: bool

class HomeFeatures(BaseModel):        # model-visible state (NEED-compatible bands); retrofits edit this
    property_type: str; age_band: str; floor_area_band: int; epc_band: str; region: str
    main_heat_fuel: Literal["gas","other"]; conservatory: bool | None
    loft_insulated: bool; cavity_insulated: bool; solar_kwp: float
    # rule-layer only (not model inputs):
    wall_type: str; wall_insulation: str; glazing: str; smart_controls: str; heating: str
class EnergyBreakdown(BaseModel):
    elec_kwh: float; gas_kwh: float; elec_p10: float; elec_p90: float; gas_p10: float; gas_p90: float
    rules_applied: list[str]; model_version: str

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
    package_saving: FuelDelta; package_carbon_kg: float
    value: dict  # baseline, low, high, pct_low, pct_high, method, caveat
    energy_before: EnergyBreakdown; energy_after: EnergyBreakdown
    assumptions: list[str]

def build_report(p: Property, retrofits: list[str] | None = None, memory: dict | None = None) -> PropertyReport
def build_portfolio(reports: list[PropertyReport]) -> PortfolioSummary   # lender view (3.12)
```

`build_report` with an explicit `retrofits` list is the what-if entry point.

---

## 5. Repo layout

```
config/            score_mappings.yaml, retrofit_catalogue.yaml, emission_factors.csv,
                   degree_days.csv, efficiencies.yaml, grants.yaml, synthetic.yaml, need_calibration.yaml
data/              generated/ (gitignored if large), splits/
src/greenhome/
  schema.py
  data/            generate.py, annualise.py, degree_day_fit.py, split.py
  score/           score.py
  engine/          feasibility.py, savings.py, carbon.py, recommend.py, report.py, portfolio.py
  models/          energy_ml.py (train + tune), energy_rules.py, value_model.py, artifacts/
  llm/             agent.py, tools.py, grounding.py, prompts/
  memory/          store.py
api/               main.py (FastAPI)
app/               customer dashboard + lender portfolio page
eval/              eval_energy.py (tuned model vs baselines, NEED test set), eval_value.py, eval_recs.py, eval_llm.py, fairness.py, scenarios/
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
- **Energy model (main focus):** LightGBM per fuel on NEED, baselines → objective → Optuna search → monotone constraints → features → ensemble → quantiles, all on CV, test touched once (3.5)
- Headline energy metrics on the NEED test set, not synthetic bills (3.1 circularity rule)
- Value model with quantile range + simple baseline (3.7)
- Eval report for models: tuned model vs baselines on consumption and retrofit savings, value interval coverage (section G); held-out test predictions file (section I)
- Owns `docs/DATA.md` (sources, licences, schema, generation rules, label rules, limitations)

### B — Engine integration, LLM, API, dashboard (strong AI)
- `schema.py` stubs on day 1 so everyone can code against them
- Savings simulator (3.6): sequential `HomeFeatures` edits + rule layer through `EnergyModel` + `report.py` orchestration + what-if path
- LLM agent, tools, prompts, grounding checker, LLM eval harness (3.9)
- FastAPI + dashboard (score waterfall, per-fuel savings bars, value range, chat)
- Heat-loss house diagram (3.13) and lender portfolio view (3.12)
- Customer memory that learns (3.11)
- Owns `docs/LLM.md` (prompts, model, eval) and the architecture diagram

### C — Scoring, rules & docs (weaker AI)
- `config/score_mappings.yaml` + `score/score.py` + monotonicity / weight tests (3.3)
- `config/retrofit_catalogue.yaml`: applicability rules, cost ranges, impact ranges, with sources (e.g. Energy Saving Trust, EPC recommendations, illustrative PS table)
- `config/emission_factors.csv` from DESNZ (3.7); `engine/carbon.py`; `engine/feasibility.py` + suitability rules (3.9)
- `efficiencies.yaml` and rule-layer effect ranges with sources (3.5); `grants.yaml` for the affordability flag (3.14)
- Fairness audit script + write-up (3.14)
- Writes the ~50 LLM evaluation scenarios with expected behaviour (B runs them)
- Owns `docs/SCORING.md` (deliverable D) and `docs/RESPONSIBLE_AI.md` (deliverable H); assembles the final submission checklist

Why this split works: C's pieces are pure functions with small inputs and outputs, so C can write them with light AI help and verify them by hand. A and B can use their stronger tools to produce C's boilerplate (tests, YAML loaders) when C asks.

### Hand-offs / dependencies
| When | What | From → To |
|---|---|---|
| Hour 1 | `schema.py` stubs + this contract agreed | B → all |
| ~Hour 4 | First cleaned EPC book + synthetic bills (even rough) | A → B, C |
| ~Hour 6 | Catalogue (incl. rule-layer ranges) + emission factors + feasibility + efficiencies | C → A, B |
| ~Hour 8 | `EnergyModel` first cut (untuned LightGBM + baselines); tuning continues after | A → B |
| ~Hour 8 | `score()` working | C → B (report) |
| ~Hour 12 | Value model `predict_range()` | A → B |
| ~Hour 14 | End-to-end `build_report` on one property | B |
| ~Hour 18 | Heat-loss house, portfolio view, memory | B |
| Then | LLM eval on C's scenarios, energy-model tuning report, fairness audit, metrics into EVALUATION.md | A + B + C |

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
