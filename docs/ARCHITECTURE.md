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
 │ Data layer            │  synthetic "world" calibrated to NEED / EPC / Land Registry
 │  properties, monthly  │  + hidden latent vars (occupancy, true fabric quality)
 │  energy, sales,       │  that models never see -> realistic noise for evaluation
 │  retrofit history     │
 └─────────┬────────────┘
           ▼
 ┌──────────────────────┐
 │ Feature engineering   │  annualise (seasonal-weighted), kWh/m², data-quality flags,
 │                       │  degree-day regression per home (heating slope + base load)
 └─────────┬────────────┘
           ▼
 ┌──────────────────────┐   ┌──────────────────────────────────────┐
 │ Green Home Score      │   │ Energy model (two swappable versions) │
 │ 7 sub-scores, fixed w │   │  V1 grey box: heat-loss physics,      │
 │                       │   │     calibrated to bills, + ML residual│
 │                       │   │  V2 ML only: LightGBM                 │
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
 │ edits home state, re- │  │ DESNZ factors │  │ quantile, before vs after) │
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

### 3.1 Data (synthetic world, calibrated to public data)
- Generate ~10–20k properties. Distributions for type, age band, floor area, region, rating, heating, insulation, glazing taken from EPC open-data summary stats; consumption benchmarks from NEED tables; price levels from Land Registry regional medians.
- Each property has **hidden latent variables** (occupancy factor, true U-values, behaviour). Consumption = physics-ish demand model × latents × seasonal profile + noise. Models only see observable fields, so there is genuine prediction error to evaluate.
- **Avoid circular evaluation with the grey-box model (3.5).** The generator draws each home's *true* U-values, air-tightness and internal gains from distributions around the RdSAP defaults, and adds behaviour terms the physics model does not have (thermostat setpoint, heating hours, non-linear occupancy). The energy model only sees the age-band default tables. If the generator and model shared equations and parameters, accuracy would be meaningless.
- Monthly energy: 12–24 months, with ~10% of properties missing some months and ~5% with missing/conflicting fields (e.g. rating A but single glazing) to exercise the scenario tests.
- `retrofit_history`: a subset of homes get a retrofit mid-series, so we have *observed* before/after consumption = held-out reference labels for the savings model.
- `sales_history`: sale prices generated from a hedonic formula (location, size, type, age, rating premium from literature, market index by date, noise). `current_value_estimate` is a separate field from historical `price`.
- Everything seeded and reproducible: `python -m greenhome.data.generate --seed 42`.

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

### 3.5 Energy model: two versions behind one interface
Both versions implement the same interface, so the savings simulator, detector and dashboard do not care which one runs:

```python
class EnergyModel(Protocol):
    def predict(self, state: HomeState) -> EnergyBreakdown   # elec_kwh, gas_kwh, heat_demand_kwh, base_load_kwh, per-element heat loss
```

`HomeState` is the property's physical state (fabric, heating system, controls, solar). A retrofit edits the state, and the model predicts again. The rest of this section describes what goes on inside `predict`.

**V1, grey box (physics + ML): primary version**
1. *Physics.* Heat loss coefficient HLC (W/K) = Σ U_element × area_element + 0.33 × air_changes × volume.
   - Element areas (walls, roof, floor, windows) are estimated from floor area, property type and storeys (RdSAP-style geometry rules).
   - U-values come from age band + insulation level + glazing type (RdSAP default tables).
   - Space-heating demand (kWh) = HLC × HDD_region × 24 / 1000 − useful gains. HDD = heating degree days.
   - Fuel = demand / heating efficiency (boiler ~0.85, heat pump SCOP ~2.8–3.2, direct electric 1.0).
2. *Calibration to the home's bills (PRISM method).* Regress monthly gas (or electricity, for electrically heated homes) on monthly regional HDD: slope = actual heating response, intercept = base load (hot water, cooking).
   - Calibration factor k = actual slope / physics slope. It captures occupancy, setpoint and hidden draughts.
   - Fewer than 6 usable months, or a poor fit → k = 1 and flag `uncalibrated`.
   - k is clipped to a documented range so a single bad bill cannot dominate.
3. *ML residual.* LightGBM predicts what physics misses (electric base load, hot water, residual error) from observable features + k.
4. *Retrofit effect* = change the state and predict again:

   | Retrofit | State change |
   |---|---|
   | Loft insulation | roof U-value |
   | Wall insulation | wall U-value |
   | Glazing | window U-value + air changes |
   | Smart thermostat | effective setpoint, so effective HDD |
   | Heat pump | heating efficiency, fuel switches gas → electricity |

   Diminishing returns (topping up existing loft insulation saves less) and retrofit interactions fall out of the physics automatically.

**V2, ML only (required ML baseline + comparison)**
- LightGBM predicts annual elec and gas from observable features, including retrofit-state features.
- Retrofit effect = predict(after features) − predict(before features).
- Use LightGBM **monotone constraints** (e.g. more insulation can never increase predicted gas), so what-ifs stay physically sensible.
- Optional per-home scaling by actual / predicted baseline, mirroring k in V1.

**Physics only** (V1 without step 3) is reported as a third, zero-training baseline.

**Evaluation**
Compare V1, V2 and physics-only on two things:
- baseline annual consumption on held-out homes (MAE/MAPE per fuel);
- **retrofit savings on held-out homes from `retrofit_history`** (predicted vs "observed" before/after change, per fuel).

The catalogue's % ranges (PS table, Energy Saving Trust) are kept as sanity bounds: flag any prediction outside them in the eval report. A config switch (`energy_model: greybox | ml`) selects the version, and the dashboard can show both side by side.

Config owned by C: `config/u_values.yaml` (RdSAP tables), `config/geometry.yaml` (area rules), `config/degree_days.csv` (regional monthly HDD), `config/efficiencies.yaml` (boiler/heat-pump), each with a source.

### 3.6 Savings simulator (per fuel, sequential)
1. Start from the baseline `HomeState`. Apply retrofits **in priority order** (fabric → controls → heating → generation). Each one edits the state, and `EnergyModel.predict` runs again, so every step acts on what the previous steps left.
2. Solar is applied after the energy model: kWp × regional yield (~850–1000 kWh/kWp) × self-consumption ratio → reduces grid import only, not demand.
3. A heat pump lowers gas and **raises electricity**; this shows as a negative electricity saving.
4. Report per retrofit the **marginal** saving in package order (these add up to the package total), plus the package's elec/gas/total and a low/high band (from U-value/SCOP ranges and model error).

### 3.7 Carbon
`ΔC = ΔE_elec × EF_elec + ΔE_gas × EF_gas` with UK Government GHG Conversion Factors (DESNZ). Use the latest published year; the 2026 values are ~0.131 kg CO2e/kWh (electricity, generation) and ~0.182 (natural gas, gross CV). Check against the official file and store in `config/emission_factors.csv` with year + URL.

### 3.8 Value model
- LightGBM on sales history (features: region, type, floor area, bedrooms, age, rating, retrofits, sale date / market index). Quantile objectives (p10/p50/p90) or a bootstrap ensemble.
- Uplift = predict(after-scenario rating + features) − predict(baseline), **same property, same date, everything else fixed**. Return a range (% and £).
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
A bank-side page that aggregates `PropertyReport`s across all homes (the synthetic "mortgage book"):
- EPC / Green Home Score distribution, and the share of homes rated Poor or Very poor.
- **Financed emissions** of the book (sum of per-home CO2e, PCAF-style attribution by loan-to-value; LTV synthetic and documented) and the reduction if the top-N homes retrofit.
- Ranked list of customers with the highest improvement potential: who to contact about green finance.
- Breakdowns by region, property type and age band, plus filters.

Computed by batch-running `build_report`, cached as a table. No new models.

### 3.13 Heat-loss house diagram (UX)
An SVG house on the customer dashboard. Roof, walls, windows and floor are coloured by their share of heat loss (from V1's per-element W/K; for V2 or uncalibrated homes, sub-scores are used and labelled as such).
- Clicking a part shows its retrofit, saving and suitability.
- Toggling a retrofit reruns `run_scenario` and recolours the house.

The house, the score gauge and the savings bars share one scenario state, so the what-if is visual as well as available through chat.

### 3.14 Fairness audit (Responsible AI)
Run on the test split; results go in `docs/RESPONSIBLE_AI.md`.
- Score, Poor-flag rate, recommendation count and energy-model error, broken down by region, property type, age band and value band.
- Check that the energy model's error is not systematically worse for some groups (e.g. pre-1919 solid-wall homes or flats).
- Check that value uplift is not presented as large only in high-value areas, without the caveat.
- **Affordability flag:** low-value or high-energy-burden homes see low-cost and grant-eligible options first (grant rules in config, with sources).
- Document limitations: the data is synthetic, so the audit tests our pipeline's behaviour, not real-world fairness.

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

class Element(BaseModel): name: str; area_m2: float; u_value: float; heat_loss_w_per_k: float; share: float
class HomeState(BaseModel):           # physical state; retrofits edit this
    elements: list[Element]; air_changes_per_hr: float; heating_system: str; heating_efficiency: float
    smart_controls: str; solar_kwp: float; region: str
class EnergyBreakdown(BaseModel):
    elec_kwh: float; gas_kwh: float; heat_demand_kwh: float; base_load_kwh: float
    hlc_w_per_k: float | None; calibration_k: float | None; calibrated: bool; model_version: str

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
    energy_before: EnergyBreakdown; energy_after: EnergyBreakdown   # per-element heat loss feeds the house diagram
    assumptions: list[str]

def build_report(p: Property, retrofits: list[str] | None = None, memory: dict | None = None,
                 energy_model: str = "greybox") -> PropertyReport
def build_portfolio(reports: list[PropertyReport]) -> PortfolioSummary   # lender view (3.12)
```

`build_report` with an explicit `retrofits` list is the what-if entry point.

---

## 5. Repo layout

```
config/            score_mappings.yaml, retrofit_catalogue.yaml, emission_factors.csv,
                   u_values.yaml, geometry.yaml, degree_days.csv, efficiencies.yaml, grants.yaml
data/              generated/ (gitignored if large), splits/
src/greenhome/
  schema.py
  data/            generate.py, annualise.py, degree_day_fit.py, split.py
  score/           score.py
  engine/          feasibility.py, savings.py, carbon.py, recommend.py, report.py, portfolio.py
  models/          physics.py, energy_greybox.py, energy_ml.py, value_model.py, artifacts/
  llm/             agent.py, tools.py, grounding.py, prompts/
  memory/          store.py
api/               main.py (FastAPI)
app/               customer dashboard + lender portfolio page
eval/              eval_energy.py (V1 vs V2 vs physics-only), eval_value.py, eval_recs.py, eval_llm.py, fairness.py, scenarios/
tests/
docs/              ARCHITECTURE.md, SCORING.md, DATA.md, EVALUATION.md, RESPONSIBLE_AI.md, diagram
```

Stack: Python 3.11, pandas, LightGBM, scikit-learn, Pydantic, FastAPI. Dashboard: Streamlit for speed (or React + Vite if time allows).

---

## 6. Team split

Two teammates have strong AI coding subscriptions (A, B); one has a weaker one (C). C therefore owns the work that is **specification-heavy, deterministic, and easy to verify with unit tests** (small amounts of code, lots of careful reading and sourcing), which is also where many judging marks sit (Green Home Score 15%, documentation, Responsible AI). A and B own the parts that need large volumes of generated code.

### A — Data & ML (strong AI)
- Synthetic data generator with latents, missingness, retrofit history, sales history (3.1)
- Annualisation + customer-level splits (3.2)
- Degree-day regression per home + calibration factor k (3.5)
- Energy model **V1 grey box** (physics + calibration + ML residual) and **V2 ML only** (monotone LightGBM) behind the `EnergyModel` interface; physics-only and NEED-median baselines (3.5)
- Generator kept independent of the physics model's tables (3.1 circularity rule)
- Value model with quantile range + simple baseline (3.7)
- Eval report for models: V1 vs V2 vs baselines on consumption and retrofit savings, value interval coverage (section G); held-out test predictions file (section I)
- Owns `docs/DATA.md` (sources, licences, schema, generation rules, label rules, limitations)

### B — Engine integration, LLM, API, dashboard (strong AI)
- `schema.py` stubs on day 1 so everyone can code against them
- Savings simulator (3.6): sequential `HomeState` edits through `EnergyModel` + `report.py` orchestration + what-if path
- LLM agent, tools, prompts, grounding checker, LLM eval harness (3.9)
- FastAPI + dashboard (score waterfall, per-fuel savings bars, value range, chat)
- Heat-loss house diagram (3.13) and lender portfolio view (3.12)
- Customer memory that learns (3.11)
- Owns `docs/LLM.md` (prompts, model, eval) and the architecture diagram

### C — Scoring, rules & docs (weaker AI)
- `config/score_mappings.yaml` + `score/score.py` + monotonicity / weight tests (3.3)
- `config/retrofit_catalogue.yaml`: applicability rules, cost ranges, impact ranges, with sources (e.g. Energy Saving Trust, EPC recommendations, illustrative PS table)
- `config/emission_factors.csv` from DESNZ (3.7); `engine/carbon.py`; `engine/feasibility.py` + suitability rules (3.9)
- Physics config tables with sources: `u_values.yaml`, `geometry.yaml`, `degree_days.csv`, `efficiencies.yaml`; `grants.yaml` for the affordability flag (3.5, 3.14)
- Fairness audit script + write-up (3.14)
- Writes the ~50 LLM evaluation scenarios with expected behaviour (B runs them)
- Owns `docs/SCORING.md` (deliverable D) and `docs/RESPONSIBLE_AI.md` (deliverable H); assembles the final submission checklist

Why this split works: C's pieces are pure functions with small inputs and outputs, so C can write them with light AI help and verify them by hand. A and B can use their stronger tools to produce C's boilerplate (tests, YAML loaders) when C asks.

### Hand-offs / dependencies
| When | What | From → To |
|---|---|---|
| Hour 1 | `schema.py` stubs + this contract agreed | B → all |
| ~Hour 4 | First generated dataset (even rough) | A → B, C |
| ~Hour 6 | Catalogue + emission factors + feasibility + physics config tables | C → A, B |
| ~Hour 8 | `EnergyModel` V1 (physics + calibration) and V2 first cut | A → B |
| ~Hour 8 | `score()` working | C → B (report) |
| ~Hour 12 | Value model `predict_range()` | A → B |
| ~Hour 14 | End-to-end `build_report` on one property | B |
| ~Hour 18 | Heat-loss house, portfolio view, memory | B |
| Then | LLM eval on C's scenarios, V1 vs V2 eval, fairness audit, metrics into EVALUATION.md | A + B + C |

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
