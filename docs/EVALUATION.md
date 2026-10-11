# Evaluation

## Energy model — first cut (mid submission 1)

Reproduce with `make train`, which writes `eval/results/energy_first_cut.json`. The data is NEED anonymised 50k, **real metered 2023 consumption**, valid (`V`) readings only. Gas is modelled for gas-heated homes; electricity for all homes. The split is fixed at property level, 70/15/15, stratified by region × heating fuel (`data/splits/need_split.csv`). **The test split has not been scored**; numbers below are 5-fold CV on train and the val split.

| Fuel | Model | CV MAE (kWh) | CV R² (log) | Val MAE (kWh) | Val MAE, EPC homes only | Val R² (log) |
|---|---|---|---|---|---|---|
| Gas | Global median | 5,003 | −0.01 | 4,930 | 4,721 | −0.01 |
| Gas | Median of similar homes (back-off) | 4,241 | 0.24 | 4,131 | 3,894 | 0.26 |
| Gas | Log-linear (ridge) | 4,118 | 0.28 | 4,029 | 3,829 | 0.30 |
| Gas | LightGBM (monotone, untuned) | 4,119 | 0.28 | 4,015 | 3,823 | 0.30 |
| Gas | **LightGBM (monotone, Optuna-tuned)** | **4,079** | **0.28** | **3,972** | **3,784** | **0.30** |
| Elec | Global median | 1,672 | 0.00 | 1,715 | 1,728 | 0.00 |
| Elec | Median of similar homes (back-off) | 1,587 | 0.09 | 1,614 | 1,620 | 0.11 |
| Elec | Log-linear (ridge) | 1,589 | 0.10 | 1,618 | 1,624 | 0.11 |
| Elec | LightGBM (monotone, untuned) | 1,550 | 0.15 | 1,581 | 1,572 | 0.16 |
| Elec | **LightGBM (monotone, Optuna-tuned)** | **1,549** | **0.15** | **1,580** | **1,570** | **0.16** |

CV MAE fold-to-fold sd is ≈ 20–30 kWh. Gas medians are ≈ 10,100 kWh and electricity ≈ 2,600 kWh. All rows use the **parity feature set** (below), i.e. only what an EPC home can also supply.

**Feature parity check** (`python -m greenhome.models.energy_ml --feature-sets` → `eval/results/energy_feature_sets.json`, LightGBM 5-fold CV MAE):

| Feature set | Gas | Elec |
|---|---|---|
| Full NEED (incl. loft flag and years since each install) | 4,117 ± 27 | 1,550 ± 24 |
| **Parity** (shipped: cavity and PV flags kept, loft flag and install years dropped) | **4,119 ± 27** | **1,550 ± 24** |
| No install flags at all | 4,123 ± 26 | 1,551 ± 24 |

Dropping what EPC homes can't supply costs nothing measurable (well inside fold noise), so the shipped model sees exactly the same features in training and in deployment.

**Reading the numbers**
- **Gas:** both learned models clearly beat the no-model references, cutting MAE by about 18% against the global median. LightGBM and the log-linear model are tied within fold noise. The gas signal is mostly additive in these bands.
- **Electricity:** LightGBM beats every baseline by more than the fold noise, cutting MAE by about 2.4% against log-linear, so interactions matter here.
- **Accuracy ceiling:** homes with *identical* features (type × age × floor area × EPC × region) still differ with a log sd of 0.52, against 0.64 overall. So these features can explain at most about 33% of log-gas variance, and LightGBM already reaches R² 0.30. The rest is occupant behaviour: a home's own 2022 use correlates 0.85 with its 2023 use. A model that is not allowed to see a home's history (by design, §3.5) cannot remove that. The remaining gains come from new parity-safe features and tuning, not from a different algorithm (IMD quintile is already a feature).

**Data preparation decisions** (leakage audit)
- **Parity features** (`need_data.epc_features`, tested to match `features()` exactly): the loft install flag is dropped (EPC can't tell a top-up from an original deep loft) and so are years since install (EPC has no dates). `cavity_installed` comes from EPC's "filled cavity" wording (a retrofit; "as built, insulated" is original): 20.9% of EPC homes vs 14.4% in NEED. `pv_installed` comes from solar > 0: 1.3% vs 1.4%.
- Install flags count only measures installed **before** the target year. NEED contains 2024 installs, which would leak future information. Homes that installed a measure during 2023 are dropped, because their target is partly treated.
- `COUNCIL_TAX_BAND` is excluded: EPC homes don't have it (feature parity rule).
- "No EPC" homes (32%) stay in training as `has_epc = 0`. Metrics are also reported for EPC homes only, which matches deployment.
- Conservatory (81.5% missing, with missingness tied to EPC presence) is coded as an explicit "unknown" level.
- Monotone constraints: worse EPC band and larger floor area never lower predictions. For gas, cavity insulation never raises it; for electricity, PV never raises it.

## V1 grey box and the physics-only baseline

Reproduce with `python -m greenhome.models.energy_greybox` (part of `make train`) → `eval/results/energy_v1.json`. Same parity features, folds and splits as above; test untouched.

| Fuel | Model | CV MAE (kWh) | CV R² (log) | Val MAE (kWh) | Val R² (log) |
|---|---|---|---|---|---|
| Gas | Physics only (no training) | 10,731 | −1.21 | 10,578 | −1.19 |
| Gas | Physics × one scale factor (fitted on train) | 4,548 | 0.14 | 4,439 | 0.16 |
| Gas | V2: LightGBM (untuned) | 4,119 | 0.28 | 4,015 | 0.30 |
| Gas | V1, first form: residual that also saw log physics (untuned) | 4,118 | 0.28 | 4,010 | 0.30 |
| Gas | V1, first form, tuned | 4,085 | 0.27 | 3,967 | 0.30 |
| Gas | V2: LightGBM, tuned | 4,079 ± 26 | 0.28 | 3,972 | 0.30 |
| Gas | **V1 shipped: physics^β × residual on the description, tuned (β = 0.38)** | **4,089 ± 28** | **0.27** | **3,987** | **0.30** |
| Elec | Physics only (no training) | 1,659 | 0.01 | 1,680 | 0.03 |
| Elec | Physics × one scale factor | 1,621 | 0.02 | 1,653 | 0.03 |
| Elec | V2: LightGBM (untuned) | 1,550 | 0.15 | 1,581 | 0.16 |
| Elec | V1, first form (untuned) | 1,549 | 0.15 | 1,578 | 0.16 |
| Elec | V2: LightGBM, tuned | 1,549 ± 23 | 0.15 | 1,580 | 0.16 |
| Elec | **V1 shipped: physics^β × residual, tuned (β = 0.3)** | **1,549 ± 23** | **0.15** | **1,581** | **0.16** |

**Reading the numbers**
- **Physics overpredicts NEED gas about 2×** (median 20,400 vs 10,100 kWh; the fitted scale factor is 0.51). This is the documented *prebound effect*: real homes, especially older and poorly rated ones, are heated less, and to lower temperatures, than steady-state physics assumes (Sunikka-Blank & Galvin, 2012). NEED homes have no fabric fields, so their physics uses age-band default fabric, which adds to the gap. Electricity physics (SAP lighting and appliances) is close: scale factor 0.89.
- **V1 ties V2 on NEED, as expected.** For NEED homes the physics prediction is a fixed function of the same bands the residual model already sees, so it can't add accuracy there. V1 is shipped anyway (ARCHITECTURE §3.5 default: V2 isn't clearly better). Its value is elsewhere: retrofits act through U-values and efficiencies with diminishing returns, it gives per-element heat loss for the house diagram, and real customers' bills calibrate it (k).
- Physics-only beats the global median only after one scale factor. Uncalibrated, it is worse than every baseline for gas, which is why the residual and calibration steps exist.
- **Why the shipped V1 changed form.** The first V1 fed log physics to the residual trees as a feature. Its levels were good, but its *retrofit deltas* weren't: trees are step functions and were free to move against physics, so on 3,000 real homes about 18% of loft and glazing recommendations in gas homes came out as an *increase* in use (clipped to 0). The shipped form is log(use) = β log(physics) + GBDT(description): the residual never sees physics, so a retrofit changes the prediction by exactly (physics after / before)^β, smooth and never the wrong sign. It costs 10 kWh of CV MAE on gas (well inside the fold sd of 28) and nothing on electricity. V2 is not clearly better beyond fold noise, so V1 still ships (ARCHITECTURE §3.5).

## Retrofit validity on observed installs

`energy_greybox` also writes `retrofit_check` to `energy_v1.json`. β for gas is fitted on the observed loft and cavity installs in the NEED **train** split only (`data/need_labels.py`: per-home use 2 years before vs after, difference-in-differences against never-treated homes); the val-split installs are held out. The test-split installs are unseen.

| Measure (gas-heated homes) | Split | Installs | Observed median change | Physics only | **V1 (physics^β)** |
|---|---|---|---|---|---|
| Loft insulation | train (fitted) | 1,512 | −3.1% | −5.2% | −2.0% |
| Loft insulation | **val (held out)** | 303 | **−2.4%** | −5.3% | **−2.1%** |
| Cavity wall insulation | train (fitted) | 960 | −10.4% | −28.3% | −11.9% |
| Cavity wall insulation | **val (held out)** | 193 | **−12.2%** | −28.1% | **−11.8%** |

- Physics alone overstates retrofit savings 2–3× (the prebound effect again); one elasticity, β = 0.38, brings both measures to within ~1 point of the observed median on held-out installs. The first-form V1 got loft right (−3.3%) but cavity badly wrong (−4.7%).
- Loft installs alone imply β ≈ 0.57 and cavity installs ≈ 0.35. The simulator uses that range as each fabric saving's low / high band.
- In the engine (3,000 real homes, `engine/savings.py`), the median V1 saving is 3.2% of gas for loft, 14.9% for cavity walls (these homes have fully empty cavities), 20% for solid walls and 5% for glazing; none is zero or negative. Glazing and solid-wall insulation have no NEED installs to check against, so they rest on the same β.

## Two prediction modes: description only vs the home's own bills

Reproduce with `python -m greenhome.models.energy_bills` (part of `make train`) → `eval/results/energy_own_bills.json`. Real NEED use, 2022 → 2023, same splits; 96% (gas) / 94% (electricity) of val homes have a valid 2022 reading, the rest fall back to mode 2.

| Fuel | Model | CV MAE (kWh) | Val MAE (kWh) | Val R² (log) | Val within ±20% | Val within ±30% |
|---|---|---|---|---|---|---|
| Gas | Mode 2: V1, description only | 4,089 ± 28 | 3,987 | 0.30 | 34.6% | 51.8% |
| Gas | Last year's bill (no model) | 2,025 ± 29 | 1,975 | 0.68 | 70.5% | 81.9% |
| Gas | Last year × median trend (no model) | 2,019 ± 28 | 1,977 | 0.69 | 70.3% | 81.3% |
| Gas | **Mode 1: change model (GBDT, L1)** | **1,972 ± 27** | **1,924** | **0.71** | **71.0%** | **81.6%** |
| Elec | Mode 2: V1, description only | 1,549 ± 23 | 1,581 | 0.16 | 27.7% | 40.4% |
| Elec | Last year's bill (no model) | 925 ± 17 | 969 | 0.53 | 58.4% | 72.0% |
| Elec | **Mode 1: change model (GBDT, L1)** | **901 ± 20** | **942** | **0.57** | **59.0%** | **72.2%** |

**Reading the numbers**
- **Description-only accuracy is capped by the inputs.** Among homes with identical type, age, size, EPC band and region, use still varies so much that these bands explain at most R² ≈ 0.25 (gas) / 0.12 (electricity); V1 reaches 0.30 / 0.16 with the extra parity features. Occupants, not buildings, drive most of the rest (building factors explain ~40% of domestic energy variation in Huebner et al., 2015).
- **The home's own bill is the strongest single predictor**, and most of mode 1's accuracy comes from it. Plain GBDTs with the bill as a feature (L2 on log) lose to "last year's bill" in kWh MAE; predicting the *change* with an L1 objective beats it by ~2.6% (gas) and ~2.6% (electricity) on CV and val. The model's job here is shrinking unusual years back towards what the home's description predicts. For totals the deployed forecast is scaled by Σ actual / Σ prediction on train (1.003 gas, 1.031 electricity), not Duan smearing: smearing overshot val totals by 4–5% because year-on-year errors are relatively larger for small users. With the ratio factor, val totals are within 0.4% (gas) / 0.2% (electricity).
- **Classification view (val, mode 2, untuned V1):** picking the top 25% of users, V1 has precision 0.55 / recall 0.53 for gas (0.47 / 0.45 electricity), against 0.25 for a random pick.

## Hyperparameter tuning

`python -m greenhome.models.tune --trials 50` → `eval/results/energy_tuning.json` and `config/energy_params.yaml` (read by every LightGBM fit, so V1's residual and V2 share settings). Optuna TPE, 50 trials per fuel, over learning rate, leaves, minimum leaf size, feature and row sampling, L1/L2 penalties, minimum split gain and the objective (L2 or Huber on log kWh, Tweedie on raw kWh). Scored by 5-fold CV MAE in kWh on the train split only; val and test never seen.

| Fuel | Untuned CV MAE | Tuned CV MAE | Val MAE untuned → tuned | Chosen |
|---|---|---|---|---|
| Gas | 4,119 ± 27 | **4,079 ± 26** | 4,015 → **3,972** (−1.1%) | Huber loss (α 0.43), 17 leaves, ≥ 199 homes per leaf |
| Elec | 1,550 ± 24 | 1,550 ± 23 | 1,581 → 1,580 | Huber loss, 42 leaves (no gain) |

Tuning bought ~1% on gas and nothing on electricity, consistent with the feature ceiling above: the inputs, not the settings, limit description-only accuracy. Huber was picked for both fuels because it is less pulled by the few very high or very low users.

**Totals vs single homes (smearing).** Log-scale models predict the *typical* home, which sits below the average because use is skewed. Summed over a portfolio that undercounts by ~12–15% (gas) and ~18–31% (electricity). Duan's smearing factor, mean(exp(residual)) on the training data, corrects it: 1.15 (gas), 1.18 / 1.31 (electricity in gas / other homes) for the NEED fallback (`config/need_calibration.yaml`, applied in `models/expected_use.py`), and recorded per model in `energy_v1.json`. With it, description-only totals are within ~3% for 100 homes and ~1% for 1,000 homes (val), while single homes stay at ~29% median error.

**Deployment.** `models/energy.py` is the one interface (`load_model("greybox" | "ml" | "physics")`). `python -m greenhome.models.energy` (end of `make train`) refits V1 and V2 on the train split, saves them to `src/greenhome/models/artifacts/` (gitignored), and checks the saved models reproduce the val scores above exactly. The savings simulator calls it for every fabric retrofit, mode 1 (saved too) sets the baseline for billed homes, and the score's consumption sub-score compares against V1's typical similar home (`similar_home_kwh`). Without the raw NEED file, install the trained models with `make models SRC=<zip or link>` (`make package-models` makes the zip).

## Held-out test set (scored once)

`make eval-test` → `eval/results/energy_test.json` and `eval/results/test_predictions.csv`. The NEED test split (15%: 5,494 gas-heated homes, 6,799 for electricity) was never used for fitting, tuning, the elasticity or the bands. Every model choice above was fixed before this run. The shipped artifacts are scored as deployed; baselines are refitted on train. 95% CIs are bootstraps over test homes (1,000 draws).

| Fuel | Model | MAE (kWh) [95% CI] | Median APE | R² (log) [95% CI] | R² (kWh) | Within ±20% |
|---|---|---|---|---|---|---|
| Gas | Global median | 4,993 | 36.5% | −0.01 | −0.03 | 27.3% |
| Gas | Median of similar homes | 4,177 | 30.7% | 0.25 | 0.25 | 33.3% |
| Gas | Log-linear | 4,061 | 30.3% | 0.28 | 0.28 | 33.7% |
| Gas | Physics × one scale factor | 4,546 | 33.5% | 0.14 | 0.18 | 29.8% |
| Gas | V2: LightGBM, tuned | 4,008 [3,902–4,108] | 29.4% | 0.28 [0.26–0.30] | 0.31 | 35.0% |
| Gas | **V1 shipped (physics^β × residual)** | **4,018 [3,919–4,124]** | **29.6%** | **0.27 [0.25–0.30]** | **0.31** | **35.2%** |
| Gas | This home: last year's bill (no model) | 2,005 [1,929–2,084] | 11.4% | 0.67 | 0.72 | 69.5% |
| Gas | **This home: mode 1 change model** | **1,953 [1,877–2,029]** | **10.9%** | **0.69 [0.66–0.72]** | **0.73** | **70.0%** |
| Elec | Global median | 1,670 | 40.9% | 0.00 | −0.09 | 24.5% |
| Elec | Median of similar homes | 1,569 | 37.9% | 0.10 | 0.03 | 26.7% |
| Elec | Log-linear | 1,573 | 38.6% | 0.11 | 0.04 | 26.7% |
| Elec | V2: LightGBM, tuned | 1,533 [1,486–1,578] | 37.2% | 0.16 [0.14–0.18] | 0.09 | 27.5% |
| Elec | **V1 shipped** | **1,535 [1,485–1,584]** | **37.2%** | **0.16 [0.14–0.18]** | **0.09** | **27.1%** |
| Elec | This home: last year's bill (no model) | 917 [882–950] | 16.0% | 0.54 | 0.56 | 58.9% |
| Elec | **This home: mode 1 change model** | **886 [851–920]** | **15.6%** | **0.58 [0.56–0.61]** | **0.60** | **58.5%** |

- **Test agrees with CV and val.** V1 gas 4,018 (val 3,987, CV 4,089); V1 and V2 are tied within their CIs, so V1 ships (ARCHITECTURE §3.5). Every trained model beats every no-model baseline on description only.
- **Mode 1 beats last year's bill on the same homes:** −2.6% MAE for gas (paired bootstrap 95% CI −3.1% to −2.0%) and −3.3% for electricity (−4.1% to −2.6%). Small but real; the bill carries most of the signal.
- **80% bands hold on new homes:** test coverage 79.2% gas, 80.7% electricity (val 80.0%), with misses split evenly above and below. They are wide (median width 122% of the prediction for gas, 161% for electricity), which honestly reflects how much occupants vary.
- **At scale, errors cancel:** predicted totals for random groups of 100 test homes are within 3.4% (gas) / 4.4% (electricity) of the metered total (median), and within 1.7% / 1.6% for 1,000 homes. Group-level R² (test homes grouped by region × type × age × floor-area band, ≥ 10 homes per group, weighted): **0.79 for gas**, 0.13 for electricity (0.09 adding heating type, an exploratory check added after the first run). Electricity differences between building groups are small next to occupant variation, so even group averages are hard to predict from buildings.
- **Ranking:** picking the top 25% of users, V1 precision / recall 0.55 / 0.54 (gas) and 0.45 / 0.44 (electricity) vs 0.25 random.
- **Retrofit installs, test split (first look):** loft observed −1.3% (319 homes) vs V1 −2.1% and physics −5.3%; cavity observed −11.4% (225) vs V1 −13.4% and physics −31.5%. V1 stays within ~1–2 points; physics alone overstates by 3–4×.

**Errors by subgroup (V1, test; full table in `energy_test.json` `subgroups_v1`, for the fairness audit).** Band coverage stays 76–84% in every region, type, age, EPC and deprivation group with ≥ 30 homes, so no group gets falsely confident bands. Single-home error is higher for flats (gas median APE 40.5% vs 25–33% for houses) and the smallest homes (40.8% under 50 m²); it falls with deprivation quintile (32.4% most deprived → 25.7% least). Median bias is within ±6% everywhere except gas in F/G-rated homes (+24%, only 42 homes): the model over-predicts their use, consistent with the prebound effect being strongest in the worst-rated homes. That matters for the "worse than similar homes" detector, which should not flag F/G homes on this basis alone.

**Test predictions file (deliverable I).** `eval/results/test_predictions.csv`: one row per test home and fuel (12,293 rows), keyed by `need_row` (row number in the public NEED 50k file), with the actual 2023 use, V1 p10 / p50 / p90 / mean and every other model's prediction (kWh). Schema in `docs/DATA.md`.

**Not done:** quantile bands for V2 (V1 ships, so only V1 has them).

## Post-test checks (train-split CV only; the test score above is unaffected)

**Library bake-off and interactions** (`python -m greenhome.models.bakeoff --trials 30` → `eval/results/energy_bakeoff.json`), on the shipped V1 target, same 5 folds:

| Fuel | LightGBM (shipped) | + type × area, EPC × age | CatBoost, 30 Optuna trials | Average of the two |
|---|---|---|---|---|
| Gas CV MAE | 4,089 ± 28 | 4,088 ± 31 | 4,082 ± 31 | 4,080 ± 30 |
| Elec CV MAE | 1,549 ± 23 | 1,549 ± 24 | 1,550 ± 24 | 1,548 ± 23 |

Every contender is within 9 kWh (0.2%) of the shipped model, against a fold sd of ~28: no library or interaction is clearly better, so LightGBM stays and the test score stands for the shipped model. Trees already find the interactions; as with tuning, the inputs are the limit.

**Savings checks** (`eval/eval_savings.py` → `eval/results/savings_checks.json`):
- **Solar, recalibrated.** The simulator's solar rule (3.5 kWp × 900 kWh/kWp × 40% self-consumption) predicted a 1,260 kWh/yr cut in import, but real NEED PV installs cut it by a median of 487 kWh (difference-in-differences, train split): the rule doubled real savings. `self_consumption` is now 0.155, calibrated on the 193 train installs (it bundles real self-consumption with the smaller systems of 2010–2020 installs; NEED has no kWp). Held-out check: val installs (57) observed −483 kWh vs rule −488; the 29 test installs show −190 kWh, too few to read much into.
- **Catalogue ranges** (1,000 real homes): model-based savings as % of heating demand are inside the catalogue's typical range for cavity walls (median 19%), solid walls (26%) and glazing (7%), with 3–6% of homes below. **Loft is below the catalogue's 10–25% for 83% of homes** (median 4.2%). Here the real installs side with the model: NEED loft installs cut gas by ~1–3% (above). The catalogue's loft range is now 2–25% (lower end from the observed NEED loft installs, which are mostly top-ups; post-test change to a sanity bound only, no model touched); 2.6% of homes fall below it.
- **V1 vs V2 on real installs** (frozen shipped models, post-test reporting). V2 learns a retrofit only through NEED's `cavity_installed` flag; switching it 0 → 1 on the homes that really had cavity walls filled:

  | Cavity wall installs | Installs | Observed | **V1** | V2 |
  |---|---|---|---|---|
  | train | 960 | −10.4% | −11.9% | −3.6% |
  | val | 193 | −12.2% | −11.8% | −3.5% |
  | test | 225 | −11.4% | −13.4% | −3.8% |

  V2 sees about a third of the real saving, probably because a GBDT trained on consumption levels learns the flag as a weak between-home difference (filled and unfilled homes differ in other ways too), not as a before-after change. V1's physics change with one fitted elasticity matches every split within ~2 points. V2 has no loft feature (the loft flag fails EPC parity), so it can't predict loft savings at all. This is the main reason V1 ships for savings even though the two tie on consumption accuracy.
- **Synthetic mid-series retrofits** (all 1,175 events in `data/generated`, every measure; hidden true change from `_hidden/retrofit_truth.csv`, labels only). The engine sees the description and the bills from before the install (annualised with seasonal weights) and simulates that one retrofit. Saving = before − after, kWh/yr:

  | Measure | Events | True median | V1 median | V1 total / true total | V2 total / true total | Sign right (V1) |
  |---|---|---|---|---|---|---|
  | Heat pump | 220 | 5,189 (gas +7,283, elec −2,176) | 4,769 (gas +6,549, elec −1,832) | 0.89 | 0.87 | 100% |
  | Solar PV | 258 | 483 | 461 | 1.01 | 1.00 | 100% |
  | Smart thermostat | 305 | 210 | 417 | 1.79 | 1.78 | 100% |
  | Double glazing | 142 | 262 | 219 | 0.87 | 0.82 | 86% |
  | Cavity wall | 68 | 688 | 1,229 | 1.63 | 0.60 | 87% |
  | Solid wall | 124 | 799 | 1,802 | 1.78 | 1.58 | 92% |
  | Loft | 58 | 197 | 384 | 1.96 | 1.81 | 81% |
  | **All** | **1,175** | 458 | 543 | **1.04** | 0.97 | 96% |

  - Across all events the predicted total is within 4% of the truth. The heat pump's electricity rise shows as a negative saving with the right size (−1,832 vs −2,176 kWh), and solar matches.
  - Fabric and controls savings run 1.6–2× the generator's. The generator gives every home the NEED *average* effect (e.g. −10.4% of gas for any cavity fill), while the engine's physics gives more to homes with fully empty cavities, thin lofts or no controls, which are exactly the homes these events pick. On the real installs above, which carry each home's own state, V1 matches. Part of the gap is the generator's simplification, but part may be real: treat fabric savings for the worst homes as the optimistic end.
  - The true per-home saving falls inside the engine's low–high band for only 28% (V1) of events. The band reflects uncertainty in the *average* effect (the β range), not how much one home varies (the generator draws each home's effect with a 50% spread), so it should not be read as an 80% interval for a single home.
  - Nothing was tuned on these checks.

**Figures** (`eval/eda_energy.py` → `eval/figures/`, data figures from the train split): target distributions (log-normal, truncated by NEED at 1,000–50,000 kWh gas and 500–25,000 kWh electricity), gas by EPC and age band, the within-cell spread (p10–p90 of use / identical-home median: 0.48–1.73 gas, 0.48–2.23 electricity, the noise ceiling), physics vs metered gas (median ratio 0.505, the prebound effect), the retrofit check, and test calibration by decile.

## Value model (ARCHITECTURE §3.8)

`make train-value` → `eval/results/value_v1.json` (5-fold CV grouped by home, val) and the artifacts; `make eval-value` → `eval/results/value_test.json`, the single test scoring. Data: real Price Paid resales 2018–2025 matched to the certificate lodged **before** each sale (new builds excluded), each indexed to Jul 2026 with the UK HPI for its type and local authority. Fixed customer split: 106,140 train / 22,970 val / 22,477 test sales.

| Model (test split unless noted) | Median error | Within ±10% | Within ±20% | R² (log) | MAE (£) |
|---|---|---|---|---|---|
| LA median × fixed rating premium (the spec's baseline) | 32.6% | 16.5% | 32.0% | 0.17 | 150,200 |
| LA × type median | 24.6% | 22.0% | 41.7% | 0.48 | 117,600 |
| Hedonic log-linear, no location (CV) | 22.2% | 23.7% | 45.6% | 0.37 | 110,800 |
| **Value model (LightGBM level)** | **14.9% [14.6–15.1]** | **35.3%** | **62.8%** | **0.76 [0.75–0.77]** | **73,900** |
| *Homes sold before (2,466 test sales):* indexed last sale | 8.5% | 55.6% | 81.3% | 0.85 | 45,400 |
| *Homes sold before:* **85% indexed last sale + 15% value model** | **8.2%** | **56.8%** | **82.9%** | **0.87** | **44,100** |

- **80% band:** test coverage 79.3% (val 80.0%), misses split evenly (10.3% below, 10.5% above); median width 59% of the estimate. Coverage is 78–81% in every local authority, property type and rating band with ≥ 50 sales.
- **Errors by group:** median error 12.9% Southwark, 13.6% Leeds, 17.4% Cornwall (more rural, more varied homes); 16.5% flats; 20.9–21.0% for F and G homes (few, varied sales). These go to the fairness audit (`value_test.json` `subgroups`).
- **Location is the missing input.** Inside a local authority, the only location signal is the deprivation quintile; a postcode-level feature would be the biggest single improvement (the shared property file carries no postcodes, by design).

**Rating premium (the value uplift).** Hedonic log-linear model on train sales, postcode-sector fixed effects (259 sectors, within-sector R² 0.68), standard errors clustered by sector:

| Band vs D | Premium (95% CI) | Same model without location | Re-estimated on test sales only |
|---|---|---|---|
| A/B | +1.0% (−1.0 to +3.0) | −1.5% | +0.6% |
| C | +0.7% (+0.1 to +1.3) | −0.7% | +0.7% |
| E | −2.7% (−3.4 to −2.0) | +1.9% | −4.1% |
| F/G | −5.1% (−6.6 to −3.7) | +6.9% | −5.5% |

- Without location controls the premium is **backwards**: within an LA, poorly rated homes are often expensive central period houses and A/B homes newer flats in cheaper areas. Comparing homes on the same streets fixes the order, and the test-only refit agrees.
- **Repeat sales** (the same home sold twice with a newer certificate in between, 2,587 train pairs): +6.4% per band gained (5.5–7.4%). This is an upper bound, because homes re-rated between sales have often been renovated as well. The shipped sector model is the conservative estimate.
- The uplift in a report is premium(band after) − premium(band before) for the same home on the same date, with the CI of the difference, so it is 0 when a retrofit doesn't change the band, and the range can include 0 (e.g. C → B, since A/B homes are rare). On 500 real homes the median range is 0% to +2.1%, at most +8%. Always labelled indicative, an association in historical sales, not a causal effect.
- **Rating after a retrofit:** SAP score regressed on the fabric fields over 376,769 real certificates (train-split customers, R² 0.70). Gains vs the base case: loft from < 50 mm +10.7, cavity fill +6.7, solid-wall insulation +8.1, single → new double glazing +6.6, old non-condensing → condensing boiler +14.3, solar +9.1; a smart thermostat vs a programmer ≈ 0 (SAP doesn't credit it).

## Recommendation ranking (ARCHITECTURE §3.9)

`eval/eval_recommend.py` (`make eval-recommend`) → `eval/results/recommend_check.json`. Does the engine put the best-value retrofits first?

- **Reference labels** come from the synthetic generator's hidden world, not from the engine: for each home and each retrofit that both consider possible, the true annual bill saving is the generator's typical-year change under the home's hidden occupancy, behaviour and per-home retrofit effect (mean of 20 effect draws), at Ofgem cap prices. True value = saving per £ of mid cost. The generator uses NEED observed effects and its own assumptions, while the engine uses the grey-box model and only sees the home's description and bills. Both share the NEED calibration, prices and catalogue costs, so this is a semi-independent check.
- **Sample:** 1,691 synthetic homes with complete descriptions and no mid-series retrofit, 2.7 shared retrofits per home on average. The suitability thresholds were set before this ran; nothing was tuned on it.

| Ranking | NDCG@3, bill saving per £ | Top pick truly best, bill saving per £ | NDCG@3, carbon per £ | Top pick truly best, carbon per £ |
|---|---|---|---|---|
| **Engine** (suitability, then payback) | **0.978** | **91.1%** | **0.968** | **89.2%** |
| Cheapest first | 0.968 | 88.8% | 0.904 | 75.3% |
| Fixed order, everything feasible (spec baseline) | 0.813 | 37.6% | 0.846 | 45.5% |
| Random | 0.803 | 39.4% | 0.820 | 39.4% |

- **Against the spec's baseline** (fabric, controls, heating, generation, in fixed order), the engine's top pick is right 53 points more often (CI 51–56). The fixed order is no better than random at choosing the first measure.
- **Against cheapest first**, a strong no-model reference because costs differ far more than savings, the gain on bill value is small but real: +1.0 NDCG points (CI 0.4–1.6) and +2.2 top-1 points (0.4–4.1). On carbon per £ it is clear: +6.4 NDCG points (5.8–7.0) and +13.8 top-1 points (12.1–15.7), because the engine knows which measures actually cut emissions (heat pumps) rather than just which are cheap.
- **Limitations:** synthetic truth only (no public data records several measures per home); costs are catalogue mid-points, the same for every home; solar counts only the grid-import cut (no export payments), so it ranks low; a heat pump rarely lowers the bill at October 2026 cap prices (gas 7.97p vs electricity 26.32p/kWh), so it ranks on its carbon cut.

## Post-test experiment: more than one year of the home's own history (mode 1)

`eval/exp_mode1_history.py` → `eval/results/exp_mode1_history.json`. Train 5-fold CV and val only; the test split is untouched. The shipped change model (log 2023 / 2022, L1) also given 2021 (deployable: customer bills cover 24 months) or 2021 + 2020 (exploratory: three years of bills aren't available for EPC homes).

| Fuel | Model | CV MAE (sd) | Val MAE | Val median APE | Val R² (log) |
|---|---|---|---|---|---|
| Gas | Last year's bill | 2,025 (29) | 1,975 | 11.1% | 0.68 |
| Gas | **Shipped, 1 year** | 1,972 (27) | 1,924 | 10.6% | 0.71 |
| Gas | 2 years | 1,946 (28) | 1,901 | 10.5% | 0.72 |
| Gas | 3 years | 1,940 (24) | 1,898 | 10.6% | 0.72 |
| Gas | Mean of last 2 years × trend | 2,070 (33) | 2,023 | 11.2% | 0.70 |
| Elec | Last year's bill | 925 (17) | 969 | 15.8% | 0.53 |
| Elec | **Shipped, 1 year** | 901 (20) | 942 | 15.6% | 0.57 |
| Elec | 2 years | 879 (22) | 922 | 15.3% | 0.59 |
| Elec | 3 years | 876 (21) | 920 | 15.4% | 0.59 |
| Elec | Mean of last 2 years × trend | 925 (26) | 974 | 16.7% | 0.56 |

- A second year cuts val MAE by 1.2% for gas (paired bootstrap 95% CI −1.8% to −0.6%) and 2.2% for electricity (−2.8% to −1.5%); CV agrees. A third year adds ~0.1–0.2%.
- Averaging past years is **worse** than last year alone (+5.1% gas, +3.3% electricity): the years are not exchangeable (2020–21 lockdowns raised home use, the 2022 price shock cut it), so history has to enter as separate features, not a mean.
- **Not shipped.** The gain is real but small next to the cost (a second annual figure on `Property`, retraining and re-hosting the models). The home's own last bill is close to the ceiling for single-home accuracy.

## Post-test checks: rating-band bias and who saves more (train / val only)

- **F/G over-prediction is mostly noise.** The test split showed V1 over-predicting gas for F/G-rated homes by +24% (42 homes). On val the same model is +6.9% for F/G (43 homes) and +1.7% for E (372), against −2% to −6% for A–D. NEED has too few F/G homes (< 1%) to fit a correction reliably, so none is applied; the detector caveat above stands.
- **β does not vary reliably with rating band.** The elasticity implied by loft and cavity installs per EPC band jumps between bands without a pattern (cavity 0.31–0.45 on train, loft noisy and sometimes the wrong sign in bands with < 150 installs). A per-band β would be fitted on noise, so the single β stays.
- **Physics doesn't tell real homes apart on how much they save.** Splitting the train + val installs into thirds by the physics-predicted cut:

  | Cavity wall installs | Installs | Physics cut | V1 prediction | Observed |
  |---|---|---|---|---|
  | Biggest physics cut | 385 | −35.9% | −15.6% | −10.1% |
  | Middle | 384 | −28.3% | −11.9% | −12.3% |
  | Smallest physics cut | 384 | −18.0% | −7.3% | −10.1% |

  Real savings are flat (~−10 to −12%) while V1 scales with physics. The slope of observed on physics log-change is 0.01 on train (95% CI −0.12 to 0.15) against V1's 0.38, and 0.21 on val (−0.09 to 0.51, inconclusive). Loft is too noisy to tell (physics barely varies between NEED loft homes). So V1 gets the **average** saving right, but its per-home spread is not supported by real installs: it likely overstates savings for the homes physics rates worst and understates them for the best. This matches the synthetic check (fabric savings 1.6–2× for the worst homes).
  - Caveat: NEED homes have no fabric fields, so their physics differs only through age, type and size defaults. EPC homes use their real walls, loft and glazing, which may carry real signal this test can't see.
  - **Not changed.** Shrinking every home toward the measure average would remove the per-home physics (diminishing returns, interactions, the house diagram) on evidence from one measure, and the val estimate is inconclusive. Read per-home fabric savings as the average effect adjusted by physics, with the low–high band as the range; portfolio totals are unaffected.
