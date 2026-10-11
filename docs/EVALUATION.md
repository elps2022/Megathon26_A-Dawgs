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
- **The home's own bill is the strongest single predictor**, and most of mode 1's accuracy comes from it. Plain GBDTs with the bill as a feature (L2 on log) lose to "last year's bill" in kWh MAE; predicting the *change* with an L1 objective beats it by ~2.6% (gas) and ~2.6% (electricity) on CV and val. The model's job here is shrinking unusual years back towards what the home's description predicts.
- **Classification view (val, mode 2, untuned V1):** picking the top 25% of users, V1 has precision 0.55 / recall 0.53 for gas (0.47 / 0.45 electricity), against 0.25 for a random pick.

## Hyperparameter tuning

`python -m greenhome.models.tune --trials 50` → `eval/results/energy_tuning.json` and `config/energy_params.yaml` (read by every LightGBM fit, so V1's residual and V2 share settings). Optuna TPE, 50 trials per fuel, over learning rate, leaves, minimum leaf size, feature and row sampling, L1/L2 penalties, minimum split gain and the objective (L2 or Huber on log kWh, Tweedie on raw kWh). Scored by 5-fold CV MAE in kWh on the train split only; val and test never seen.

| Fuel | Untuned CV MAE | Tuned CV MAE | Val MAE untuned → tuned | Chosen |
|---|---|---|---|---|
| Gas | 4,119 ± 27 | **4,079 ± 26** | 4,015 → **3,972** (−1.1%) | Huber loss (α 0.43), 17 leaves, ≥ 199 homes per leaf |
| Elec | 1,550 ± 24 | 1,550 ± 23 | 1,581 → 1,580 | Huber loss, 42 leaves (no gain) |

Tuning bought ~1% on gas and nothing on electricity, consistent with the feature ceiling above: the inputs, not the settings, limit description-only accuracy. Huber was picked for both fuels because it is less pulled by the few very high or very low users.

**Totals vs single homes (smearing).** Log-scale models predict the *typical* home, which sits below the average because use is skewed. Summed over a portfolio that undercounts by ~12–15% (gas) and ~18–31% (electricity). Duan's smearing factor, mean(exp(residual)) on the training data, corrects it: 1.15 (gas), 1.18 / 1.31 (electricity in gas / other homes) for the NEED fallback (`config/need_calibration.yaml`, applied in `models/expected_use.py`), and recorded per model in `energy_v1.json`. With it, description-only totals are within ~3% for 100 homes and ~1% for 1,000 homes (val), while single homes stay at ~29% median error.

**Deployment.** `models/energy.py` is the one interface (`load_model("greybox" | "ml" | "physics")`). `python -m greenhome.models.energy` (end of `make train`) refits V1 and V2 on the train split, saves them to `src/greenhome/models/artifacts/` (gitignored), and checks the saved models reproduce the val scores above exactly. The savings simulator calls it for every fabric retrofit.

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

**Not done:** the LightGBM vs CatBoost bake-off (dropped for time: tuning showed the settings aren't the limit) and quantile bands for V2.
