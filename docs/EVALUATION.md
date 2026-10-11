# Evaluation

## Energy model — first cut (mid submission 1)

Reproduce with `make train`, which writes `eval/results/energy_first_cut.json`. The data is NEED anonymised 50k, **real metered 2023 consumption**, valid (`V`) readings only. Gas is modelled for gas-heated homes; electricity for all homes. The split is fixed at property level, 70/15/15, stratified by region × heating fuel (`data/splits/need_split.csv`). **The test split has not been scored**; numbers below are 5-fold CV on train and the val split.

| Fuel | Model | CV MAE (kWh) | CV R² (log) | Val MAE (kWh) | Val MAE, EPC homes only | Val R² (log) |
|---|---|---|---|---|---|---|
| Gas | Global median | 5,003 | −0.01 | 4,930 | 4,721 | −0.01 |
| Gas | Median of similar homes (back-off) | 4,241 | 0.24 | 4,131 | 3,894 | 0.26 |
| Gas | Log-linear (ridge) | 4,118 | 0.28 | 4,029 | 3,829 | 0.30 |
| Gas | **LightGBM (monotone, untuned)** | **4,119** | **0.28** | **4,015** | **3,823** | **0.30** |
| Elec | Global median | 1,672 | 0.00 | 1,715 | 1,728 | 0.00 |
| Elec | Median of similar homes (back-off) | 1,587 | 0.09 | 1,614 | 1,620 | 0.11 |
| Elec | Log-linear (ridge) | 1,589 | 0.10 | 1,618 | 1,624 | 0.11 |
| Elec | **LightGBM (monotone, untuned)** | **1,550** | **0.15** | **1,581** | **1,572** | **0.16** |

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
| Gas | **V1: physics × LightGBM residual** | **4,118** | **0.28** | **4,010** | **0.30** |
| Elec | Physics only (no training) | 1,659 | 0.01 | 1,680 | 0.03 |
| Elec | Physics × one scale factor | 1,621 | 0.02 | 1,653 | 0.03 |
| Elec | V2: LightGBM (untuned) | 1,550 | 0.15 | 1,581 | 0.16 |
| Elec | **V1: physics × LightGBM residual** | **1,549** | **0.15** | **1,578** | **0.16** |

**Reading the numbers**
- **Physics overpredicts NEED gas about 2×** (median 20,400 vs 10,100 kWh; the fitted scale factor is 0.51). This is the documented *prebound effect*: real homes, especially older and poorly rated ones, are heated less, and to lower temperatures, than steady-state physics assumes (Sunikka-Blank & Galvin, 2012). NEED homes have no fabric fields, so their physics uses age-band default fabric, which adds to the gap. Electricity physics (SAP lighting and appliances) is close: scale factor 0.89.
- **V1 ties V2 on NEED, as expected.** For NEED homes the physics prediction is a fixed function of the same bands the residual model already sees, so it can't add accuracy there. V1 is shipped anyway (ARCHITECTURE §3.5 default: V2 isn't clearly better). Its value is elsewhere: retrofits act through U-values and efficiencies with diminishing returns, it gives per-element heat loss for the house diagram, and real customers' bills calibrate it (k).
- Physics-only beats the global median only after one scale factor. Uncalibrated, it is worse than every baseline for gas, which is why the residual and calibration steps exist.

**Next (§3.5 steps 5–7):** Optuna tuning, the LightGBM vs CatBoost bake-off, feature work (interactions; IMD quintile is already in), conformal p10/p90, subgroup errors, and checking loft/cavity/PV toggles against NEED's observed DiD effects. Then a single scoring of the test split.
