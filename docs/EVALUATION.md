# Evaluation

## Energy model — first cut (mid submission 1)

Reproduce with `make train`, which writes `eval/results/energy_first_cut.json`. The data is NEED anonymised 50k, **real metered 2023 consumption**, valid (`V`) readings only. Gas is modelled for gas-heated homes; electricity for all homes. The split is fixed at property level, 70/15/15, stratified by region × heating fuel (`data/splits/need_split.csv`). **The test split has not been scored**; numbers below are 5-fold CV on train and the val split.

| Fuel | Model | CV MAE (kWh) | CV R² (log) | Val MAE (kWh) | Val MAE, EPC homes only | Val R² (log) |
|---|---|---|---|---|---|---|
| Gas | Global median | 5,003 | −0.01 | 4,930 | 4,721 | −0.01 |
| Gas | Median of similar homes (back-off) | 4,241 | 0.24 | 4,131 | 3,894 | 0.26 |
| Gas | Log-linear (ridge) | 4,118 | 0.28 | 4,029 | 3,829 | 0.30 |
| Gas | **LightGBM (monotone, untuned)** | **4,117** | **0.28** | **4,016** | **3,819** | **0.30** |
| Elec | Global median | 1,672 | 0.00 | 1,715 | 1,728 | 0.00 |
| Elec | Median of similar homes (back-off) | 1,587 | 0.09 | 1,614 | 1,620 | 0.11 |
| Elec | Log-linear (ridge) | 1,589 | 0.10 | 1,618 | 1,624 | 0.11 |
| Elec | **LightGBM (monotone, untuned)** | **1,550** | **0.15** | **1,580** | **1,573** | **0.16** |

CV MAE fold-to-fold sd is ≈ 20–30 kWh. Gas medians are ≈ 10,100 kWh and electricity ≈ 2,600 kWh.

**Reading the numbers**
- **Gas:** both learned models clearly beat the no-model references, cutting MAE by about 18% against the global median. LightGBM and the log-linear model are tied within fold noise. The gas signal is mostly additive in these bands.
- **Electricity:** LightGBM beats every baseline by more than the fold noise, cutting MAE by about 2.4% against log-linear, so interactions matter here.
- **Accuracy ceiling:** homes with *identical* features (type × age × floor area × EPC × region) still differ with a log sd of 0.52, against 0.64 overall. So these features can explain at most about 33% of log-gas variance, and LightGBM already reaches R² 0.30. The rest is occupant behaviour: a home's own 2022 use correlates 0.85 with its 2023 use. A model that is not allowed to see a home's history (by design, §3.5) cannot remove that. The remaining gains come from new parity-safe features (e.g. IMD from postcode) and tuning, not from a different algorithm.

**Data preparation decisions** (leakage audit)
- Install flags count only measures installed **before** the target year. NEED contains 2024 installs, which would leak future information. Homes that installed a measure during 2023 are dropped, because their target is partly treated.
- `COUNCIL_TAX_BAND` is excluded: EPC homes don't have it (feature parity rule).
- "No EPC" homes (32%) stay in training as `has_epc = 0`. Metrics are also reported for EPC homes only, which matches deployment.
- Conservatory (81.5% missing, with missingness tied to EPC presence) is coded as an explicit "unknown" level.
- Monotone constraints: worse EPC band and larger floor area never lower predictions. For gas, loft and cavity insulation never raise it; for electricity, PV never raises it.

**Next (§3.5 steps 5–7):** Optuna tuning, the LightGBM vs CatBoost bake-off, feature work (IMD and interactions), conformal p10/p90, subgroup errors, and checking loft/cavity/PV toggles against NEED's observed DiD effects. Then a single scoring of the test split.
