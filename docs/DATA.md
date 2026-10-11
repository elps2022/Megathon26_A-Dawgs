# Data

## Real data in one command

`make build` (`python -m greenhome.data.build --seed 42`, about 2 min) runs the whole real data layer in dependency order: EPC ingest → synthetic parts → Price Paid join → valuation → NEED retrofit labels → features. Each step is described below and can also be run alone (`make epc`, `make real`, `make sales`, `make value`, `make labels`, `make features`). The calibration tables (`make calibrate`) are a separate step because their outputs are committed to `config/`. `make data` still builds the interim synthetic dataset, which needs no raw data.

## Sources and licences

Raw files go in `data/raw/` and are never committed. Everything we commit from them is an aggregate (the `config/` calibration tables and `eval/results/*.json`).

| Source | File in `data/raw/` | Publisher and link | Licence | Used for |
|---|---|---|---|---|
| EPC domestic certificates: Leeds, Cornwall, Southwark (lodged 2012-01 to 2026-09) | `epc/epc_domestic_leeds_cornwall_southwark.csv` | Ministry of Housing, Communities and Local Government, [Get energy performance of buildings data](https://get-energy-performance-data.communities.gov.uk) | Open Government Licence v3.0 for the non-address data. Address fields: Royal Mail and Ordnance Survey copyright, so they are never shared or committed | The real property book |
| NEED anonymised 50k sample (2025 release, consumption to 2023) | `need/anon2025_50k.csv`, `need/metadata.ods` | Department for Energy Security and Net Zero, [NEED anonymised data 2025](https://www.gov.uk/government/statistics/national-energy-efficiency-data-framework-need-anonymised-data-2025) | Open Government Licence v3.0 | Energy-model training and test, observed retrofit effects, synthetic-bill calibration |
| Price Paid Data 2018-2025, filtered to the three local authorities | `ppd/pp-YYYY-filtered.csv` (no header row) | HM Land Registry, [Price Paid Data](https://www.gov.uk/government/statistical-data-sets/price-paid-data-downloads) | Open Government Licence v3.0. Contains HM Land Registry data © Crown copyright and database right 2026. Address data: Royal Mail and Ordnance Survey copyright | Sales join, current values, price calibration |
| UK House Price Index, July 2026 | `hpi/UK-HPI-full-file-2026-07.csv` | HM Land Registry, ONS and partners, [UK HPI data downloads](https://www.gov.uk/government/statistical-data-sets/uk-house-price-index-data-downloads-july-2026) | Open Government Licence v3.0 | Indexing sale prices to today (by local authority and property type) |
| HadUK-Grid regional monthly mean temperature | `metoffice/Tmean_*.txt` | Met Office, [UK and regional series](https://www.metoffice.gov.uk/research/climate/maps-and-data/uk-and-regional-series) | Open Government Licence v3.0 | Heating degree days (base 15.5 °C) |
| ONS Postcode Directory, August 2026 (LS, WF, BD, TR, PL, EX, SE files) | `onspd/ONSPD_AUG_2026_UK_*.csv` | Office for National Statistics, Open Geography Portal | Open Government Licence v3.0. Contains OS data © Crown copyright and database right; Royal Mail data © Royal Mail copyright and database right; NRS data © Crown copyright | Postcode → LSOA 2011 for the IMD join |
| English Indices of Deprivation 2019, File 1 | `imd/File_1_-_IMD2019_Index_of_Multiple_Deprivation.xlsx` | Ministry of Housing, Communities and Local Government, English indices of deprivation 2019 | Open Government Licence v3.0 | IMD quintile per home (fairness audit, energy-model feature) |
| Greenhouse gas conversion factors 2026 | committed as `config/emission_factors.csv` | Department for Energy Security and Net Zero, [conversion factors 2026](https://www.gov.uk/government/publications/greenhouse-gas-reporting-conversion-factors-2026) | Open Government Licence v3.0 | Carbon (0.13096 kg CO2e/kWh electricity, 0.18231 gas) |

Synthetic fields (bills, `customer_id`, smart-thermostat upgrades, LTV) have no public source. Their rules are in `config/synthetic.yaml`, each with a `source`. These credits also go in the README for the final submission.

### Join and label rules (summary)

- **EPC → home:** the latest certificate per UPRN. `property_id` is a hash of the UPRN.
- **Postcode → IMD:** ONSPD `lsoa11cd`, then IMD 2019 decile → quintile ((decile + 1) // 2, 1 = most deprived), matching NEED's bands.
- **Sale → home:** postcode plus flat and house numbers, one UPRN only (92.2% matched). Features come from the certificate lodged on or before the sale date.
- **Value:** last sale × UK HPI by local authority and type. The historical `price` is kept separate from `current_value_estimate`.
- **NEED energy-model split:** fixed 70/15/15 (`data/splits/need_split.csv`). Homes that installed a measure during the target year are dropped. Install flags only count installs before the target year. The test set is scored once.
- **NEED retrofit effects:** difference-in-differences of the median post/pre ratio, 2 years each side, installs 2010-2020 (`config/need_calibration.yaml`).

## Interim synthetic dataset (needs no raw data)

Run `make data` (or `uv run python -m greenhome.data.generate --seed 42`). It takes ~6 s, needs only `config/`, and writes CSVs to `data/generated/`, which is gitignored. The same seed always gives identical files.

The data covers 10,000 EPC-like homes in Leeds, Cornwall and Southwark, with 24 monthly bills (Oct 2024 to Sep 2026) driven by real Met Office weather.

| File | Grain | Columns / notes |
|---|---|---|
| `properties.csv` | 1 row per home | `Property` fields (ARCHITECTURE §4) plus `local_authority`, `wall_type` (solid/cavity), `has_own_roof`, `gas_meter`. The state is **at series start, before any retrofit event**. `annual_elec_kwh` / `annual_gas_kwh` are annualised from the monthly bills by `data/annualise.py` (§3.2). They're None when a fuel has no readings, and `data_flags` records low coverage or unavailability. `recent_*_kwh` / `prior_*_kwh` are the latest 12 months and the 12 before, each annualised on its own (None when that year's coverage isn't good, or for gas without a gas meter). The heating fuel's years are weather-corrected to the typical year with the home's own degree-day fit when it is usable (`degree_day_fit.year_weather_factors`); other fuels stay as billed. The mode 1 two-year model uses them. `heating_share` is the space-heating share of the heating fuel at typical weather from the same fit (None without a usable fit); the savings simulator uses it to split heating from base load. For the ~12% of homes with a retrofit event, the annual figure mixes pre- and post-retrofit months. |
| `monthly_energy.csv` | home × month | `elec_import_kwh` (grid import, already net of solar self-consumption), `gas_kwh`. **Missing month = row absent.** `gas_kwh` NaN = gas meter not matched (data missing). `gas_kwh` 0 = home has no gas supply (`gas_meter` false), a real zero. |
| `retrofit_events.csv` | 1 per retrofitted home (~12%) | `retrofit`, `install_month`. Bills from that month onward reflect the change, so these give observed before/after pairs. |
| `sales.csv` | sale | `sale_date`, `price` (2018-01 to 2026-07). A historical price, separate from `current_value_estimate` in properties. |
| `loans.csv` | mortgaged homes (~70%) | `ltv`, `outstanding_balance` for the lender view. |
| `splits.csv` | customer | `split` ∈ train/val/test (70/15/15). Join through `properties.customer_id`. |
| `_hidden/latents.csv` | home | **Evaluation only, never features.** Occupancy and behaviour latents, true SAP/band, injected-conflict flag, true floor area, value-quality latent. |
| `_hidden/retrofit_truth.csv` | retrofit event | **Labels only.** True typical-weather annual change per fuel (`true_elec_change_kwh_yr`, `true_gas_change_kwh_yr`). |

Values to know:
- `heating`: gas_condensing, gas_non_condensing, oil, lpg, storage_heaters, direct_electric, heat_pump, district. Oil, LPG and district homes have unmetered heating fuel (only electricity in bills), so flag them.
- `wall_insulation`: none / partial / full. `glazing`: single / partial / double_old / double_new / triple. `smart_controls`: none / programmer / smart_thermostat / smart_trvs.
- `loft_insulation_mm` NaN with `has_own_roof` false means "not applicable" (a flat with no roof). NaN with `has_own_roof` true means missing.
- `age_band`: pre-1930, 1930-1972, 1973-1999, 2000+ (the NEED bands).

Injected data problems: ~10% of homes miss 1–9 months, ~2% of gas homes have no gas readings, 3% lack floor area, 4% have no EPC, 3% of houses lack a loft depth, and ~5% of D-or-worse homes are relabelled "A" (conflicting fields).

## How it is generated

All constants are in `config/synthetic.yaml`, each block with a `source`. The calibration tables come from real data via `make calibrate` (needs `data/raw/`) and are committed:

| File | From | Content |
|---|---|---|
| `config/need_calibration.yaml` | NEED 50k (2023) | Log-linear models of annual gas and electricity by type × floor-area band × age band × EPC band; **observed retrofit effects** (difference-in-differences, 2 years each side, valid readings only, gas effects against gas-heated never-treated homes): loft −2.7% gas, cavity wall −11.0% gas, PV −13.9% electricity |
| `config/degree_days.csv` | Met Office HadUK-Grid | Monthly HDD (base 15.5 °C, monthly-mean approximation) per LA |
| `config/price_calibration.yaml` | Price Paid 2018–2025 | Median price by LA × type |
| `config/hpi_index.csv` | UK HPI Jul 2026 | Monthly index per LA |

Consumption = the NEED model × hidden occupancy × behaviour × idiosyncratic noise (the NEED residual variance split across these), times smart-control and retrofit multipliers. This is then spread over months with real HDD. Electric heating converts gas-equivalent heat demand via efficiency (heat pump SCOP 2.6–3.3). Solar reduces grid import through a self-consumption share. Because the bills are built from the NEED-fitted model, **never report energy-model accuracy on them**: headline accuracy comes from the held-out NEED test set (ARCHITECTURE §3.5). SAP points come from a separate synthetic rule.

Prices = Price Paid median for LA × type × (area / reference area)^0.7 × EPC premium × hidden quality × HPI(date) / HPI(mean over the Price Paid years pooled in `price_calibration.yaml`) × noise.

Known limitations: HDD from monthly means understates shoulder months. Storage-heater bills (~10 MWh/yr) are on the high side. Simulated solar savings (~1,150 kWh/yr) are larger than NEED's observed −12% (≈ −320 kWh): NEED's PV is older and smaller, which makes a useful sanity check for the savings model. Glazing, solid wall and smart-control effects are assumptions, not observations.

## Real property book (EPC ingest)

Run `make epc` (`python -m greenhome.data.epc_ingest`, about 15 s). It needs `data/raw/epc/`, `data/raw/onspd/` and `data/raw/imd/`. Raw files are never committed (EPC and Price Paid addresses are under Royal Mail terms).

- **Input:** 731,331 domestic certificates for Leeds, Cornwall and Southwark (lodged 2012-01 to 2026-09).
- **Dedup:** the latest certificate per UPRN. Certificates without a UPRN (9,561) and park homes (303) are dropped. Result: **542,711 homes** (Leeds 253,557, Cornwall 191,954, Southwark 97,200).
- **Mapping:** free-text EPC fields map to the `Property` vocabulary with ordered rules in `config/epc_mapping.yaml`. Mapped share per field is 99.3-100% (`eval/results/epc_ingest_summary.json`). Anything assumed is listed in `data_flags`: for example `wall_insulation_assumed`, `loft_mm_assumed_from_description`, `glazing_install_date_unknown`, `gas_meter_inferred`. Oil, LPG and district-heated homes (72,557) are flagged `heating_fuel_unmetered`: their heating fuel is not in the electricity or gas bills. Mains-gas heating implies a gas supply, so 282 certificates whose mains-gas flag says "no" get `gas_meter` = true and `gas_meter_inferred_from_heating`. Gas room heaters (gas fires) map to `gas_non_condensing`, not to a boiler. Fields that can't be mapped become `"unknown"` and are treated as missing (default sub-score + flag; a survey is needed before the affected retrofits).
- **IMD:** postcode → LSOA 2011 (ONSPD Aug 2026) → IMD 2019 decile → NEED-style quintile (1 = most deprived). 100% joined.
- **Outputs (gitignored):** `data/processed/epc_properties.parquet` (with postcode, for internal joins) and `epc_properties_share.parquet` (no postcode, safe to share on Drive). `property_id` is a hash of the UPRN.
- **Filled by later steps:** bills and `customer_id` (`make real`), `current_value_estimate` (`make value`).

Known limitations:
- EPC has no smart-thermostat field, so `smart_controls` only gives none / programmer / zone control (as `smart_trvs`).
- Condensing vs non-condensing gas boilers is inferred from the main-heating efficiency rating, so only ~1% come out non-condensing, which is likely too low.
- Solar kWp is estimated from the % of roof covered × ground-floor footprint.
- Bedrooms are estimated as habitable rooms − 1.
- Certificates can be up to ~14 years old, and EPC homes skew towards those sold or let since 2012.

### Synthetic parts for the real homes (`make real`, ~1 min, after `make epc`)

`greenhome.data.synthetic_parts` adds only the fields that have no public source; it never alters real EPC fields. Seed 42 gives identical output. Outputs go to `data/processed/real/` (gitignored), and the aggregates are in `eval/results/real_synthetic_summary.json`.

| File | Content |
|---|---|
| `properties.parquet` / `properties_share.parquet` | The 542,711 real homes plus `annual_elec_kwh` / `annual_gas_kwh` (annualised by `data/annualise.py`), `recent_*_kwh` / `prior_*_kwh` (latest 12 months and the 12 before), `customer_id` and the final `smart_controls`. The share file has no postcode |
| `monthly_energy/part-*.parquet` | 12,637,637 monthly readings (Oct 2024 to Sep 2026): `elec_import_kwh`, `gas_kwh`. A missing month = absent row; gas NaN = unmatched meter |
| `loans.parquet` | 222,973 mortgages: `ltv`, plus `outstanding_balance` = LTV × value once `make value` has run (unavailable for homes with no value) |
| `splits.parquet` | train/val/test per customer (70/15/15) |
| `_hidden/latents.parquet` | **Evaluation only:** occupancy, behaviour, noise, heat-pump SCOP, unmatched-meter and missing-month counts |

- **Bills:** 537,732 homes get bills. The rest (unknown type, age, floor area or heating) are flagged `no_synthetic_bills_unknown_cell` and go down the "estimated from similar homes" or "unavailable" path. Typical medians: gas-boiler homes ≈ 7,832 kWh gas + 2,464 kWh electricity; heat pumps ≈ 4,924 kWh electricity. A gas meter in a home not heated by gas (17,521 electric or district homes) carries the hot-water and cooking load on gas (median 1,545 kWh) instead of a fabricated 0.
- **Customers:** 518,011, including 9,905 landlords holding 34,605 homes.
- **Smart controls:** 22% smart thermostat (synthetic upgrade of programmer homes), 10% zone control, 51% programmer and 16% none.
- **No conflicts injected into real records** (see ARCHITECTURE §3.1). Use the interim synthetic dataset for conflicting-data scenarios.

### Price Paid sales joined to EPC homes (`make sales`, ~45 s, after `make epc`)

`greenhome.data.sales_join` matches Land Registry Price Paid sales (2018-2025) to EPC homes. Output: `data/processed/real/sales.parquet` (gitignored, no address or postcode). Aggregates are in `eval/results/sales_join_summary.json`.

- **Sales kept:** category A (standard market sales; B = repossessions, bulk and company sales) and residential types only. That leaves 190,371 sales in the three local authorities.
- **Matching:** within the same postcode, a sale matches an EPC home when its flat and house numbers (SAON then PAON) equal the numbers in the EPC address, in the same order. Fallbacks: the same numbers in any order, or, for sales with no numbers, the house name appearing in the EPC address. A sale matches only when exactly one UPRN fits. Unit words (Flat, Apartment) and punctuation are ignored, and "21 A" counts as "21A".
- **Match rate: 92.2%** (175,457 sales). By local authority: Leeds 95.1%, Cornwall 89.6%, Southwark 89.9%. Unmatched: 3,390 ambiguous, 346 with no EPC in the postcode, 11,177 with no address fit, and 1 whose home is not in the book (its latest certificate is excluded).
- **Precision proxy:** the Price Paid property type agrees with the EPC type for 93% of matches. Most disagreements are neighbouring built forms (semi-detached vs end-terrace, terrace vs semi), which the two sources record differently. Flats agree in 97% of cases.
- **Leakage rule:** each sale gets the features of the latest certificate lodged **on or before** the sale date, never a later one. 168,351 sales (96%) have one (median 308 days before the sale). The rest keep the price but have `has_pre_sale_epc = False` and empty features.
- **Coverage:** 153,511 homes (28% of the book) sold at least once since 2018. The other homes have no recent price, so their value stays unavailable until the value model predicts it.
- `price` is the historical sale price. It is kept separate from `current_value_estimate`.

### Current value estimate (`make value`, ~4 s, after `make real` and `make sales`)

`greenhome.data.valuation` sets `current_value_estimate` = last matched sale price × HPI(now) / HPI(sale month). It uses the UK HPI index for the sale's property type (detached, semi, terraced or flat) in its local authority, up to **July 2026**, the latest HPI month. It updates `properties.parquet`, `properties_share.parquet` and `loans.parquet` in `data/processed/real/` (recomputed from scratch each run) and writes `valuations.parquet` (last sale date, price and both index values). Aggregates are in `eval/results/valuation_summary.json`.

- **153,511 homes valued (28%)**, each flagged `value_hpi_indexed_last_sale`. Medians: Leeds £262k, Cornwall £288k, Southwark £553k. The median index change since the sale is +8.3%.
- **The other 72% stay unavailable** (no sale since 2018). Nothing is imputed; the value model (ARCHITECTURE 3.8) is what predicts them.
- **Loans:** 71,491 of 222,973 mortgages now have an `outstanding_balance`, so the lender view's financed emissions work for them.
- Limitations: works done since the sale aren't reflected, and 14,676 values come from new-build sales, which carry a new-build premium that fades over time.

## Observed retrofit labels from NEED (`make labels`, ~1 s)

`greenhome.data.need_labels` gives each NEED home with a real loft insulation, cavity wall insulation or solar PV install (2010-2020) its own before/after label, for validating the savings simulator on held-out real installs. Output: `data/processed/need_retrofit_labels.parquet` (gitignored). Aggregates are in `eval/results/need_retrofit_labels_summary.json`.

- **Label:** `pre_kwh` / `post_kwh` = mean annual use over the 2 years before and the 2 years after the install year (the install year is dropped, and all four readings must be valid). `change` = post / pre − 1. `did_change` divides out the same-years trend of never-treated homes (computed from non-test homes only).
- **Rules** (same as the pooled effects in `need_calibration.yaml`): gas labels only for gas-heated homes, solar on electricity, and homes with another install within 2 years are dropped.
- **Counts (train / val / test):** loft 1,512 / 303 / 319, cavity wall 960 / 193 / 225, solar 193 / 57 / 29. Each home keeps its fixed NEED split, so the test labels belong to homes the energy model never trains on.
- **Train + val medians (DiD):** loft −3.1% gas, cavity wall −10.9% gas, solar −15.5% electricity. These are close to the pooled effects in `need_calibration.yaml` (−2.7%, −11.0%, −13.9%), which use the same rules. **Test medians are withheld** until the final savings evaluation.
- Limitation: only 29 solar labels in test, so the solar savings check has wide uncertainty. Report it with a bootstrap interval.

## Stage 2 features (`make features`, ~15 s; also built for the interim data by `make data`)

`greenhome.data.features` writes one row per home: `data/processed/real/features.parquet` (gitignored) and `data/generated/features.csv` for the interim dataset. Aggregates are in `eval/results/features_summary.json`.

| Columns | Content |
|---|---|
| `prop_type` … `pv_installed` | The energy model's inputs, built by `need_data.epc_features` with exactly NEED's codes (tested). Unknowns stay NaN / "unknown" |
| `heating_fuel`, `fit_*`, `base_kwh_per_month`, `slope_kwh_per_hdd` | Per-home degree-day regression of monthly bills on regional HDD (`data/degree_day_fit.py`): base load, heating response (V1's calibration input), R², months used, and `fit_issue` when unusable (`no_readings`, `too_few_months`, `implausible_fit`, `weak_fit`, `solar_confounded`) |
| `annual_gas_kwh_wn`, `annual_elec_kwh_wn` | Weather-normalised annual use: the heating fuel at typical-year (2015–2024) HDD when the fit is usable, otherwise the annualised figure + `weather_normalisation_unavailable`. Non-heating electricity stays annualised, matching NEED (gas is weather corrected there, electricity isn't) |
| `elec_kwh_per_m2`, `gas_kwh_per_m2`, `purchased_kwh_per_m2` | Intensities from the normalised figures; NaN + `intensity_unavailable` without floor area or a fuel |
| `elec_months`, `gas_months`, `*_year_share` | Coverage: months observed and the share of a typical year they cover |
| `feature_flags` | `weather_normalisation_unavailable`, `heating_fuel_unmetered`, `intensity_unavailable` |

- **Fits:** 97.7% of gas-heated homes and 98.2% of electrically heated homes have a usable fit (≥ 6 months, positive slope, base ≥ 0, R² ≥ 0.5; `config/annualisation.yaml`). The rest have no readings (8,035: unmatched gas meters or no bills) or are electrically heated with solar (2,051): solar lowers summer grid import, which bills alone can't separate from weather (it steepened the slope by ~30%), so those homes aren't normalised.
- **Normalisation effect:** the billing period (Oct 2024 – Sep 2026) was 9–11% milder than 2015–2024, so normalised heating-fuel use is ~7% (gas) / ~5% (electric heating) above the raw annualised figure.
- **Circularity:** synthetic bills are generated from the same HDD, so the median R² (0.99) only shows the pipeline works. It is not evidence about real homes.
- Median purchased intensity: 126 kWh/m²/yr; available for 97.8% of homes. Oil, LPG and district homes understate it (heating fuel not metered) and carry `heating_fuel_unmetered`.

## Held-out energy test predictions (`make eval-test`, deliverable I)

`eval/results/test_predictions.csv` (committed): the energy models' predictions for the NEED **test** split, scored once (`eval/eval_energy.py`; metrics in `docs/EVALUATION.md` and `eval/results/energy_test.json`).

| Column | Meaning |
|---|---|
| `need_row` | Row number (0-based) in the public NEED anonymised 50k file (`anon2025_50k.csv`); the split is in `data/splits/need_split.csv` |
| `fuel` | `gas` (gas-heated homes, weather-corrected annual gas) or `elec` (all homes, annual electricity) |
| `actual_kwh` | Metered 2023 use, valid (`V`) readings only; homes that installed a measure in 2023 are excluded |
| `v1_p10`, `v1_p50`, `v1_p90` | V1 (shipped) conformalised 80% band and median prediction for the home's actual use |
| `v1_mean` | V1 expected (mean) use, smearing-corrected: the value to sum for totals |
| `v2_ml_kwh`, `log_linear_kwh`, `cell_median_kwh`, `global_median_kwh`, `physics_scaled_kwh` | Description-only models and baselines (typical-home predictions) |
| `mode1_last_year_kwh`, `mode1_last_year_x_trend_kwh`, `mode1_change_model_l1_kwh` | "This home" mode: last year's (2022) bill, the bill x median trend, and the change model; homes without a valid 2022 reading fall back to V1 |

Every model was fitted on the train split only; V1's bands were calibrated on val. Description-only features: property type, age band, floor-area band, EPC band, region, conservatory, deprivation quintile, gas heated, cavity / PV installed before 2023.

## Value-model training data (`make train-value`, ~35 s; needs `data/raw/hpi` and the raw-derived `properties.parquet`)

`models/value_model.training_sales()` builds the training table from `sales.parquet` (results in `docs/EVALUATION.md`):

- **Rows:** matched Price Paid resales 2018–2025 with a certificate lodged **before** the sale (`has_pre_sale_epc`); new builds excluded (their premium doesn't apply to an existing home); prices ≤ £10,000 dropped (non-market transfers). 151,587 sales: 106,140 train / 22,970 val / 22,477 test, by the fixed customer split (`splits.parquet`), so all sales of a home sit on one side.
- **Target:** `log_value_now` = log(price × HPI(type, LA, Jul 2026) / HPI(type, LA, sale month)), the sale indexed to the valuation month with the UK HPI series for its Price Paid property type (detached / semi / terraced / flat) in its local authority.
- **Features** (all from the pre-sale certificate, plus deprivation): local authority, property type, floor area, bedrooms (22% unknown, left missing), age band, SAP score, EPC band, IMD quintile. Tenure is not used: it is unknown for homes that haven't sold.
- **This home:** `prior_log_value_now`, the same home's previous sale (strictly earlier) indexed to the same month; 11% of sales have one.
- **Location for the rating premium only:** the postcode sector (e.g. `SE15 4`) from the raw-derived `properties.parquet`, used as a fixed effect when estimating the EPC-band premium. It isn't stored in any shared file or model; the shipped value model holds only the band premia and their covariance.
- **SAP gain per retrofit:** SAP score regressed on the fabric fields over the real certificates of train-split customers (376,769 homes with known heating), coefficients in `value_meta.json`.

## "Worse than similar homes" detector

`engine/detector.py` compares a home's billed annual use per fuel with the p90 of V1's band for homes with the same description; `eval/eval_detector.py` checks the flag rate on the NEED test split (real metered use): 9.7% electricity / 10.2% gas, 5.6–12.5% in every subgroup (`eval/results/detector_check.json`). On the real property book the flag fires for 7% of gas and 15% of electricity bills, but those bills are synthetic, so that rate says nothing about accuracy.
