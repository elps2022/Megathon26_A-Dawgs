# Data

## Interim synthetic dataset (use this now)

Run `make data` (or `uv run python -m greenhome.data.generate --seed 42`). It takes ~6 s, needs only `config/`, and writes CSVs to `data/generated/`, which is gitignored. The same seed always gives identical files.

The data covers 10,000 EPC-like homes in Leeds, Cornwall and Southwark, with 24 monthly bills (Oct 2024 to Sep 2026) driven by real Met Office weather.

| File | Grain | Columns / notes |
|---|---|---|
| `properties.csv` | 1 row per home | `Property` fields (ARCHITECTURE §4) plus `local_authority`, `wall_type` (solid/cavity), `has_own_roof`, `gas_meter`. The state is **at series start, before any retrofit event**. `annual_elec_kwh` / `annual_gas_kwh` are annualised from the monthly bills by `data/annualise.py` (§3.2). They're None when a fuel has no readings, and `data_flags` records low coverage or unavailability. For the ~12% of homes with a retrofit event, the annual figure mixes pre- and post-retrofit months. |
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
| `config/need_calibration.yaml` | NEED 50k (2023) | Log-linear models of annual gas and electricity by type × floor-area band × age band × EPC band; **observed retrofit effects** (difference-in-differences, 2 years each side): loft −2.5% gas, cavity wall −10.2% gas, PV −12.4% electricity |
| `config/degree_days.csv` | Met Office HadUK-Grid | Monthly HDD (base 15.5 °C, monthly-mean approximation) per LA |
| `config/price_calibration.yaml` | Price Paid 2018–2021 | Median price by LA × type |
| `config/hpi_index.csv` | UK HPI Jul 2026 | Monthly index per LA |

Consumption = the NEED model × hidden occupancy × behaviour × idiosyncratic noise (the NEED residual variance split across these), times smart-control and retrofit multipliers. This is then spread over months with real HDD. Electric heating converts gas-equivalent heat demand via efficiency (heat pump SCOP 2.6–3.3). Solar reduces grid import through a self-consumption share. Because the bills are built from the NEED-fitted model, **never report energy-model accuracy on them**: headline accuracy comes from the held-out NEED test set (ARCHITECTURE §3.5). SAP points come from a separate synthetic rule.

Prices = Price Paid median for LA × type × (area / reference area)^0.7 × EPC premium × hidden quality × HPI(date) / HPI(2018–21 mean) × noise.

Known limitations: HDD from monthly means understates shoulder months. Storage-heater bills (~10 MWh/yr) are on the high side. Simulated solar savings (~1,150 kWh/yr) are larger than NEED's observed −12% (≈ −320 kWh): NEED's PV is older and smaller, which makes a useful sanity check for the savings model. Glazing, solid wall and smart-control effects are assumptions, not observations.

## Real property book (EPC ingest)

Run `make epc` (`python -m greenhome.data.epc_ingest`, about 15 s). It needs `data/raw/epc/`, `data/raw/onspd/` and `data/raw/imd/`. Raw files are never committed (EPC and Price Paid addresses are under Royal Mail terms).

- **Input:** 731,331 domestic certificates for Leeds, Cornwall and Southwark (lodged 2012-01 to 2026-09).
- **Dedup:** the latest certificate per UPRN. Certificates without a UPRN (9,561) and park homes (303) are dropped. Result: **542,711 homes** (Leeds 253,557, Cornwall 191,954, Southwark 97,200).
- **Mapping:** free-text EPC fields map to the `Property` vocabulary with ordered rules in `config/epc_mapping.yaml`. Mapped share per field is 99.3-100% (`eval/results/epc_ingest_summary.json`). Anything assumed is listed in `data_flags`: for example `wall_insulation_assumed`, `loft_mm_assumed_from_description`, `glazing_install_date_unknown`, `gas_meter_inferred`. Fields that can't be mapped become `"unknown"` and are treated as missing (default sub-score + flag; a survey is needed before the affected retrofits).
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
| `properties.parquet` / `properties_share.parquet` | The 542,711 real homes plus `annual_elec_kwh` / `annual_gas_kwh` (annualised by `data/annualise.py`), `customer_id` and the final `smart_controls`. The share file has no postcode |
| `monthly_energy/part-*.parquet` | 12,637,637 monthly readings (Oct 2024 to Sep 2026): `elec_import_kwh`, `gas_kwh`. A missing month = absent row; gas NaN = unmatched meter |
| `loans.parquet` | 222,973 mortgages: `ltv`, plus `outstanding_balance` = LTV × value once `make value` has run (unavailable for homes with no value) |
| `splits.parquet` | train/val/test per customer (70/15/15) |
| `_hidden/latents.parquet` | **Evaluation only:** occupancy, behaviour, noise, heat-pump SCOP, unmatched-meter and missing-month counts |

- **Bills:** 537,732 homes get bills. The rest (unknown type, age, floor area or heating) are flagged `no_synthetic_bills_unknown_cell` and go down the "estimated from similar homes" or "unavailable" path. Typical medians: gas-boiler homes ≈ 7,832 kWh gas + 2,464 kWh electricity; heat pumps ≈ 4,935 kWh electricity.
- **Customers:** 518,011, including 9,905 landlords holding 34,605 homes.
- **Smart controls:** 22% smart thermostat (synthetic upgrade of programmer homes), 10% zone control, 51% programmer and 16% none.
- **No conflicts injected into real records** (see ARCHITECTURE §3.1). Use the interim synthetic dataset for conflicting-data scenarios.

### Price Paid sales joined to EPC homes (`make sales`, ~45 s, after `make epc`)

`greenhome.data.sales_join` matches Land Registry Price Paid sales (2018-2025) to EPC homes. Output: `data/processed/real/sales.parquet` (gitignored, no address or postcode). Aggregates are in `eval/results/sales_join_summary.json`.

- **Sales kept:** category A (standard market sales; B = repossessions, bulk and company sales) and residential types only. That leaves 190,371 sales in the three local authorities.
- **Matching:** within the same postcode, a sale matches an EPC home when its flat and house numbers (SAON then PAON) equal the numbers in the EPC address, in the same order. Fallbacks: the same numbers in any order, or, for sales with no numbers, the house name appearing in the EPC address. A sale matches only when exactly one UPRN fits. Unit words (Flat, Apartment) and punctuation are ignored, and "21 A" counts as "21A".
- **Match rate: 92.2%** (175,457 sales). By local authority: Leeds 95.1%, Cornwall 89.6%, Southwark 89.9%. Unmatched: 3,390 ambiguous, 346 with no EPC in the postcode, and 11,178 with no address fit.
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
