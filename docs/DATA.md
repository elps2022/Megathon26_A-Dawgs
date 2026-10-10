# Data

## Interim synthetic dataset (use this now)

Run `make data` (or `uv run python -m greenhome.data.generate --seed 42`). It takes ~6 s, needs only `config/`, and writes CSVs to `data/generated/`, which is gitignored. The same seed always gives identical files.

The data covers 10,000 EPC-like homes in Leeds, Cornwall and Southwark, with 24 monthly bills (Oct 2024 to Sep 2026) driven by real Met Office weather.

| File | Grain | Columns / notes |
|---|---|---|
| `properties.csv` | 1 row per home | `Property` fields (ARCHITECTURE §4) plus `local_authority`, `wall_type` (solid/cavity), `has_own_roof`, `gas_meter`. The state is **at series start, before any retrofit event**. `annual_elec_kwh` / `annual_gas_kwh` are not included: compute them from monthly data (§3.2). `data_flags` is empty, so the pipeline fills it. |
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

Consumption = the NEED model × hidden occupancy × behaviour × idiosyncratic noise (the NEED residual variance split across these), times smart-control and retrofit multipliers. This is then spread over months with real HDD. Electric heating converts gas-equivalent heat demand via efficiency (heat pump SCOP 2.6–3.3). Solar reduces grid import through a self-consumption share. The generator **does not use the grey-box physics tables**, so evaluating V1 against it is not circular. SAP points come from a separate synthetic rule.

Prices = Price Paid median for LA × type × (area / reference area)^0.7 × EPC premium × hidden quality × HPI(date) / HPI(2018–21 mean) × noise.

Known limitations: HDD from monthly means understates shoulder months. Storage-heater bills (~10 MWh/yr) are on the high side. Simulated solar savings (~1,150 kWh/yr) are larger than NEED's observed −12% (≈ −320 kWh): NEED's PV is older and smaller, which makes a useful sanity check for the savings model. Glazing, solid wall and smart-control effects are assumptions, not observations.

## Real data (planned)

See ARCHITECTURE §3.1. Raw files go in `data/raw/` and are never committed (EPC and Price Paid addresses are under Royal Mail terms).
