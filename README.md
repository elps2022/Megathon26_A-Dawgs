# Green Home Score & Retrofit Recommendation Engine

Megathon X, Problem Statement 2.

## Setup

Requires Python 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                      # creates .venv with all dependencies (dev group adds pytest)
```

## Data and trained models (hosted externally)

Datasets and model weights are not in this repository. Both zips are in this Google Drive folder: <https://drive.google.com/drive/folders/1Ss5zbpSmv5ZF5RBw2gXpmxL4ENdWJbH3>. Download them, then install with one command each:

| What | File in the folder | Install |
|---|---|---|
| Trained models (energy V1 / V2 / mode 1 with bands, value model), 4.3 MB | `energy_models.zip` | `make models SRC=path/to/energy_models.zip` |
| Processed real data (property book, loans, splits, features, matched sales, monthly bills), 105 MB | `real_data.zip` | `make real-data SRC=path/to/real_data.zip` |

`SRC` also accepts a direct Google Drive file link ("Anyone with the link") instead of a downloaded zip. The processed data contains no postcodes or address fields.

Without either download, the synthetic dataset can be generated locally (no downloads needed):

```bash
make data                    # data/generated/*.csv, seed 42, ~6 s
```

## Run

```bash
make app                     # Streamlit dashboard (customer view + lender portfolio)
make api                     # FastAPI on http://localhost:8000 (docs at /docs)
make test                    # test suite
```

## Rebuild everything from the raw public data (optional)

Download the raw files listed under Credits into `data/raw/` (layout in `docs/DATA.md`), then:

```bash
make calibrate               # calibration tables in config/ from the raw files
make build                   # real data layer: EPC ingest, synthetic bills, sales join, valuation, labels, features
make train                   # energy models (5-fold CV + validation), saved to src/greenhome/models/artifacts/
make train-value             # value model, saved to the same folder
make eval-test               # single test-set scoring of the energy models, savings checks, figures
make eval-value              # single test-set scoring of the value model
make package-models          # dist/energy_models.zip      (the hosted models zip)
make package-data            # dist/real_data.zip          (the hosted data zip)
```

## Credits

**Datasets** (all Open Government Licence v3.0):
- Energy Performance of Buildings Register, domestic certificates for Leeds, Cornwall and Southwark: Ministry of Housing, Communities and Local Government, <https://get-energy-performance-data.communities.gov.uk>. Address fields: Royal Mail and Ordnance Survey copyright (never redistributed).
- National Energy Efficiency Data-Framework (NEED) anonymised data 2025: Department for Energy Security and Net Zero, <https://www.gov.uk/government/statistics/national-energy-efficiency-data-framework-need-anonymised-data-2025>.
- Price Paid Data 2018–2025: HM Land Registry, <https://www.gov.uk/government/statistical-data-sets/price-paid-data-downloads>. Contains HM Land Registry data © Crown copyright and database right 2026.
- UK House Price Index, July 2026: HM Land Registry and ONS, <https://www.gov.uk/government/statistical-data-sets/uk-house-price-index-data-downloads-july-2026>.
- HadUK-Grid regional monthly mean temperature: Met Office, <https://www.metoffice.gov.uk/research/climate/maps-and-data/uk-and-regional-series>.
- ONS Postcode Directory, August 2026: Office for National Statistics. Contains OS data © Crown copyright and database right; Royal Mail data © Royal Mail copyright and database right.
- English Indices of Deprivation 2019: Ministry of Housing, Communities and Local Government.
- Greenhouse gas reporting conversion factors 2026: Department for Energy Security and Net Zero, <https://www.gov.uk/government/publications/greenhouse-gas-reporting-conversion-factors-2026>.

**Libraries:** pandas, NumPy, PyArrow, scikit-learn, LightGBM, CatBoost, Optuna, Pydantic, PyYAML, odfpy, openpyxl, FastAPI, Uvicorn, Streamlit, httpx, matplotlib, pytest, uv.

## Licence

MIT, see [LICENSE](LICENSE).
