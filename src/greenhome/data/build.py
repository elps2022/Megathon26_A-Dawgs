"""Real data layer in one command (ARCHITECTURE 3.1): clean + join the real data, then the synthetic parts.

    uv run python -m greenhome.data.build --seed 42      (make build; needs the raw downloads in data/raw/)

Steps, in dependency order:
  1 epc_ingest       EPC certificates -> property book, IMD join          data/processed/epc_properties*.parquet
  2 synthetic_parts  bills, customer_id, smart controls, LTV, splits      data/processed/real/
  3 sales_join       Price Paid sales -> EPC homes, pre-sale features     data/processed/real/sales.parquet
  4 valuation        current_value_estimate + loan balances               data/processed/real/ (in place)
  5 need_labels      observed per-home retrofit before/after (NEED)       data/processed/need_retrofit_labels.parquet
  6 features         stage 2: parity model inputs, weather-normalised use, data/processed/real/features.parquet
                     intensities, coverage

Calibration tables (`make calibrate`) are a separate step because their outputs are committed to config/.
The interim synthetic dataset (`make data`, data/generated/) needs no raw data and is unchanged.
"""

from __future__ import annotations

import argparse
import time

from greenhome.data import epc_ingest, features, need_labels, sales_join, synthetic_parts, valuation


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=42)
    seed = ap.parse_args().seed
    steps = [("epc_ingest", epc_ingest.main),
             ("synthetic_parts", lambda: synthetic_parts.main(["--seed", str(seed)])),
             ("sales_join", sales_join.main),
             ("valuation", valuation.main),
             ("need_labels", need_labels.main),
             ("features", lambda: features.main([]))]
    for name, step in steps:
        t0 = time.time()
        print(f"== {name}", flush=True)
        step()
        print(f"== {name} done in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
