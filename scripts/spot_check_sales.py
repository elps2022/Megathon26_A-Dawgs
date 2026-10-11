"""Print 10 random matched sales next to their EPC address, to eyeball the Price Paid <-> EPC join.

    PYTHONPATH=src .venv/bin/python scripts/spot_check_sales.py      (needs data/raw/ and make sales)
"""
from pathlib import Path
import pandas as pd
from greenhome.data.calibrate import PPD_COLS
from greenhome.data.sales_join import property_id

s = pd.read_parquet("data/processed/real/sales.parquet").sample(10)
ppd = pd.concat(pd.read_csv(f, header=None, names=PPD_COLS, dtype=str) for f in sorted(Path("data/raw/ppd").glob("pp-*.csv")))
ppd["sale_id"] = ppd.tid.str.strip("{}")
epc = pd.read_csv("data/raw/epc/epc_domestic_leeds_cornwall_southwark.csv", usecols=["uprn", "address"], low_memory=False)
epc = epc.dropna(subset=["uprn"]).drop_duplicates("uprn", keep="last")
epc["property_id"] = property_id(epc.uprn)
m = s.merge(ppd[["sale_id", "saon", "paon", "street", "postcode"]], on="sale_id").merge(epc[["property_id", "address"]], on="property_id")
pd.set_option("display.width", 200)
print(m[["match_tier", "saon", "paon", "street", "postcode", "address", "price", "ppd_property_type", "property_type"]].to_string(index=False))
