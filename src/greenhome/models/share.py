"""Share the trained energy models without the raw data (model weights are hosted outside the repo).

Training (`make labels && make train`) needs the raw NEED file, which only the data owner downloads. Everyone
else installs the trained artifacts instead:

    uv run python -m greenhome.models.share pack                  -> dist/energy_models.zip   (make package-models)
    uv run python -m greenhome.models.share install <zip | URL>   -> src/greenhome/models/artifacts/   (make models SRC=...)
    uv run python -m greenhome.models.share pack-data             -> dist/real_data.zip        (make package-data)
    uv run python -m greenhome.models.share install-data <zip | URL> -> data/processed/real/   (make real-data SRC=...)

The data bundle holds only the shareable processed files: no postcodes or address fields (Royal Mail / OS terms
cover those), and none of the hidden synthetic ground truth (_hidden/), which must never be used as features.

A URL can be a direct download link or a Google Drive share link ("Anyone with the link"). Only the expected
model files are extracted (LightGBM text models and meta.json, no paths), and the install is checked by loading
every model.
"""

from __future__ import annotations

import io
import json
import re
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from greenhome.models.energy import ARTIFACTS, ROOT

DIST = ROOT / "dist" / "energy_models.zip"
DATA_DIST = ROOT / "dist" / "real_data.zip"
REAL = ROOT / "data" / "processed" / "real"
DATA_FILES = ["properties_share.parquet", "loans.parquet", "splits.parquet", "features.parquet", "sales.parquet",
              "valuations.parquet"]
SAFE_DATA = re.compile(r"^(monthly_energy/)?[a-z0-9_-]+\.parquet$")
SAFE_NAME = re.compile(r"^[a-z0-9_]+\.(txt|json)$")


def pack(out: Path = DIST) -> Path:
    files = sorted(f for f in ARTIFACTS.glob("*") if SAFE_NAME.match(f.name))
    if not any(f.name == "meta.json" for f in files):
        raise SystemExit(f"No trained models in {ARTIFACTS}: run `make labels && make train` first (needs data/raw/need).")
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, f.name)
    print(f"{out} ({out.stat().st_size / 1e6:.1f} MB, {len(files)} files). Upload it (e.g. Google Drive, 'Anyone with "
          f"the link') or send it; teammates run: make models SRC=<file or link>")
    return out


def _drive_direct(url: str) -> str:
    m = re.search(r"drive\.google\.com/(?:file/d/|open\?id=|uc\?(?:export=download&)?id=)([\w-]+)", url)
    return f"https://drive.google.com/uc?export=download&id={m.group(1)}" if m else url


def _read(src: str) -> bytes:
    if re.match(r"^https?://", src):
        with urllib.request.urlopen(_drive_direct(src), timeout=60) as r:
            return r.read()
    return Path(src).expanduser().read_bytes()


def install(src: str) -> None:
    data = _read(src)
    if not zipfile.is_zipfile(io.BytesIO(data)):
        raise SystemExit("That isn't a zip file. For Google Drive, share the file as 'Anyone with the link'.")
    with zipfile.ZipFile(io.BytesIO(data)) as z, tempfile.TemporaryDirectory() as tmp:
        names = [n for n in z.namelist() if SAFE_NAME.match(n)]
        if "meta.json" not in names:
            raise SystemExit("The zip has no meta.json: make it with `make package-models`.")
        for n in names:
            (Path(tmp) / n).write_bytes(z.read(n))
        meta = json.loads((Path(tmp) / "meta.json").read_text())
        from greenhome.models.energy import GreyboxModel, MLModel
        GreyboxModel(Path(tmp)), MLModel(Path(tmp))   # every referenced model file loads
        if "value_meta.json" in names:
            import lightgbm as lgb
            for f in json.loads((Path(tmp) / "value_meta.json").read_text())["files"].values():
                lgb.Booster(model_file=str(Path(tmp) / f))
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        for n in names:
            shutil.copy(Path(tmp) / n, ARTIFACTS / n)
    print(f"Installed {len(names)} files to {ARTIFACTS} (V1 gas val MAE {meta['greybox']['gas']['val']['mae_kwh']}).")


def pack_data(out: Path = DATA_DIST) -> Path:
    files = [REAL / f for f in DATA_FILES] + sorted((REAL / "monthly_energy").glob("*.parquet"))
    missing = [f.name for f in files if not f.exists()]
    if missing:
        raise SystemExit(f"Missing {missing}: run `make build` first (needs data/raw/).")
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:     # parquet is already compressed
        for f in files:
            z.write(f, str(f.relative_to(REAL)))
    print(f"{out} ({out.stat().st_size / 1e6:.0f} MB, {len(files)} files: no postcodes, no hidden ground truth). "
          f"Teammates run: make real-data SRC=<file or link>")
    return out


def install_data(src: str) -> None:
    data = _read(src)
    if not zipfile.is_zipfile(io.BytesIO(data)):
        raise SystemExit("That isn't a zip file. For Google Drive, share the file as 'Anyone with the link'.")
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        names = [n for n in z.namelist() if SAFE_DATA.match(n)]
        if "properties_share.parquet" not in names:
            raise SystemExit("The zip has no properties_share.parquet: make it with `make package-data`.")
        (REAL / "monthly_energy").mkdir(parents=True, exist_ok=True)
        for n in names:
            (REAL / n).write_bytes(z.read(n))
    print(f"Installed {len(names)} files to {REAL}.")


def main() -> None:
    args = sys.argv[1:]
    if args[:1] == ["pack"]:
        pack()
    elif args[:1] == ["install"] and len(args) == 2:
        install(args[1])
    elif args[:1] == ["pack-data"]:
        pack_data()
    elif args[:1] == ["install-data"] and len(args) == 2:
        install_data(args[1])
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
