"""
download_datasets.py — fetch diverse public datasets for pipeline testing.

Datasets downloaded
-------------------
Domain          Source  Name                               Folder
-----------     ------  --------------------------------   ---------------------------
Customer        UCI     Bank Marketing (41k customers)     data/raw/bank_marketing/
Campaign        UCI     Bank Marketing (same file)         (same folder, dual-purpose)
E-commerce      UCI     Online Retail II (1M transactions) data/raw/online_retail/
Sensor / IoT    UCI     Air Quality UCI (9k readings)      data/raw/air_quality/
Sensor / IoT    UCI     Occupancy Detection (20k records)  data/raw/occupancy/
Product spend   UCI     Wholesale Customers (440 buyers)   data/raw/wholesale_customers/
Customer seg.   Kaggle  Mall Customers (200 customers)     data/raw/mall_customers/
Employee / HR   Kaggle  IBM HR Attrition (1.5k employees)  data/raw/ibm_hr/

Usage
-----
    python download_datasets.py              # download everything
    python download_datasets.py --list       # list datasets without downloading
    python download_datasets.py --only uci   # only UCI datasets (no Kaggle key needed)
    python download_datasets.py --only kaggle
"""
from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import textwrap
import zipfile
from pathlib import Path

import requests

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

# ── Colour helpers ────────────────────────────────────────────────────────────

def _green(s): return f"\033[32m{s}\033[0m"
def _yellow(s): return f"\033[33m{s}\033[0m"
def _red(s):   return f"\033[31m{s}\033[0m"
def _bold(s):  return f"\033[1m{s}\033[0m"


# ── README writer ─────────────────────────────────────────────────────────────

def _write_readme(folder: Path, title: str, source: str, description: str,
                  entity_col: str, use_case: str) -> None:
    readme = folder / "README.md"
    readme.write_text(textwrap.dedent(f"""\
        # {title}

        **Source**: {source}
        **Entity column**: `{entity_col}`
        **Use case for this pipeline**: {use_case}

        ## Description
        {description}
    """))


# ─────────────────────────────────────────────────────────────────────────────
#  UCI downloads (via ucimlrepo — no API key needed)
# ─────────────────────────────────────────────────────────────────────────────

def _uci(dataset_id: int, folder_name: str, title: str, entity_col: str,
         use_case: str, description: str) -> bool:
    folder = RAW_DIR / folder_name
    if any(folder.glob("*.csv")):
        print(f"  {_yellow('skip')}  {title} (already downloaded)")
        return True
    try:
        from ucimlrepo import fetch_ucirepo
        print(f"  {_bold('fetch')} {title} (UCI #{dataset_id}) …", end=" ", flush=True)
        ds = fetch_ucirepo(id=dataset_id)
        df = ds.data.original if hasattr(ds.data, "original") else ds.data.features
        folder.mkdir(parents=True, exist_ok=True)
        out = folder / f"{folder_name}.csv"
        df.to_csv(out, index=False)
        _write_readme(folder, title, f"UCI ML Repository (id={dataset_id})",
                      description, entity_col, use_case)
        print(_green(f"✓  {len(df):,} rows → {out.name}"))
        return True
    except Exception as exc:
        print(_red(f"✗  {exc}"))
        return False


def download_uci_datasets() -> list[str]:
    results = []

    ok = _uci(
        dataset_id=222,
        folder_name="bank_marketing",
        title="Bank Marketing (Customer + Campaign)",
        entity_col="client ID (row index)",
        use_case="Cluster customers by demographics and campaign-response behaviour",
        description=(
            "41,188 records from a Portuguese bank's telemarketing campaigns. "
            "Features: age, job, marital status, education, balance, call duration, "
            "number of contacts, previous campaign outcome, subscription outcome (y)."
        ),
    )
    results.append("bank_marketing" if ok else "bank_marketing:FAILED")

    ok = _uci(
        dataset_id=352,
        folder_name="online_retail",
        title="Online Retail II (E-commerce transactions)",
        entity_col="CustomerID",
        use_case="Aggregate per-customer RFM features (recency, frequency, monetary) "
                 "to cluster buyer personas",
        description=(
            "1,067,371 transactions from a UK online retailer (2009-2011). "
            "Each row is one invoice line: CustomerID, StockCode, Description, "
            "Quantity, UnitPrice, Country."
        ),
    )
    results.append("online_retail" if ok else "online_retail:FAILED")

    ok = _uci(
        dataset_id=360,
        folder_name="air_quality",
        title="Air Quality UCI (Sensor data)",
        entity_col="sensor location (implicit, single site)",
        use_case="Cluster hourly sensor reading patterns across the day/week",
        description=(
            "9,358 hourly readings from 5 chemical sensors in an Italian city (2004-2005). "
            "Features: CO, NMHC, NOx, NO2, O3 sensor responses, temperature, "
            "relative humidity, absolute humidity."
        ),
    )
    results.append("air_quality" if ok else "air_quality:FAILED")

    ok = _uci(
        dataset_id=357,
        folder_name="occupancy",
        title="Occupancy Detection (IoT / Sensor)",
        entity_col="Timestamp (time-windowed rows)",
        use_case="Cluster building occupancy patterns by environmental sensor profile",
        description=(
            "20,560 readings from a room equipped with environmental sensors. "
            "Features: Temperature, Humidity, Light, CO2, HumidityRatio, Occupancy "
            "(ground truth label)."
        ),
    )
    results.append("occupancy" if ok else "occupancy:FAILED")

    ok = _uci(
        dataset_id=292,
        folder_name="wholesale_customers",
        title="Wholesale Customers (Product categories)",
        entity_col="row index (one row per wholesale customer)",
        use_case="Cluster wholesale buyers by annual spend across 6 product categories",
        description=(
            "440 wholesale customers of a Portuguese distributor. "
            "Features: Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicassen "
            "annual spend (monetary units). Classic product-preference segmentation benchmark."
        ),
    )
    results.append("wholesale_customers" if ok else "wholesale_customers:FAILED")

    return results


# ─────────────────────────────────────────────────────────────────────────────
#  Kaggle downloads
# ─────────────────────────────────────────────────────────────────────────────

def _kaggle_available() -> bool:
    try:
        from kaggle import KaggleApi  # noqa: F401
        return True
    except Exception:
        return False


def _kaggle_download(slug: str, folder_name: str, title: str,
                     expected_file: str, entity_col: str,
                     use_case: str, description: str) -> bool:
    folder = RAW_DIR / folder_name
    if (folder / expected_file).exists():
        print(f"  {_yellow('skip')}  {title} (already downloaded)")
        return True
    folder.mkdir(parents=True, exist_ok=True)
    print(f"  {_bold('fetch')} {title} (Kaggle: {slug}) …", end=" ", flush=True)
    try:
        from kaggle import KaggleApi
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(slug, path=str(folder), unzip=True, quiet=True)
        files = list(folder.glob("**/*.csv"))
        if not files:
            files = list(folder.glob("**/*.json")) + list(folder.glob("**/*.parquet"))
        count = sum(1 for _ in files)
        _write_readme(folder, title, f"Kaggle ({slug})", description, entity_col, use_case)
        print(_green(f"✓  {count} file(s) downloaded"))
        return True
    except Exception as exc:
        print(_red(f"✗  {exc}"))
        shutil.rmtree(folder, ignore_errors=True)
        return False


def download_kaggle_datasets() -> list[str]:
    if not _kaggle_available():
        print(_red("  Kaggle package not importable — skipping Kaggle datasets."))
        return []

    results = []

    ok = _kaggle_download(
        slug="vjchoudhary7/customer-segmentation-tutorial-in-python",
        folder_name="mall_customers",
        title="Mall Customers (Customer segmentation)",
        expected_file="Mall_Customers.csv",
        entity_col="CustomerID",
        use_case="Benchmark: small (200 customers), clean, perfect for quick pipeline tests",
        description=(
            "200 mall customers with Age, Annual Income (k$), Spending Score (1-100). "
            "Classic segmentation benchmark — great for sanity-checking cluster quality."
        ),
    )
    results.append("mall_customers" if ok else "mall_customers:FAILED")

    ok = _kaggle_download(
        slug="pavansubhasht/ibm-hr-analytics-attrition-dataset",
        folder_name="ibm_hr",
        title="IBM HR Analytics (Employee attrition)",
        expected_file="WA_Fn-UseC_-HR-Employee-Attrition.csv",
        entity_col="EmployeeNumber",
        use_case="Cluster employee profiles by HR features; detect attrition-risk personas",
        description=(
            "1,470 employee records with 35 features: age, department, job role, "
            "satisfaction scores, years at company, attrition label, etc."
        ),
    )
    results.append("ibm_hr" if ok else "ibm_hr:FAILED")

    return results


# ─────────────────────────────────────────────────────────────────────────────
#  Summary
# ─────────────────────────────────────────────────────────────────────────────

def _print_summary(all_results: list[str]) -> None:
    print()
    print(_bold("=" * 60))
    print(_bold("  Dataset Download Summary"))
    print(_bold("=" * 60))

    rows = []
    for r in all_results:
        if r.endswith(":FAILED"):
            name = r[:-7]
            rows.append((_red("FAILED"), name))
        else:
            folder = RAW_DIR / r
            csvs = list(folder.glob("*.csv"))
            size = sum(f.stat().st_size for f in csvs) / 1024 / 1024 if csvs else 0
            rows.append((_green("  OK  "), f"{r:<24} {size:.1f} MB"))

    for status, detail in rows:
        print(f"  {status}  {detail}")

    print()
    print("All datasets saved under:", _bold(str(RAW_DIR.resolve())))
    print()
    print(_bold("Next steps:"))
    print("  • Each folder has a README.md describing the entity column and use case.")
    print("  • To run the pipeline on a different dataset, point config.yaml")
    print("    data_path to the new CSV and adjust entity_id_col.")
    print("  • Online Retail and Bank Marketing need feature-engineering before")
    print("    clustering — the feature_engineer agent handles this automatically.")


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Download public datasets for pipeline testing")
    parser.add_argument("--list", action="store_true", help="List datasets without downloading")
    parser.add_argument("--only", choices=["uci", "kaggle"], help="Download only one source")
    args = parser.parse_args()

    if args.list:
        print(_bold("\nDatasets available for download:\n"))
        datasets = [
            ("UCI",    "bank_marketing",   "Bank Marketing — 41k customers, campaign responses"),
            ("UCI",    "online_retail",    "Online Retail II — 1M e-commerce transactions"),
            ("UCI",    "air_quality",      "Air Quality — 9k IoT/chemical sensor readings"),
            ("UCI",    "occupancy",           "Occupancy Detection — 20k environmental sensor rows"),
            ("UCI",    "wholesale_customers", "Wholesale Customers — 440 buyers × 6 product categories"),
            ("Kaggle", "mall_customers",      "Mall Customers — 200 rows, quick sanity check"),
            ("Kaggle", "ibm_hr",              "IBM HR Analytics — 1.5k employee profiles"),
        ]
        for src, folder, desc in datasets:
            status = _green("downloaded") if any((RAW_DIR / folder).glob("*.csv")) else "not downloaded"
            print(f"  [{src:6s}] {folder:<22} {desc}  ({status})")
        print()
        return

    all_results: list[str] = []

    if args.only != "kaggle":
        print(_bold("\n── UCI ML Repository datasets ─────────────────────────────"))
        all_results += download_uci_datasets()

    if args.only != "uci":
        print(_bold("\n── Kaggle datasets ────────────────────────────────────────"))
        all_results += download_kaggle_datasets()

    _print_summary(all_results)


if __name__ == "__main__":
    main()
