"""Load the 9 Olist CSVs from data/raw/olist/ into data/olist.sqlite.

Usage:
    python scripts/load_olist.py

Rebuilds the database from scratch on every run, so it is safe to rerun.
"""

import os
import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw" / "olist"
DB_PATH = ROOT / "data" / "olist.sqlite"

# CSV filename -> table name. The generic rule is to drop the "olist_" prefix
# and "_dataset" suffix; the category translation file is named explicitly.
TABLES = {
    "olist_customers_dataset.csv": "customers",
    "olist_geolocation_dataset.csv": "geolocation",
    "olist_order_items_dataset.csv": "order_items",
    "olist_order_payments_dataset.csv": "order_payments",
    "olist_order_reviews_dataset.csv": "order_reviews",
    "olist_orders_dataset.csv": "orders",
    "olist_products_dataset.csv": "products",
    "olist_sellers_dataset.csv": "sellers",
    "product_category_name_translation.csv": "product_category_translation",
}

INDEX_COLUMNS = ("order_id", "customer_id", "product_id", "seller_id")

ZIP_CODE_SUFFIX = "zip_code_prefix"


def read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV, keeping *zip_code_prefix columns as text.

    The suffix match is generic: any present or future zip code column keeps
    its original 5-digit values (e.g. "01320") instead of being parsed as an
    integer. All other columns keep pandas' inferred types.
    """
    header = pd.read_csv(path, nrows=0, encoding="utf-8-sig").columns
    dtype = {col: "string" for col in header if col.endswith(ZIP_CODE_SUFFIX)}
    return pd.read_csv(path, encoding="utf-8-sig", dtype=dtype)


def load_table(conn: sqlite3.Connection, csv_name: str, table: str) -> int:
    """Load one CSV into its SQLite table as-is. Returns the row count."""
    df = read_csv(RAW_DIR / csv_name)

    # Store timestamps as ISO text: stringify any datetime columns so they
    # land in SQLite as TEXT, leaving the raw string values untouched.
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.strftime("%Y-%m-%d %H:%M:%S")

    df.to_sql(table, conn, if_exists="replace", index=False)

    for col in INDEX_COLUMNS:
        if col in df.columns:
            conn.execute(
                f'CREATE INDEX IF NOT EXISTS "idx_{table}_{col}" '
                f'on "{table}" ("{col}")'
            )
    conn.commit()

    (count,) = conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()
    return count


def main() -> None:
    if DB_PATH.exists():
        os.remove(DB_PATH)
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    try:
        for csv_name, table in TABLES.items():
            count = load_table(conn, csv_name, table)
            print(f"{table}: {count} rows")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
