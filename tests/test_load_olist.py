"""Focused regression tests for scripts/load_olist.py.

No test framework required; run with the project environment:

    .venv/bin/python tests/test_load_olist.py
"""

import sqlite3
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import load_olist  # noqa: E402


def test_zip_code_prefix_keeps_leading_zero():
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "customers.csv").write_text(
            "customer_id,customer_zip_code_prefix\n" "aaa,01320\n" "bbb,14409\n"
        )
        real_raw_dir = load_olist.RAW_DIR
        load_olist.RAW_DIR = Path(tmp)
        try:
            conn = sqlite3.connect(":memory:")
            try:
                count = load_olist.load_table(conn, "customers.csv", "customers")
                rows = conn.execute(
                    "SELECT customer_zip_code_prefix,"
                    " typeof(customer_zip_code_prefix)"
                    " FROM customers ORDER BY customer_id"
                ).fetchall()
            finally:
                conn.close()
        finally:
            load_olist.RAW_DIR = real_raw_dir
    assert count == 2
    assert rows == [("01320", "text"), ("14409", "text")]


if __name__ == "__main__":
    test_zip_code_prefix_keeps_leading_zero()
    print("all tests passed")
