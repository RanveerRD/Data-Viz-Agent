#!/usr/bin/env bash
# Build the AdventureWorks sample database in local Postgres using
# lorint's AdventureWorks-for-Postgres scripts. Everything downloaded goes
# in data/raw/adventureworks/ (git-ignored via data/).
#
# Usage:
#   scripts/load_adventureworks.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AW_DIR="$ROOT/data/raw/adventureworks"
DB="adventureworks"

ZIP_URL="https://github.com/Microsoft/sql-server-samples/releases/download/adventureworks/AdventureWorks-oltp-install-script.zip"
INSTALL_SQL_URL="https://raw.githubusercontent.com/lorint/AdventureWorks-for-Postgres/master/install.sql"
UPDATE_CSVS_URL="https://raw.githubusercontent.com/lorint/AdventureWorks-for-Postgres/master/update_csvs.rb"

# 1. Create the download folder if missing.
mkdir -p "$AW_DIR"

# 2. Download what is not already there.
[ -f "$AW_DIR/AdventureWorks-oltp-install-script.zip" ] || curl -fSL -o "$AW_DIR/AdventureWorks-oltp-install-script.zip" "$ZIP_URL"
[ -f "$AW_DIR/install.sql" ] || curl -fSL -o "$AW_DIR/install.sql" "$INSTALL_SQL_URL"
[ -f "$AW_DIR/update_csvs.rb" ] || curl -fSL -o "$AW_DIR/update_csvs.rb" "$UPDATE_CSVS_URL"

# 3. Unzip only the CSV files, flattened, into the same folder
# (-o overwrites without prompting so the script is rerunnable).
unzip -j -o -q "$AW_DIR/AdventureWorks-oltp-install-script.zip" '*.csv' -d "$AW_DIR"

# 4. Fix up the CSVs for Postgres (must run from that folder).
cd "$AW_DIR"
ruby update_csvs.rb

# 5. Recreate the database.
dropdb --if-exists "$DB"
createdb "$DB"

# 6. Install schema and data (relative ./X.csv paths need this folder).
psql -d "$DB" < install.sql

# 7. Report table counts per schema and key row counts.
psql -d "$DB" -c "SELECT schemaname AS schema, COUNT(*) AS tables FROM pg_tables WHERE schemaname IN ('humanresources', 'person', 'production', 'purchasing', 'sales') GROUP BY 1 ORDER BY 1;"
psql -d "$DB" -c "SELECT 'sales.salesorderheader' AS table_name, COUNT(*) AS rows FROM sales.salesorderheader UNION ALL SELECT 'sales.salesorderdetail', COUNT(*) FROM sales.salesorderdetail UNION ALL SELECT 'sales.customer', COUNT(*) FROM sales.customer UNION ALL SELECT 'production.product', COUNT(*) FROM production.product UNION ALL SELECT 'person.person', COUNT(*) FROM person.person;"
