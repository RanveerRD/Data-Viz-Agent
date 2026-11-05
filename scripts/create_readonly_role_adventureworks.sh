#!/usr/bin/env bash
# Create (or update) a read-only Postgres role for AdventureWorks and verify it.
#
# Usage:
#   scripts/create_readonly_role_adventureworks.sh
#
# Reads ANALYST_READER_PASSWORD from .env (that one line only, never sourced
# whole, never printed, never passed as a CLI argument).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$ROOT/.env"
DB="adventureworks"
ROLE="analyst_reader"
TABLE="sales.salesorderheader"

command -v psql >/dev/null 2>&1 || { echo "ERROR: psql not found" >&2; exit 1; }
[ -f "$ENV_FILE" ] || { echo "ERROR: .env not found at $ENV_FILE" >&2; exit 1; }

# Read only the one needed line; keep everything after the first '=' so
# passwords containing '=' survive. Do not source the file.
ANALYST_READER_PASSWORD="$(grep -E '^ANALYST_READER_PASSWORD=' "$ENV_FILE" | cut -d= -f2- | tail -n 1 | tr -d '\r' || true)"
# Strip one pair of matching surrounding quotes, if present.
first_char="${ANALYST_READER_PASSWORD:0:1}"
last_char="${ANALYST_READER_PASSWORD: -1}"
if [ "${#ANALYST_READER_PASSWORD}" -ge 2 ] && [ "$first_char" = '"' ] && [ "$last_char" = '"' ]; then
  ANALYST_READER_PASSWORD="${ANALYST_READER_PASSWORD:1:${#ANALYST_READER_PASSWORD}-2}"
elif [ "${#ANALYST_READER_PASSWORD}" -ge 2 ] && [ "$first_char" = "'" ] && [ "$last_char" = "'" ]; then
  ANALYST_READER_PASSWORD="${ANALYST_READER_PASSWORD:1:${#ANALYST_READER_PASSWORD}-2}"
fi
if [ -z "${ANALYST_READER_PASSWORD:-}" ]; then
  echo "ERROR: ANALYST_READER_PASSWORD is missing or empty in .env" >&2
  exit 1
fi
export ANALYST_READER_PASSWORD
trap 'unset ANALYST_READER_PASSWORD PGPASSWORD' EXIT

# --- Upsert the role (idempotent) -------------------------------------------
ROLE_EXISTS="$(psql -d "$DB" -tAc "SELECT 1 FROM pg_roles WHERE rolname='$ROLE'" || true)"
if [ "$ROLE_EXISTS" = "1" ]; then
  psql -d "$DB" -v ON_ERROR_STOP=1 <<'EOSQL'
\getenv pw ANALYST_READER_PASSWORD
ALTER ROLE analyst_reader WITH LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE PASSWORD :'pw';
EOSQL
else
  psql -d "$DB" -v ON_ERROR_STOP=1 <<'EOSQL'
\getenv pw ANALYST_READER_PASSWORD
CREATE ROLE analyst_reader WITH LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE PASSWORD :'pw';
EOSQL
fi

psql -d "$DB" -v ON_ERROR_STOP=1 <<'EOSQL'
ALTER ROLE analyst_reader SET default_transaction_read_only = on;
ALTER ROLE analyst_reader SET statement_timeout = '10s';
ALTER ROLE analyst_reader SET idle_in_transaction_session_timeout = '30s';
EOSQL

# --- Grants ------------------------------------------------------------------
psql -d "$DB" -v ON_ERROR_STOP=1 -c "GRANT CONNECT ON DATABASE \"$DB\" TO \"$ROLE\";"

SCHEMAS="$(psql -d "$DB" -tAc "SELECT nspname FROM pg_namespace WHERE nspname NOT LIKE 'pg\\_%' AND nspname <> 'information_schema' AND nspname NOT LIKE 'pg_temp%';")"
for schema in $SCHEMAS; do
  psql -d "$DB" -v ON_ERROR_STOP=1 -c "GRANT USAGE ON SCHEMA \"$schema\" TO \"$ROLE\";"
  psql -d "$DB" -v ON_ERROR_STOP=1 -c "GRANT SELECT ON ALL TABLES IN SCHEMA \"$schema\" TO \"$ROLE\";"
  psql -d "$DB" -v ON_ERROR_STOP=1 -c "ALTER DEFAULT PRIVILEGES IN SCHEMA \"$schema\" GRANT SELECT ON TABLES TO \"$ROLE\";"
done

# --- Checks as analyst_reader -------------------------------------------------
FAILS=0

check_pass() { echo "$1: PASS - $2"; }
check_fail() { echo "$1: FAIL - $2"; FAILS=$((FAILS + 1)); }

# 1. Login + SELECT (must pass before the rest are meaningful).
if out="$(PGPASSWORD="$ANALYST_READER_PASSWORD" psql -U "$ROLE" -d "$DB" -tAc "SELECT count(*) FROM $TABLE;" 2>&1)"; then
  check_pass "1. SELECT count(*) from $TABLE" "returned $out"
else
  check_fail "1. SELECT count(*) from $TABLE" "login or SELECT failed"
  echo "Aborting: check 1 must pass first." >&2
  exit 1
fi

# Helper: expect the SQL to be denied with SQLSTATE 42501 or 25006.
expect_denied() {
  local label="$1"
  local sql="$2"
  local out
  out="$(PGPASSWORD="$ANALYST_READER_PASSWORD" psql -U "$ROLE" -d "$DB" -v VERBOSITY=verbose -v ON_ERROR_STOP=0 -c "$sql" 2>&1 || true)"
  if echo "$out" | grep -Eq '42501|25006|permission denied|read-only transaction|read-only SQL transaction'; then
    check_pass "$label" "denied as expected"
  else
    first_line="$(echo "$out" | head -n 3 | tr '\n' ' ')"
    check_fail "$label" "expected SQLSTATE 42501/25006, got: $first_line"
  fi
}

# 2. INSERT denied.
expect_denied "2. INSERT denied" "BEGIN; INSERT INTO $TABLE DEFAULT VALUES; ROLLBACK;"

# 3. UPDATE and DELETE denied (both must be denied).
expect_denied "3a. UPDATE denied" "BEGIN; UPDATE $TABLE SET SalesOrderID = SalesOrderID WHERE SalesOrderID = -1; ROLLBACK;"
expect_denied "3b. DELETE denied" "BEGIN; DELETE FROM $TABLE WHERE SalesOrderID = -1; ROLLBACK;"

# 4. CREATE TABLE denied.
expect_denied "4. CREATE TABLE denied" "CREATE TABLE analyst_reader_test_should_fail (id integer);"

# 5. SET read-only off, then INSERT still denied by grants.
expect_denied "5. SET read-only off + INSERT still denied" "SET default_transaction_read_only = off; BEGIN; INSERT INTO $TABLE DEFAULT VALUES; ROLLBACK;"

# 6. Long query cancelled by statement_timeout (expect SQLSTATE 57014).
start=$SECONDS
out6="$(PGPASSWORD="$ANALYST_READER_PASSWORD" psql -U "$ROLE" -d "$DB" -v VERBOSITY=verbose -v ON_ERROR_STOP=0 -c "SELECT pg_sleep(15);" 2>&1 || true)"
elapsed=$((SECONDS - start))
if echo "$out6" | grep -Eq '57014|statement timeout|canceling statement|cancelling statement'; then
  if [ "$elapsed" -lt 15 ]; then
    check_pass "6. pg_sleep(15) cancelled" "cancelled after ~${elapsed}s"
  else
    check_fail "6. pg_sleep(15) cancelled" "timeout error but took ${elapsed}s"
  fi
else
  first_line="$(echo "$out6" | head -n 3 | tr '\n' ' ')"
  check_fail "6. pg_sleep(15) cancelled" "expected timeout (57014), got: $first_line"
fi

# 7. No write privileges via has_table_privilege (permission layer, independent
# of default_transaction_read_only).
for priv in INSERT UPDATE DELETE TRUNCATE; do
  if priv_out="$(psql -d "$DB" -tAc "SELECT has_table_privilege('analyst_reader','$TABLE','$priv');" 2>&1)"; then
    priv_trimmed="$(echo "$priv_out" | tr -d '[:space:]')"
    if [ "$priv_trimmed" = "f" ]; then
      check_pass "7. has_table_privilege $priv is false" "returned f"
    else
      check_fail "7. has_table_privilege $priv is false" "returned $priv_trimmed"
    fi
  else
    check_fail "7. has_table_privilege $priv is false" "query failed"
  fi
done

if [ "$FAILS" -gt 0 ]; then
  echo "$FAILS check(s) FAILED" >&2
  exit 1
fi
echo "All checks PASSED"
