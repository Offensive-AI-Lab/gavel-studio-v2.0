"""One-off bootstrap script: create the central server's database if it
doesn't exist yet. Connects to the default `postgres` admin DB to issue
the CREATE.

Run once after installing Postgres locally:
    python central-server/_create_db.py
"""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
import psycopg2

# Load the central server's .env so we use the same DATABASE_URL the
# server itself will use at runtime.
load_dotenv(Path(__file__).resolve().parent / ".env")

url = os.environ.get("DATABASE_URL")
if not url:
    sys.exit("DATABASE_URL not set — fill in central-server/.env first.")

# Parse the target DB name off the URL; we connect to `postgres` to run
# CREATE DATABASE (you can't CREATE DATABASE from inside the DB you're
# creating).
from urllib.parse import urlparse
parsed = urlparse(url)
target_db = parsed.path.lstrip("/")
admin_url = url.replace(f"/{target_db}", "/postgres", 1)

# NOTE: do NOT use `with psycopg2.connect(...)` here. The context manager
# implicitly wraps the body in a transaction, and CREATE DATABASE can't
# run inside a transaction block. We open + close manually with
# autocommit set before any work.
admin_conn = psycopg2.connect(admin_url)
admin_conn.autocommit = True
try:
    with admin_conn.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (target_db,))
        if cur.fetchone():
            print(f"[create_db] '{target_db}' already exists - nothing to do.")
        else:
            cur.execute(f'CREATE DATABASE "{target_db}"')
            print(f"[create_db] Created database '{target_db}'.")
finally:
    admin_conn.close()

# Enable citext on the target DB. CREATE EXTENSION is transaction-safe,
# so a regular connection works fine here.
target_conn = psycopg2.connect(url)
target_conn.autocommit = True
try:
    with target_conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS citext")
        print(f"[create_db] citext extension ready on '{target_db}'.")
finally:
    target_conn.close()
