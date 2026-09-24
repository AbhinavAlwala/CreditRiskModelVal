"""Load the CSV into SQLite and run each query in sql/risk_analysis.sql."""
from pathlib import Path
import sqlite3
import pandas as pd

ROOT = Path(__file__).resolve().parent
csv_path = ROOT / "data" / "credit_portfolio.csv"
sql_path = ROOT / "sql" / "risk_analysis.sql"

if not csv_path.exists():
    raise SystemExit("Run `python run_analysis.py` first.")

df = pd.read_csv(csv_path)
with sqlite3.connect(":memory:") as conn:
    df.to_sql("credit_portfolio", conn, index=False, if_exists="replace")
    blocks = [q.strip() for q in sql_path.read_text().split(";") if q.strip()]
    for i, query in enumerate(blocks, start=1):
        print(f"\n--- Query {i} ---")
        try:
            print(pd.read_sql_query(query, conn).to_string(index=False))
        except Exception as exc:
            print("Query failed:", exc)
