"""Draw a stratified sample of the IoT-23 conn logs that keeps every attack type.

Each class (detailed_label, or "Benign") is sampled independently with keep probability

    p = clamp(frac, lower=min_rows / n, upper=max_rows / n), capped at 1

so classes with <= min_rows rows are kept entirely (rare attacks never disappear), big
classes are thinned to `frac`, and huge classes are capped at roughly `max_rows`.
Rows are picked by hashing (scenario, uid, seed), so the sample is reproducible.
"""

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
SRC_GLOB = str(ROOT / "datasets/iot23_parquet/*.parquet")
OUT_PATH = ROOT / "datasets/sample/iot23_sample.parquet"

BASE_SQL = """
SELECT
    * EXCLUDE (label, detailed_label, filename),
    regexp_extract(filename, '([^/]+)\\.parquet$', 1) AS scenario,
    lower(label) AS label,
    coalesce(detailed_label, 'Benign') AS detailed_label
FROM read_parquet(?, filename = true)
WHERE label IS NOT NULL
"""


def class_sizes(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return con.execute(
        f"SELECT detailed_label, count(*) AS class_n FROM ({BASE_SQL}) GROUP BY 1", [SRC_GLOB]
    ).df()


def sample_iot23(
    frac: float = 0.01,
    min_rows: int = 1_000,
    max_rows: int = 500_000,
    seed: int = 42,
    memory_limit: str = "8GB",
) -> pd.DataFrame:
    con = duckdb.connect()
    con.sql(f"SET memory_limit = '{memory_limit}'")
    con.sql("SET preserve_insertion_order = false")

    # Pass 1: tiny per-class table with keep probabilities.
    probs = class_sizes(con)
    n = probs["class_n"]
    probs["p"] = np.minimum(1.0, np.maximum(min_rows / n, np.minimum(frac, max_rows / n)))
    con.register("probs", probs)

    # Pass 2: one streaming scan, filtered against the small lookup table.
    return con.execute(
        f"""
        SELECT b.*
        FROM ({BASE_SQL}) b JOIN probs USING (detailed_label)
        WHERE (hash(b.scenario, b.uid, ?) % 1000000000) / 1e9 < probs.p
        """,
        [SRC_GLOB, seed],
    ).df()


if __name__ == "__main__":
    df = sample_iot23()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"{len(df):,} rows written to {OUT_PATH}")
    print(df["detailed_label"].value_counts().to_string())
