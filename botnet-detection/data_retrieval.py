"""Convert the labeled IoT-23 Zeek conn logs into a Parquet dataset, chunk by chunk.

The raw pcaps carry no labels; the labels live in each scenario's bro/conn.log.labeled.
Files are streamed in chunks so memory stays bounded regardless of file size.
"""

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).parent
SRC = ROOT / "botnet-detection/datasets/opt/Malware-Project/BigDataset/IoTScenarios"
OUT = ROOT / "botnet-detection/datasets/iot23_parquet"
CHUNK_ROWS = 1_000_000

COLUMNS = [
    "ts", "uid", "id.orig_h", "id.orig_p", "id.resp_h", "id.resp_p", "proto", "service",
    "duration", "orig_bytes", "resp_bytes", "conn_state", "local_orig", "local_resp",
    "missed_bytes", "history", "orig_pkts", "orig_ip_bytes", "resp_pkts", "resp_ip_bytes",
    "tunnel_parents_label",  # last three fields are space-separated, split below
]
NUMERIC = [
    "ts", "id.orig_p", "id.resp_p", "duration", "orig_bytes", "resp_bytes", "missed_bytes",
    "orig_pkts", "orig_ip_bytes", "resp_pkts", "resp_ip_bytes",
]


def read_conn_log(path: Path, chunksize: int = CHUNK_ROWS):
    """Yield DataFrame chunks from one conn.log.labeled file."""
    reader = pd.read_csv(
        path,
        sep="\t",
        comment="#",
        header=None,
        names=COLUMNS,
        dtype=str,
        chunksize=chunksize,
    )
    for chunk in reader:
        tail = chunk.pop("tunnel_parents_label").str.split(r"\s+", n=2, expand=True)
        chunk["tunnel_parents"] = tail[0]
        chunk["label"] = tail[1]
        chunk["detailed_label"] = tail[2]
        chunk = chunk.replace("-", pd.NA)
        for col in NUMERIC:
            # float64 everywhere so every chunk shares one schema, even when a chunk has no NaNs
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce").astype("float64")
        yield chunk


def convert(log_path: Path, out_path: Path):
    writer = None
    rows = 0
    try:
        for chunk in read_conn_log(log_path):
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(out_path, table.schema, compression="zstd")
            writer.write_table(table.cast(writer.schema))
            rows += len(chunk)
    finally:
        if writer is not None:
            writer.close()
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for log_path in sorted(SRC.glob("*/bro/conn.log.labeled")):
        scenario = log_path.parent.parent.name
        out_path = OUT / f"{scenario}.parquet"
        if out_path.exists():
            print("skip", scenario)
            continue
        rows = convert(log_path, out_path)
        print(f"wrote {out_path.name}: {rows:,} rows")


if __name__ == "__main__":
    main()
