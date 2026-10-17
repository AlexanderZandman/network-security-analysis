import numpy as np
import pandas as pd

from model.features import create_features_df
from suricata.suricata_event import SuricataEvent


def map_suricata_events_to_flow_df(suricata_events: list[SuricataEvent], scenario: str = "suricata-eve") -> pd.DataFrame:
    rows = []
    for event in suricata_events:
        if not isinstance(event, SuricataEvent):
            raise TypeError(f"Expected SuricataEvent, got {type(event)}")
        if not event.is_flow:
            continue

        flow = event.flow
        rows.append({
            "ts": flow.start or event.timestamp,
            "end": flow.end,
            "scenario": scenario,
            "uid": str(event.flow_id),
            "id.orig_h": event.src_ip,
            "id.orig_p": event.src_port,
            "id.resp_h": event.dest_ip,
            "id.resp_p": event.dest_port,
            "proto": (event.proto or "").lower() or None,
            "orig_pkts": flow.pkts_toserver,
            "resp_pkts": flow.pkts_toclient,
            "orig_ip_bytes": flow.bytes_toserver,
            "resp_ip_bytes": flow.bytes_toclient,
            "history": None,  # no Suricata equivalent of Zeek's TCP history
        })

    columns = ["ts", "end", "scenario", "uid", "id.orig_h", "id.orig_p", "id.resp_h", "id.resp_p",
               "proto", "orig_pkts", "resp_pkts", "orig_ip_bytes", "resp_ip_bytes", "history"]
    df = pd.DataFrame(rows, columns=columns)

    df["ts"] = pd.to_datetime(df["ts"], utc=True, format="ISO8601", errors="coerce")
    end = pd.to_datetime(df.pop("end"), utc=True, format="ISO8601", errors="coerce")

    df["duration"] = (end - df["ts"]).dt.total_seconds()
    # Zeek leaves duration empty for single-packet connections; the is_single_pkt feature relies on that
    df.loc[df["orig_pkts"] + df["resp_pkts"] <= 1, "duration"] = np.nan

    return df.dropna(subset=["ts"]).reset_index(drop=True)


def map_suricata_events_to_features_data(suricata_events: list[SuricataEvent], categories: dict = None,
                                         scenario: str = "suricata-eve") -> pd.DataFrame:
    flow_df = map_suricata_events_to_flow_df(suricata_events, scenario=scenario)
    return flow_df_to_features(flow_df, categories)


def flow_df_to_features(flow_df: pd.DataFrame, categories: dict = None) -> pd.DataFrame:
    flow_df = flow_df.copy()
    if categories is not None:
        flow_df["proto"] = pd.Categorical(flow_df["proto"], categories=categories["proto"])
        flow_df["history"] = pd.Categorical(flow_df["history"], categories=categories["history"])
    return create_features_df(flow_df)
