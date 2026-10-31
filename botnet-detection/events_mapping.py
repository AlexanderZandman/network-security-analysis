from suricata_event import Suricata_Event, Flow, TCP
import pandas as pd
from datetime import datetime, timedelta

def map_suricata_events_to_dataframe(suricata_events: list[Suricata_Event]) -> pd.DataFrame:

    # Initialize an empty list to hold the event dictionaries
    events_as_dict = []

    # Loop through each Suricata_Event object and convert it to a dictionary
    for event in suricata_events:
        if not isinstance(event, Suricata_Event):
            raise TypeError(f"Expected Suricata_Event, got {type(event)}")
        
        # Calculate duration in seconds and determine if it's a single packet flow
        end = datetime.strptime(event.flow.end, "%Y-%m-%dT%H:%M:%S.%f%z")
        start = datetime.strptime(event.flow.start, "%Y-%m-%dT%H:%M:%S.%f%z")
        duration = (end - start).total_seconds()
        is_single_pkt = 1 if duration == 0 else 0

        # Append the extracted data as a dictionary to the list
        events_as_dict.append({
            "ts": datetime.strptime(event.timestamp, "%Y-%m-%dT%H:%M:%S.%f%z"),
            "source_ip": event.src_ip,
            "dest_ip": event.dest_ip,
            "source_port": event.src_port,
            "dest_port": event.dest_port,
            "proto": event.proto,
            "flow_id": event.flow_id,
            "orig_pkts": event.flow.pkts_toserver,
            "resp_pkts": event.flow.pkts_toclient,
            "orig_ip_bytes": event.flow.bytes_toserver,
            "resp_ip_bytes": event.flow.bytes_toclient,
            "has_response": 1 if event.flow.pkts_toclient > 0 else 0,
            "duration": duration,
            "is_single_pkt": is_single_pkt
        })

    df = pd.DataFrame(events_as_dict)
    time_window = df["ts"].dt.floor("10min")
    src = df.groupby(["source_ip", time_window])
    dst = df.groupby(["dest_ip", time_window])

    return pd.DataFrame({
        "proto": df["proto"].astype("category"),
        "duration": df["duration"],
        "orig_pkts": df["orig_pkts"],
        "resp_pkts": df["resp_pkts"],
        "orig_ip_bytes": df["orig_ip_bytes"],
        "resp_ip_bytes": df["resp_ip_bytes"],
        "is_single_pkt": df["is_single_pkt"],
        "has_response": df["has_response"],
        "dst_conn_cnt_in_time_window": dst["dest_ip"].transform("size"),
        "dst_uniq_src_ip_in_time_window": dst["source_ip"].transform("nunique"),
        "src_conn_cnt_in_time_window": src["flow_id"].transform("size"),
        "src_uniq_dst_ip_in_time_window": src["dest_ip"].transform("nunique"),
        "src_uniq_dst_port_in_time_window": src["dest_port"].transform("nunique"),
        "src_no_reply_ratio_in_time_window": (df["resp_pkts"] == 0).groupby(
            [df["source_ip"], time_window]).transform("mean"),
    })
