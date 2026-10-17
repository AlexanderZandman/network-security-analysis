import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from suricata.events_mapping import flow_df_to_features
from model.features import ATTACK, load_dataset

MODEL_DIR = Path(__file__).parent / "saved-models"
TRAINING_DATA_PATH = Path(__file__).parent.parent / "datasets/sample/iot23_sample.parquet"

# Suricata itself uses gid=1 for its own rules; we use a dedicated gid so
# ML-raised alerts are easy to tell apart from real Suricata signature hits.
ML_ALERT_GID = 9000000
ATTACK_SIGNATURE_ID = {"c2": 1, "attack": 2, "scanning": 3}

ATTACK_CATEGORY = {
    "c2": "Command and Control Communication Detected",
    "attack": "Potentially Malicious Traffic",
    "scanning": "Attempted Reconnaissance (Scanning)",
}


def load_models():
    botnet_detector = joblib.load(MODEL_DIR / "botnet_detector.pkl")
    attack_classifier = joblib.load(MODEL_DIR / "attack_classifier.pkl")
    return botnet_detector, attack_classifier


def load_training_reference(path=TRAINING_DATA_PATH):
    df = load_dataset(path)
    y = (df["label"] == "malicious").astype(int)
    attack_bot = df.loc[y == 1, "detailed_label"].map(ATTACK)
    classes = np.array(sorted(attack_bot.unique()))
    categories = {
        "proto": pd.Index(sorted(df["proto"].dropna().unique())),
        "history": pd.Index(sorted(df["history"].dropna().unique())),
    }
    return categories, classes


def predict(flow_df, botnet_detector, attack_classifier, categories, attack_classes, threshold=0.5,
            context_df=None):
    # If context_df is provided, it is used to compute features but not included in the output.
    # This is useful for computing features that depend on previous flows, e.g. the "is_single_pkt" feature.
    n_context = 0 if context_df is None else len(context_df)
    all_flows = flow_df if context_df is None else pd.concat([context_df, flow_df], ignore_index=True)
    features_df = flow_df_to_features(all_flows, categories).iloc[n_context:]

    botnet_proba = botnet_detector.predict_proba(features_df)[:, 1]
    is_botnet = botnet_proba > threshold

    attack_type = np.full(len(features_df), None, dtype=object)
    if is_botnet.any():
        codes = attack_classifier.predict(features_df[is_botnet])
        attack_type[is_botnet] = attack_classes[codes]

    results = pd.DataFrame({
        "uid": flow_df["uid"].values,
        "ts": flow_df["ts"].values,
        "src_ip": flow_df["id.orig_h"].values,
        "src_port": flow_df["id.orig_p"].values,
        "dest_ip": flow_df["id.resp_h"].values,
        "dest_port": flow_df["id.resp_p"].values,
        "proto": flow_df["proto"].values,
        "botnet_probability": botnet_proba,
        "is_botnet": is_botnet,
        "attack_type": attack_type,
    })
    return results


def format_fast_log_line(alert_row, detected_at):
    attack_type = alert_row["attack_type"]
    sid = ATTACK_SIGNATURE_ID[attack_type]
    category = ATTACK_CATEGORY[attack_type]
    signature = f"ML BOTNET DETECTED {attack_type.upper()} TRAFFIC (p={alert_row['botnet_probability']:.2f})"
    ts = detected_at.strftime("%m/%d/%Y-%H:%M:%S.%f")
    proto = (alert_row["proto"] or "").upper()
    src = f"{alert_row['src_ip']}:{int(alert_row['src_port'])}" if pd.notna(alert_row["src_port"]) else alert_row[
        "src_ip"]
    dst = f"{alert_row['dest_ip']}:{int(alert_row['dest_port'])}" if pd.notna(alert_row["dest_port"]) else alert_row[
        "dest_ip"]
    return (
        f"{ts}  [**] [{ML_ALERT_GID}:{sid}:1] {signature} [**] "
        f"[Classification: {category}] {{{proto}}} {src} -> {dst}"
    )


def create_eve_alert(alert_row, detected_at):
    attack_type = alert_row["attack_type"]
    sid = ATTACK_SIGNATURE_ID[attack_type]
    category = ATTACK_CATEGORY[attack_type]
    signature = f"ML BOTNET DETECTED {attack_type.upper()} TRAFFIC"
    flow_start = pd.Timestamp(alert_row["ts"])
    flow_start = flow_start.tz_localize(timezone.utc) if flow_start.tzinfo is None else flow_start.tz_convert(timezone.utc)
    return {
        "timestamp": detected_at.strftime("%Y-%m-%dT%H:%M:%S.%f%z"),
        "flow_id": alert_row["uid"],
        "event_type": "alert",
        "src_ip": alert_row["src_ip"],
        "src_port": None if pd.isna(alert_row["src_port"]) else int(alert_row["src_port"]),
        "dest_ip": alert_row["dest_ip"],
        "dest_port": None if pd.isna(alert_row["dest_port"]) else int(alert_row["dest_port"]),
        "proto": (alert_row["proto"] or "").upper(),
        "alert": {
            "action": "allowed",
            "gid": ML_ALERT_GID,
            "signature_id": sid,
            "rev": 1,
            "signature": signature,
            "category": category,
        },
        "ml": {
            "model": "botnet_detector+attack_classifier",
            "botnet_probability": round(float(alert_row["botnet_probability"]), 6),
            "attack_type": attack_type,
            "flow_start": flow_start.strftime("%Y-%m-%dT%H:%M:%S.%f%z"),
        },
    }


def write_alerts(results, fast_log_path, eve_alert_path):
    alerts = results[results["is_botnet"]]
    if alerts.empty:
        return 0

    fast_log_path.parent.mkdir(parents=True, exist_ok=True)
    eve_alert_path.parent.mkdir(parents=True, exist_ok=True)

    detected_at = datetime.now().astimezone()
    with open(fast_log_path, "a") as fast_log, open(eve_alert_path, "a") as eve_log:
        for _, alert_row in alerts.iterrows():
            fast_log.write(format_fast_log_line(alert_row, detected_at) + "\n")
            eve_log.write(json.dumps(create_eve_alert(alert_row, detected_at)) + "\n")
    return len(alerts)
