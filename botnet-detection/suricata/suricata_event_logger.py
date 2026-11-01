import json
from datetime import datetime, timezone

import pandas as pd


class SuricataEventLogger:
    # Suricata itself uses gid=1 for its own rules; we use a dedicated gid so
    # ML-raised alerts are easy to tell apart from real Suricata signature hits.
    ML_ALERT_GID = 9000000

    # attack type (9-way class, see NINEWAY in model/features.py) -> (signature id, classification).
    # Ids are fixed per class so they stay stable when the classifier is retrained.
    ATTACK_SIGNATURES = {
        "C&C": (1, "Command and Control Communication Detected"),
        "Attack": (2, "Potentially Malicious Traffic"),
        "DDoS": (3, "Detection of a Denial of Service Attack"),
        "PartOfAHorizontalPortScan": (4, "Detection of a Network Scan"),
        "PartOfAHorizontalPortScan-Attack": (5, "Detection of a Network Scan with Attack Traffic"),
        "Okiru": (6, "Okiru Botnet Activity Detected"),
        "Okiru-Attack": (7, "Okiru Botnet Attack Detected"),
        "FileDownload": (8, "Malicious File Download Detected"),
    }
    UNKNOWN_SIGNATURE = (99, "Potentially Bad Traffic")

    @classmethod
    def signature_for(cls, attack_type):
        # Signature id and classification of an attack type
        return cls.ATTACK_SIGNATURES.get(attack_type, cls.UNKNOWN_SIGNATURE)

    def __init__(self, log_file, model_name):
        self.log_file = log_file
        self.model_name = model_name

    def log_event(self, event):
        with open(self.log_file, 'a') as f:
            f.write(f"{event}\n")

    def format_fast_log_line(self, alert_row, detected_at):
        attack_type = alert_row["attack_type"]
        sid, category = self.signature_for(attack_type)
        signature = f"BOTNET DETECTED {attack_type.upper()} TRAFFIC (p={alert_row['botnet_probability']:.2f})"
        ts = detected_at.strftime("%m/%d/%Y-%H:%M:%S.%f")
        proto = (alert_row["proto"] or "").upper()
        src = f"{alert_row['src_ip']}:{int(alert_row['src_port'])}" if pd.notna(alert_row["src_port"]) else alert_row[
            "src_ip"]
        dst = f"{alert_row['dest_ip']}:{int(alert_row['dest_port'])}" if pd.notna(alert_row["dest_port"]) else \
        alert_row[
            "dest_ip"]
        return (
            f"{ts}  [**] [{self.ML_ALERT_GID}:{sid}:1] {signature} [**] "
            f"[Classification: {category}] {{{proto}}} {src} -> {dst}"
        )

    def create_eve_alert(self, alert_row, detected_at):
        attack_type = alert_row["attack_type"]
        sid, category = self.signature_for(attack_type)
        signature = f"BOTNET DETECTED {attack_type.upper()} TRAFFIC"
        flow_start = pd.Timestamp(alert_row["ts"])
        flow_start = flow_start.tz_localize(timezone.utc) if flow_start.tzinfo is None else flow_start.tz_convert(
            timezone.utc)
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
                "gid": self.ML_ALERT_GID,
                "signature_id": sid,
                "rev": 1,
                "signature": signature,
                "category": category,
            },
            "ml": {
                "model": self.model_name,
                "botnet_probability": round(float(alert_row["botnet_probability"]), 6),
                "attack_type": attack_type,
                "flow_start": flow_start.strftime("%Y-%m-%dT%H:%M:%S.%f%z"),
            },
        }

    def write_alerts(self, results, fast_log_path, eve_alert_path, df):
        alerts = results[results["is_botnet"]]
        if alerts.empty:
            return 0

        fast_log_path.parent.mkdir(parents=True, exist_ok=True)
        eve_alert_path.parent.mkdir(parents=True, exist_ok=True)

        detected_at = datetime.now().astimezone()
        with open(fast_log_path, "a") as fast_log, open(eve_alert_path, "a") as eve_log:
            for _, alert_row in alerts.iterrows():
                fast_log.write(self.format_fast_log_line(alert_row, detected_at) + "\n")
                eve_log.write(json.dumps(self.create_eve_alert(alert_row, detected_at)) + "\n")
        return len(alerts)