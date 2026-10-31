from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from model.features import ATTACK, load_dataset
from suricata.events_mapping import flow_df_to_features

MODEL_DIR = Path(__file__).parent / "saved-models"
TRAINING_DATA_PATH = Path(__file__).parent.parent / "datasets/sample/iot23_sample.parquet"

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

