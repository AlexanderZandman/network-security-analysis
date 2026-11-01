import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from model.features import BENIGN_LABEL, FEATURE_TIME_WINDOW
from suricata.events_mapping import flow_df_to_features

MODEL_DIR = Path(__file__).parent / "saved-models"
ATTACK_CLASSES_PATH = MODEL_DIR / "attack_classes.json"
EVERYTHING_CLASSES_PATH = MODEL_DIR / "everything_classes.json"

def _load_class_names(path):
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Retrain the models with model/train.py.")
    return np.array(json.loads(path.read_text()))


def load_models():
    """Load the botnet detector, the attack classifier and the class name of every attack classifier output."""
    botnet_detector = joblib.load(MODEL_DIR / "botnet_detector.pkl")
    attack_classifier = joblib.load(MODEL_DIR / "attack_classifier.pkl")
    attack_classes = _load_class_names(ATTACK_CLASSES_PATH)
    if len(attack_classes) != attack_classifier.n_classes_:
        raise ValueError(
            f"attack_classifier.pkl outputs {attack_classifier.n_classes_} classes but {len(attack_classes)} class "
            f"names are known ({attack_classes.tolist()}). Retrain it with model/train.py so "
            f"{ATTACK_CLASSES_PATH.name} is written."
        )
    return botnet_detector, attack_classifier, attack_classes


def load_everything_model():
    """Load the single-stage classifier and the class name of every output index."""
    classifier = joblib.load(MODEL_DIR / "everything_classifier.pkl")
    classes = _load_class_names(EVERYTHING_CLASSES_PATH)
    if len(classes) != classifier.n_classes_:
        raise ValueError(
            f"everything_classifier.pkl outputs {classifier.n_classes_} classes but {len(classes)} class names "
            f"are known ({classes.tolist()}). Retrain it with model/train.py so {EVERYTHING_CLASSES_PATH.name} is written."
        )
    if BENIGN_LABEL not in classes:
        raise ValueError(f"everything classifier has no '{BENIGN_LABEL}' class: {classes.tolist()}")
    return classifier, classes


def load_categories(model):
    stored = model.get_booster().get_categories(export_to_arrow=True).to_arrow()
    return {name: pd.Index(values.to_pylist()) for name, values in stored if values is not None}


def _features(flow_df, categories, context_df=None):
    # If context_df is provided, it is used to compute features but not included in the output.
    # This is useful for computing features that depend on previous flows, e.g. the "is_single_pkt" feature.
    n_context = 0 if context_df is None else len(context_df)
    all_flows = flow_df if context_df is None else pd.concat([context_df, flow_df], ignore_index=True)
    return flow_df_to_features(all_flows, FEATURE_TIME_WINDOW, categories).iloc[n_context:]


def _results_df(flow_df, botnet_proba, is_botnet, attack_type):
    return pd.DataFrame({
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


def predict(flow_df, botnet_detector, attack_classifier, categories, attack_classes, threshold=0.5,
            context_df=None):
    """Two-stage prediction: botnet detector, then attack classifier on the flagged flows."""
    features_df = _features(flow_df, categories, context_df)
    botnet_proba = botnet_detector.predict_proba(features_df)[:, 1]
    is_botnet = botnet_proba > threshold

    attack_type = np.full(len(flow_df), None, dtype=object)
    if is_botnet.any():
        codes = attack_classifier.predict(features_df[is_botnet])
        attack_type[is_botnet] = attack_classes[codes]

    return _results_df(flow_df, botnet_proba, is_botnet, attack_type)


def predict_everything(flow_df, everything_classifier, categories, everything_classes, threshold=0.5,
                       context_df=None):
    """Single-stage prediction: one classifier over benign + every attack type."""
    features_df = _features(flow_df, categories, context_df)

    proba = everything_classifier.predict_proba(features_df)
    benign_idx = int(np.flatnonzero(everything_classes == BENIGN_LABEL)[0])
    botnet_proba = 1.0 - proba[:, benign_idx]
    is_botnet = botnet_proba > threshold

    attack_proba = proba.copy()
    attack_proba[:, benign_idx] = -np.inf
    attack_type = np.full(len(features_df), None, dtype=object)
    attack_type[is_botnet] = everything_classes[attack_proba[is_botnet].argmax(axis=1)]

    return _results_df(flow_df, botnet_proba, is_botnet, attack_type)

