import json
from pathlib import Path

import joblib
import numpy as np 
from xgboost import XGBClassifier
import xgboost as xgb
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedGroupKFold
from xgboost import XGBClassifier

from features import ATTACK, BENIGN_LABEL, FEATURE_TIME_WINDOW, NINEWAY, create_features_df, load_dataset

MODEL_DIR = Path(__file__).parent / "saved-models"
TESTING_SAMPLED_DATA_PATH = Path(__file__).parent / "data/CTU-IoT-Malware-Capture-17-1.parquet"

def main():
    test_df = load_dataset(TESTING_SAMPLED_DATA_PATH).reset_index(drop=True)
    features_df = create_features_df(test_df, time_window="1min")
    
    total_number_of_rows = len(test_df)
    print(f"Total number of rows in the test set: {total_number_of_rows}")
    total_number_of_malicious_rows = test_df["label"].ne("benign").sum()
    print(f"Total number of malicious rows in the test set: {total_number_of_malicious_rows}")
    total_number_of_benign_rows = test_df["label"].eq("benign").sum()
    print(f"Total number of benign rows in the test set: {total_number_of_benign_rows}")

    print(f"Percentage of malicious rows in the test set: {total_number_of_malicious_rows / total_number_of_rows * 100:.2f}%")
    print(f"Percentage of benign rows in the test set: {total_number_of_benign_rows / total_number_of_rows * 100:.2f}% \n")

    botnet_detector = joblib.load(MODEL_DIR / "botnet_detector.json")
    binary_predictions = botnet_detector.predict(features_df)
    print(f"Predicted labels for {len(binary_predictions)} rows: {binary_predictions[:10]}")

    print(f"Classification report for the binary classifier:")
    # Get the rows where the binary classifier predicted malicious (1)
    predicted_malicious_rows_indices = features_df.index[binary_predictions == 1]
    # Get the rows where the binary classifier predicted benign (0)
    predicted_benign_rows_indices = features_df.index[binary_predictions == 0]

    predicted_malicious_rows = test_df.loc[predicted_malicious_rows_indices]
    predicted_benign_rows = test_df.loc[predicted_benign_rows_indices]
    # Get how many of the predicted malicious rows are actually malicious (true positives) and how many are benign (false positives)
    correctly_classified_as_malicious = predicted_malicious_rows["label"].ne("benign").sum()
    print(f"Number of true positives (malicious rows correctly classified as malicious): {correctly_classified_as_malicious}")
    incorrectly_classified_as_malicious = predicted_malicious_rows["label"].eq("benign").sum()
    print(f"Number of false positives (benign rows incorrectly classified as malicious): {incorrectly_classified_as_malicious}")

    # Get how many of the predicted benign rows are actually benign (true negatives) and how many are malicious (false negatives)
    correctly_classified_as_benign = predicted_benign_rows["label"].eq("benign").sum()
    print(f"Number of true negatives (benign rows correctly classified as benign): {correctly_classified_as_benign}")
    incorrectly_classified_as_benign = predicted_benign_rows["label"].ne("benign").sum()
    print(f"Number of false negatives (malicious rows incorrectly classified as benign): {incorrectly_classified_as_benign}")

    # Calculate precision and recall for both classes automatically.
    true_labels = test_df["label"].ne("benign").astype(int)
    print(classification_report(true_labels, binary_predictions, target_names=["benign", "malicious"], digits=4, zero_division=0,))

    # Get the features for the predicted botnet rows
    features_df = create_features_df(test_df.loc[predicted_malicious_rows_indices], time_window="1min")

    # Load the class names in the same integer order used by train.py.
    attack_class_names = json.loads((MODEL_DIR / "attack_classes.json").read_text())
    attack_classifier = joblib.load(MODEL_DIR / "attack_classifier.json")
    predicted_labels = attack_classifier.predict(features_df)
    predicted_attack_labels = pd.Series(predicted_labels, index=predicted_malicious_rows.index).map(
        dict(enumerate(attack_class_names))
    )
    true_attack_labels = predicted_malicious_rows["detailed_label"].map(NINEWAY)

    # The attack classifier was trained only on malicious rows. Exclude benign
    # false positives from this attack-type report.
    attack_rows = true_attack_labels.notna() & predicted_malicious_rows["label"].ne("benign")
    print("Attack-type classification report for correctly labelled malicious rows:")
    print(classification_report(true_attack_labels[attack_rows], predicted_attack_labels[attack_rows], labels=attack_class_names, target_names=attack_class_names, digits=4, zero_division=0))
    print(pd.crosstab(true_attack_labels[attack_rows], predicted_attack_labels[attack_rows], rownames=["true"], colnames=["predicted"]))

if __name__ == "__main__":
    main()