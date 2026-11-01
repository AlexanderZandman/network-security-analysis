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

from model.features import ATTACK, BENIGN_LABEL, FEATURE_TIME_WINDOW, NINEWAY, create_features_df

MODEL_DIR = Path(__file__).parent / "saved-models"
TRAINING_SAMPLED_DATA_PATH = "datasets/combined/combined-training-trimmed.parquet"

# Label mapping for the attack classifier
ATK_LABEL_MAP = {"attack": 0, "scanning": 1, "c2": 2}
ATK_LABEL_NAMES = {0: "attack", 1: "scanning", 2: "c2",}


def train_botnet_detector(df, features_df, categorized_labels):
    # y = 1 if the row is a botnet row, 0 if the row is a benign row
    y = categorized_labels.ne("benign").astype(int)

    print(f"Training botnet detector for {y.sum()} botnet rows and {len(y) - y.sum()} benign rows")
    botnet_detector = XGBClassifier(
        n_estimators=800,
        max_depth=5,
        learning_rate=0.05,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=5.0,
        reg_alpha=0.1,
        tree_method="hist",
        enable_categorical=True,
        eval_metric="aucpr",
        random_state=50,
        objective="binary:logistic",
        max_delta_step=1
    )

    cross_validation = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
    oof = np.zeros(len(df))
    for train_idx, test_idx in cross_validation.split(df, y, df["scenario"]):
        train = features_df.loc[train_idx]
        #TODO: needs to be undersampled
        test = features_df.loc[test_idx]
        botnet_detector.fit(train, y.loc[train_idx])
        oof[test_idx] = botnet_detector.predict_proba(test)[:, 1]

    botnet_detector.fit(features_df, y)
    print(f"Botnet detector trained")
    print("Botnet vs Benign Classification Report: ")
    print(f"ROC-AUC = {roc_auc_score(y, oof):.4f}")
    print(f"PR-AUC = {average_precision_score(y, oof):.4f}")
    print(classification_report(y, oof > 0.5, target_names=["benign", "botnet"], digits=4))

    # Get the predicted classes and the rows that are predicted to be botnet
    predicted_classes = (oof > 0.5).astype(int)
    print(f"Number of predicted botnet classes: {predicted_classes.sum()}")
    botnet_indices = np.where(predicted_classes == 1)[0]
    predicted_botnet_rows = features_df.iloc[botnet_indices]
    print(f"Number of predicted botnet rows: {len(predicted_botnet_rows)}")
    print(f"First 10 predicted botnet rows:\n{predicted_botnet_rows.head(10).to_dict(orient='records')}")

    return botnet_detector, predicted_botnet_rows

def _fit_multiclass_cv(classifier, features_df, y, weights, groups):
    cross_validation = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
    oof = np.full(len(y), -1, dtype=np.int64)

    for train_idx, test_idx in cross_validation.split(features_df, y, groups):
        # Rare classes can be missing from a training fold (folds are grouped by scenario), and XGBoost
        # needs the labels to be 0..k-1, so remap to the classes present in this fold and map back after.
        fold_classes = np.unique(y.iloc[train_idx])
        classifier.fit(
            features_df.loc[train_idx],
            np.searchsorted(fold_classes, y.iloc[train_idx]),
            sample_weight=weights.iloc[train_idx]
        )
        oof[test_idx] = fold_classes[classifier.predict(features_df.loc[test_idx]).astype(np.int64)]

    # Final model on all rows, so it knows every class
    classifier.fit(features_df, y, sample_weight=weights)
    return oof


def _print_multiclass_report(title, y, oof, class_names):
    names = np.array(class_names)
    print(title)
    print(classification_report(y.to_numpy(dtype=np.int64), oof, labels=np.arange(len(names)),
                                target_names=names, digits=4, zero_division=0))
    print(pd.crosstab(names[y.to_numpy(dtype=np.int64)], names[oof], rownames=["true"], colnames=["predicted"]))


def _nineway_labels(detailed_labels):
    labels = detailed_labels.map(NINEWAY)
    unknown = detailed_labels[labels.isna()]
    assert unknown.empty, (f"{len(unknown):,} row(s) with a detailed_label not in NINEWAY: "
                           f"{unknown.value_counts(dropna=False).to_dict()}")
    return labels


def _encode(labels):
    class_names = sorted(labels.unique())
    y = labels.map({name: i for i, name in enumerate(class_names)}).reset_index(drop=True)
    counts = y.value_counts()
    weights = y.map(np.sqrt(counts.max() / counts))
    print(f"{len(class_names)} classes with counts: {labels.value_counts().to_dict()}")
    return class_names, y, weights


def train_attack_classifier(df, features_df, categorized_labels, detailed_labels):
    # Only the botnet rows: the attack classifier runs on the flows the botnet detector flags,
    # so it covers the 9-way classes without Benign
    # num_class is inferred from y on every fit, since a fold can miss rare classes
    botnet_mask = categorized_labels.ne("benign")
    filtered_df = df.loc[botnet_mask].copy().reset_index(drop=True)
    # features_df is computed over all flows and only then filtered to the botnet rows: at prediction time
    # the window counts include the benign flows too, since it is not known which flows are botnet
    features_df = features_df.loc[botnet_mask].reset_index(drop=True)

    print("Training attack classifier")
    attack_class_names, y_attack, attack_weights = _encode(_nineway_labels(detailed_labels.loc[botnet_mask]))
    attack_classifier = XGBClassifier(
        n_estimators=800,
        max_depth=5,
        learning_rate=0.05,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.6,
        reg_lambda=2.0,
        reg_alpha=0.1,
        tree_method="hist",
        enable_categorical=True,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=50
    )

    oof = _fit_multiclass_cv(attack_classifier, features_df, y_attack, attack_weights, filtered_df["scenario"])
    print("Attack classifier trained")
    _print_multiclass_report("Attack Type Classification Report: ", y_attack, oof, attack_class_names)

    return attack_classifier, attack_class_names


def train_everything_classifier(df, features_df, detailed_labels):
    # Single-stage 9-way classifier: Benign + every attack class
    # num_class is inferred from y on every fit, since a fold can miss rare classes
    print("Training everything classifier")
    everything_class_names, y_multi, all_weights = _encode(_nineway_labels(detailed_labels))
    assert BENIGN_LABEL in everything_class_names, f"no {BENIGN_LABEL} rows in the training data"
    everything_classifier = XGBClassifier(
        n_estimators=800,
        max_depth=5,
        learning_rate=0.05,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        reg_alpha=0.1,
        tree_method="hist",
        enable_categorical=True,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=50
    )

    oof = _fit_multiclass_cv(everything_classifier, features_df, y_multi, all_weights, df["scenario"])
    print("Everything classifier trained")
    _print_multiclass_report("Direct Classification Report: ", y_multi, oof, everything_class_names)

    return everything_classifier, everything_class_names

# Save the trained models to disk and if not provided, skip it.
def save_models(botnet_detector = None, attack_classifier = None, everything_classifier = None,
                attack_classes = None, everything_classes = None):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if botnet_detector is not None:
        joblib.dump(botnet_detector, MODEL_DIR / "botnet_detector.pkl")
    if attack_classifier is not None:
        joblib.dump(attack_classifier, MODEL_DIR / "attack_classifier.pkl")
    if attack_classes is not None:
        (MODEL_DIR / "attack_classes.json").write_text(json.dumps(list(attack_classes)))
    if everything_classifier is not None:
        joblib.dump(everything_classifier, MODEL_DIR / "everything_classifier.pkl")
    if everything_classes is not None:
        (MODEL_DIR / "everything_classes.json").write_text(json.dumps(list(everything_classes)))
    print(f"models saved to {MODEL_DIR}")

def test_attack_classifier(df, categorized_labels, labels_vector, predicted_rows_indices):
    # False positive and false negative analysis
    # False negative: benign rows that are predicted as botnet
    selected_labels = categorized_labels.loc[predicted_rows_indices]
    benign_rows = selected_labels.eq("benign").sum()
    malicious_rows = selected_labels.ne("benign").sum()    
    total_number_of_benign_rows = categorized_labels.eq("benign").sum()
    total_number_of_malicious_rows = categorized_labels.ne("benign").sum()

    print(f"Number of benign rows in the predicted botnet rows: {benign_rows}")
    print(f"Number of malicious rows in the predicted botnet rows: {malicious_rows}")
    print(f"Total number of benign rows in the dataset: {total_number_of_benign_rows}")
    print(f"Total number of malicious rows in the dataset: {total_number_of_malicious_rows}")
    print(f"Percentage of malicious rows caught by the binary classifier: {malicious_rows / total_number_of_malicious_rows * 100:.2f}%")
    print(f"Percentage of benign rows falsely classified as malicious by the binary classifier: {benign_rows / total_number_of_benign_rows * 100:.2f}%")

    # Get the features for the predicted botnet rows
    features_df = create_features_df(df.loc[predicted_rows_indices], time_window="1min")

    # Load the trained attack classifier model
    model = xgb.XGBClassifier()
    model.load_model(MODEL_DIR / "attack_classifier.json")

    # Get the predicted labels for the predicted botnet rows
    predicted_labels = model.predict(features_df)
    print(f"Predicted labels for {len(predicted_labels)} predicted botnet rows: {predicted_labels[:10]}")
    predicted_labels = pd.Series(predicted_labels).map(ATK_LABEL_NAMES)
    print(f"Classification report for the predicted botnet rows:")
    print(classification_report(categorized_labels.loc[predicted_rows_indices], predicted_labels, digits=4, zero_division=0))
    print(pd.crosstab(categorized_labels.loc[predicted_rows_indices], predicted_labels, rownames=["true"], colnames=["predicted"]))

def main():
    # Positional cross-validation indices are used as labels with .loc, so the index must be 0..n-1
    sample_df = pd.read_parquet(TRAINING_SAMPLED_DATA_PATH).reset_index(drop=True)
    labels_vector = sample_df["detailed_label"]
    # Benign rows can have an empty detailed_label; give them the Benign class
    benign_without_label = labels_vector.isna() & sample_df["label"].str.lower().eq("benign")
    print(f"Rows with label=benign and no detailed_label, set to {BENIGN_LABEL}: {benign_without_label.sum():,}")
    labels_vector = labels_vector.mask(benign_without_label, BENIGN_LABEL)
    categorized_labels = labels_vector.map(ATTACK).fillna("benign")

    print(f"Training rows: {len(sample_df):,} with a class distribution:")
    print(labels_vector.value_counts().to_string())
    print(f"Categorised labels: {categorized_labels.head()} \n")
    print(f"Normal labels: {labels_vector.head()} \n")

    # One feature table for all models; they all use the same time window
    features_df = create_features_df(sample_df, time_window=FEATURE_TIME_WINDOW)

    botnet_detector, _ = train_botnet_detector(sample_df, features_df, categorized_labels)
    attack_classifier, attack_classes = train_attack_classifier(sample_df, features_df, categorized_labels, labels_vector)
    everything_classifier, everything_classes = train_everything_classifier(sample_df, features_df, labels_vector)

    save_models(botnet_detector, attack_classifier, everything_classifier, attack_classes, everything_classes)
    print("Training complete. Models saved to disk.")

if __name__ == "__main__":
    main()
