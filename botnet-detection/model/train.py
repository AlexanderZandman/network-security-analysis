from pathlib import Path

import joblib
import numpy as np 
from xgboost import XGBClassifier
import pandas as pd
from features import ATTACK, create_features_df, load_dataset, undersample
from sklearn.model_selection import train_test_split, StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score

MODEL_DIR = Path(__file__).parent / "saved-models"

def train_botnet_detector(df, categorized_labels):
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
    features_df = create_features_df(df, time_window="5s")
    for train_idx, test_idx in cross_validation.split(df, y, df["scenario"]):
        train = features_df.loc[train_idx]
        #TODO: needs to be undersampled
        test = features_df.loc[test_idx]
        botnet_detector.fit(train, y.loc[train_idx])
        oof[test_idx] = botnet_detector.predict_proba(test)[:, 1]
    
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
    print(f"First 10 predicted botnet rows:\n{predicted_botnet_rows.head(10)}")

    return botnet_detector, predicted_botnet_rows

def train_attack_classifier(df, categorized_labels, y):
    # Lable mapping for the attack classifier
    label_map = {"attack": 0, "scanning": 1, "c2": 2}

    # Get the indices of rows that are botnet rows
    botnet_mask = categorized_labels.ne("benign")
    
    filtered_df = df.loc[botnet_mask].copy().reset_index(drop=True)
    features_df = create_features_df(filtered_df, time_window="1min")

    # Get the labels for the attack classifier, which are only the botnet rows
    y_attack = categorized_labels.loc[botnet_mask].map(label_map).reset_index(drop=True)

    # Get the unique classes and their counts for the attack rows
    attack_classes = np.array(sorted(y_attack.unique()))

    # Get the counts for each class in the attack rows
    attack_counts = y_attack.value_counts()
    attack_weights = y_attack.map(np.sqrt(attack_counts.max() / attack_counts))

    print(f"Training attack classifier for {len(attack_classes)} classes: {attack_classes} with counts: {attack_counts.to_dict()}")
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
        num_class=3,
        eval_metric="mlogloss",
        random_state=50
    )

    cross_validation = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
    oof = np.full(len(filtered_df), -1, dtype=np.int64)

    for train_idx, test_idx in cross_validation.split(filtered_df, y_attack, filtered_df["scenario"]):
        train = features_df.loc[train_idx]
        test = features_df.loc[test_idx]

        train_y = y_attack.iloc[train_idx]

        attack_classifier.fit(
            train, 
            train_y, 
            sample_weight=attack_weights.iloc[train_idx]
        )

        result = attack_classifier.predict(test).astype(np.int64)
        print("RESULT")
        print(result)
        oof[test_idx] = attack_classes[result]
        
    print(f"Attack classifier trained")

    print("Attack Type Classification Report: ")
    print(classification_report(y_attack.to_numpy(dtype=np.int64), oof, digits=4, zero_division=0))
    print(pd.crosstab(y_attack, oof, rownames=["true"], colnames=["predicted"])) 

    return attack_classifier


def train_everything_classifier(df, categorised_labels):
    # Get all the unique types of attacks
    attack_types = set(categorised_labels)
    print(f"Training everything classifier for {len(attack_types)} classes: {list(attack_types)}")
    # Convert it to a map
    mapping = {item: i for i, item in enumerate(attack_types)}
    print(f"Everything classifier mapping: {mapping}")
    # Map the normal labels to the new mapping
    y_multi = categorised_labels.map(mapping)
    print (f"Everything classifier labels: {y_multi.head()} \n")
    # Get the counts for each class in the new mapping
    all_counts = y_multi.value_counts()
    print(f"Everything classifier counts: {all_counts.to_dict()}")
    # Get the weights for each class
    all_weights = y_multi.map(np.sqrt(all_counts.max() / all_counts))
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
        num_class=len(attack_types),
        eval_metric="mlogloss",
        random_state=50
    )

    cross_validation = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
    oof = np.full(len(y_multi), -1, dtype=np.int64)

    features_df = create_features_df(df, time_window="1min")
    for train_idx, test_idx in cross_validation.split(df, y_multi, df["scenario"]):
        train = features_df.loc[train_idx]
        test = features_df.loc[test_idx]

        everything_classifier.fit(
                train, 
                y_multi.iloc[train_idx], 
                sample_weight=all_weights.iloc[train_idx]
        )
        predictions = everything_classifier.predict(test).astype(np.int64)
        oof[test_idx] = predictions

    print(f"Everything classifier trained") 

    print("Direct Classification Report: ")
    print(classification_report(y_multi.to_numpy(dtype=np.int64), oof, digits=4, zero_division=0))
    print(pd.crosstab(y_multi.to_numpy(dtype=np.int64), oof, rownames=["true"], colnames=["predicted"]))

    return everything_classifier

# Save the trained models to disk and if not provided, skip it.
def save_models(botnet_detector = None, attack_classifier = None, everything_classifier = None):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if botnet_detector is not None:
        botnet_detector.save_model(MODEL_DIR / "botnet_detector.json")
    if attack_classifier is not None:
        attack_classifier.save_model(MODEL_DIR / "attack_classifier.json")
    if everything_classifier is not None:
        everything_classifier.save_model(MODEL_DIR / "everything_classifier.pkl")
    print(f"models saved to {MODEL_DIR}")


def main():
    TRAINING_SAMPLED_DATA_PATH = "data/combined-training/combined-training-trimmed.parquet"

    sample_df = pd.read_parquet(TRAINING_SAMPLED_DATA_PATH)
    labels_vector = sample_df["detailed_label"]
    categorized_labels = labels_vector.map(ATTACK).fillna("benign")

    print(f"Training rows: {len(sample_df):,} with a class distribution:")
    print(labels_vector.value_counts().to_string())
    print(f"Categorised labels: {categorized_labels.head()} \n")
    print(f"Normal labels: {labels_vector.head()} \n")

    botnet_detector = train_botnet_detector(sample_df, categorized_labels)
    attack_classifier = train_attack_classifier(sample_df, categorized_labels, labels_vector)
    everything_classifier = train_everything_classifier(sample_df, categorized_labels)

    save_models(botnet_detector, attack_classifier, everything_classifier)
    print("Training complete. Models saved to disk.")

if __name__ == "__main__":
    main()
