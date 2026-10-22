from pathlib import Path

import joblib
import numpy as np
from xgboost import XGBClassifier
import pandas as pd
from features import ATTACK, create_features_df, load_dataset

MODEL_DIR = Path(__file__).parent / "saved-models"

# Those are the columns that are not features but were used for testing purposes
METADATA_COLUMNS = ["row_id", "scenario", "final_class"]
CATEGORICAL_COLUMNS = ["proto", "history"]

def train_botnet_detector(features_df, categorised_lables, y):
    # y = 1 if the row is a botnet row, 0 if the row is a benign row
    y = categorised_lables.ne("benign").astype(int)

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
        random_state=42,
        objective="binary:logistic",
        scale_pos_weight=0.345, # Ratio of benign to botnet rows in the training data, to help with class imbalance
        max_delta_step=1
    )
    botnet_detector.fit(features_df, y)
    print(f"Botnet detector trained")
    return botnet_detector

def train_attack_classifier(features_df, categorised_lables, y):
    # Lable mapping for the attack classifier
    label_map = {"attack": 0, "scanning": 1, "c2": 2}
    
    # Get the indices of rows that are botnet rows
    botnet_indices = (y == 1)
    # Get the labels for the attack classifier, which are only the botnet rows
    y_attack = categorised_lables[botnet_indices].map(label_map)
    # Get the unique classes and their counts for the attack rows
    attack_classes = np.array(sorted(y_attack.unique()))
    # Get the counts for each class in the attack rows
    attack_counts = y_attack.value_counts()
    attack_weights = y_attack.map(np.sqrt(attack_counts.max() / attack_counts))

    print(y_attack)

    print(f"Training attack classifier for {len(attack_classes)} classes: {attack_classes} with counts: {attack_counts.to_dict()}")
    attack_classifier = XGBClassifier(
        n_estimators=800,
        max_depth=6,             
        learning_rate=0.05,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        reg_alpha=0.1,
        tree_method="hist",
        enable_categorical=True,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        random_state=42
    )
    attack_classifier.fit(
        features_df[botnet_indices], 
        y_attack, 
        sample_weight=attack_weights
    )
    print(f"Attack classifier trained")
    return attack_classifier


def train_everything_classifier(features_df, normal_labels):
    # Get all the unique types of attacks
    attack_types = set(normal_labels)
    print(f"Training everything classifier for {len(attack_types)} classes: {list(attack_types)}")
    # Convert it to a map
    mapping = {item: i for i, item in enumerate(attack_types)}
    # Map the normal labels to the new mapping
    y_multi = normal_labels.map(mapping)
    # Get the counts for each class in the new mapping
    all_counts = y_multi.value_counts()
    print(f"Everything classifier counts: {all_counts.to_dict()}")
    # Get the weights for each class
    all_weights = y_multi.map(np.sqrt(all_counts.max() / all_counts))
    everything_classifier = XGBClassifier(
        n_estimators=800,
        max_depth=6,
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
        random_state=42
    )
    everything_classifier.fit(
        features_df, 
        y_multi, 
        sample_weight=all_weights
    )

    print(f"Everything classifier trained") 

    return everything_classifier

# Save the trained models to disk and if not provided, skip it.
def save_models(botnet_detector = None, attack_classifier = None, everything_classifier = None):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if botnet_detector is not None:
        joblib.dump(botnet_detector, MODEL_DIR / "botnet_detector.pkl")
    if attack_classifier is not None:
        joblib.dump(attack_classifier, MODEL_DIR / "attack_classifier.pkl")
    if everything_classifier is not None:
        joblib.dump(everything_classifier, MODEL_DIR / "everything_classifier.pkl")
    print(f"models saved to {MODEL_DIR}")


def main():
    TRAINING_FEATURES_PATH = "data/combined-training/combined-training.features.parquet"
    TRAINING_SAMPLED_DATA_PATH = "data/combined-training/combined-training.sampled.parquet"

    sample_df = pd.read_parquet(TRAINING_SAMPLED_DATA_PATH)
    labels_vector = sample_df["detailed_label"]
    categorised_lables = labels_vector.map(ATTACK).fillna("benign")

    print(f"Training rows: {len(sample_df):,} with a class distribution:")
    print(labels_vector.value_counts().to_string())
    print(f"Categorised labels: {categorised_lables.head()} \n")
    print(f"Normal labels: {labels_vector.head()} \n")

    features_df = create_features_df(sample_df)
    print(f"Features dataframe columns: {features_df.columns.tolist()} \n")
    print(f"The header of the features dataframe: \n{features_df.head()} \n")

    print(f"Dataframe columns before: {sample_df.columns.tolist()}")
    for column in METADATA_COLUMNS:
        if column in sample_df.columns:
            sample_df = sample_df.drop(columns=column)
    print(f"Dataframe columns after: {sample_df.columns.tolist()} \n")

    botnet_detector = train_botnet_detector(features_df, categorised_lables, labels_vector)
    attack_classifier = train_attack_classifier(features_df, categorised_lables, labels_vector)
    everything_classifier = train_everything_classifier(features_df, categorised_lables)

    save_models(botnet_detector, attack_classifier, everything_classifier)
    print("Training complete. Models saved to disk.")

if __name__ == "__main__":
    main()
