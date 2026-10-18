import joblib
import numpy as np 
from xgboost import XGBClassifier

from features import ATTACK, create_features_df, load_dataset, undersample

def train_models(features_df, detailed_label, y = None, classification_type: str = "nineway"):

    if classification_type == "binary":
        binary_classifier = XGBClassifier(
            n_estimators=150, max_depth=7, learning_rate=0.1,
            tree_method="hist", enable_categorical=True, eval_metric="aucpr",
            scale_pos_weight=(y == 0).sum() / (y == 1).sum(),
            random_state=42,
        )
        binary_classifier.fit(features_df, y)
        return binary_classifier

    elif classification_type == "nineway":
        bot = y == 1
        attack_bot = detailed_label[bot].map(ATTACK)
        assert attack_bot.notna().all(), "new detailed_label not in ATTACK"
        classes = np.array(sorted(attack_bot.unique()))
        counts = attack_bot.value_counts()
        attack_classifier = XGBClassifier(
            n_estimators=150, max_depth=7, learning_rate=0.1,
            tree_method="hist", enable_categorical=True, random_state=42,
        )
        attack_classifier.fit(features_df[bot], np.searchsorted(classes, attack_bot),
                            sample_weight=attack_bot.map((counts.max() / counts) ** 0.5))

        return attack_classifier

    else:
        raise Exception("Invalid input")

def save_models(botnet_detector, attack_classifier):
    joblib.dump(botnet_detector, "saved-models/binary_classifier.pkl")
    joblib.dump(attack_classifier, "saved-models/nineway_classifier.pkl")
    print(f"models saved")


def main():
    df = load_dataset()
    df_undersampled = undersample(df.copy())
    # features_df = create_features_df(df)
    binary_features_df = create_features_df(df_undersampled)
    nineway_features_df = create_features_df(df)

    y_binary = (df_undersampled["label"] == "malicious").astype(int) 
    y_nineway = (df["label"] == "malicious").astype(int)

    binary_classifier = train_models(binary_features_df, df["detailed_label"], y_binary, "binary")
    nineway_classifier =  train_models(nineway_features_df, df["detailed_label"], y_nineway, "nineway")

    save_models(binary_classifier, nineway_classifier)


if __name__ == "__main__":
    main()
