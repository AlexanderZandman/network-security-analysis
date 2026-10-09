import joblib
import numpy as np
from xgboost import XGBClassifier

from features import ATTACK, create_features_df, load_dataset

def train_models(features_df, y, detailed_label):
    botnet_detector = XGBClassifier(
        n_estimators=150, max_depth=7, learning_rate=0.1,
        tree_method="hist", enable_categorical=True, eval_metric="aucpr",
        scale_pos_weight=(y == 0).sum() / (y == 1).sum(),
        random_state=42,
    )
    botnet_detector.fit(features_df, y)

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

    return botnet_detector, attack_classifier

def save_models(botnet_detector, attack_classifier):
    joblib.dump(botnet_detector, "saved-models/botnet_detector.pkl")
    joblib.dump(attack_classifier, "saved-models/attack_classifier.pkl")
    print(f"models saved")


def main():
    df = load_dataset()
    features_df = create_features_df(df)
    y = (df["label"] == "malicious").astype(int)
    botnet_detector, attack_classifier = train_models(features_df, y, df["detailed_label"])
    save_models(botnet_detector, attack_classifier)


if __name__ == "__main__":
    main()
