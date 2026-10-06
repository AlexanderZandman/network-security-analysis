import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedGroupKFold
from xgboost import XGBClassifier

from features import ATTACK, create_features_df, load_dataset


def evaluate_botnet_vs_benign_classification(features_df, cross_validation, y, groups, detailed_label):
    # oof means out-of-fold, for out-of-fold predictions.
    # Link for reference: https://www.geeksforgeeks.org/machine-learning/what-is-the-oofout-of-fold-approach/
    oof = np.zeros(len(features_df))

    # Train XGBoost classifier for botnet classification
    # n_estimators = number of trees,
    # max_depth = maximum depth of each tree,
    # learning_rate = step size shrinkage
    # tree_method = "hist" for faster training on large datasets, enable_categorical = True for categorical features
    # eval_metric = "aucpr" for area under the precision-recall curve,
    # scale_pos_weight = ratio of negative to positive samples
    # random_state = 42 for reproducibility
    for train, test in cross_validation.split(features_df, y, groups):
        xgb = XGBClassifier(
            n_estimators=150, max_depth=7, learning_rate=0.1,
            tree_method="hist", enable_categorical=True, eval_metric="aucpr",
            scale_pos_weight=(y.iloc[train] == 0).sum() / (y.iloc[train] == 1).sum(),
            random_state=42,
        )
        xgb.fit(features_df.iloc[train], y.iloc[train])
        oof[test] = xgb.predict_proba(features_df.iloc[test])[:, 1]

    print("Botnet vs Benign Classification Report: ")
    print(f"ROC-AUC = {roc_auc_score(y, oof):.4f}")
    print(f"PR-AUC = {average_precision_score(y, oof):.4f}")
    print(classification_report(y, oof > 0.5, target_names=["benign", "botnet"], digits=4))
    recall_per_attack = pd.Series(oof > 0.5, index=features_df.index)
    print(recall_per_attack[y == 1].groupby(detailed_label[y == 1]).agg(recall="mean", n="size")
          .sort_values("recall").round(3))


def evaluate_botnet_attack_type_classification(cross_validation, features_df, y, groups, detailed_label):
    bot = y == 1
    X_bot, groups_bot = features_df[bot], groups[bot]
    attack_bot = detailed_label[bot].map(ATTACK)
    assert attack_bot.notna().all(), "new detailed_label not in ATTACK"

    # unique classes and counts for class weighting
    classes = np.array(sorted(attack_bot.unique()))
    counts = attack_bot.value_counts()

    # sqrt-dampened class weights to avoid overfitting to the minority class which in this case is the C&C class
    # C&C is the most important class for botnet detection, therefore it should not be obscured by the other classes.
    weights = attack_bot.map((counts.max() / counts) ** 0.5)

    # Train XGBoost classifier for attack type classification
    # n_estimators = number of trees, max_depth = maximum depth of each tree, learning_rate = step size shrinkage
    # tree_method = "hist" for faster training on large datasets, enable_categorical = True for categorical features
    # random_state = 42 for reproducibility
    oof_attack = np.empty(len(attack_bot), dtype=object)
    for tr, te in cross_validation.split(X_bot, attack_bot, groups_bot):
        classifier = XGBClassifier(
            n_estimators=150, max_depth=7, learning_rate=0.1,
            tree_method="hist", enable_categorical=True, random_state=42,
        )
        classifier.fit(X_bot.iloc[tr], np.searchsorted(classes, attack_bot.iloc[tr]), sample_weight=weights.iloc[tr])
        oof_attack[te] = classes[classifier.predict(X_bot.iloc[te])]
    print("Attack Type Classification Report: ")
    print(classification_report(attack_bot, oof_attack, digits=4, zero_division=0))
    print(pd.crosstab(attack_bot, oof_attack, rownames=["true"], colnames=["predicted"]))


def main():
    df = load_dataset()
    features_df = create_features_df(df)
    y = (df["label"] == "malicious").astype(int)
    groups = df["scenario"]

    cross_validation = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    evaluate_botnet_vs_benign_classification(features_df, cross_validation, y, groups, df["detailed_label"])
    evaluate_botnet_attack_type_classification(cross_validation, features_df, y, groups, df["detailed_label"])


if __name__ == "__main__":
    main()
