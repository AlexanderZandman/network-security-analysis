import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif

df = pd.read_parquet("./datasets/sample/iot23_sample.parquet")
df['label'] = df['label'].str.lower()

#get distribution
label_counts = df['detailed_label'].value_counts()

#plot
plt.figure(figsize=(12, 6))
sns.barplot(x=label_counts.index, y=label_counts.values)
plt.xticks(rotation=45, ha='right')
plt.title("Distribution of Attack Types in the Dataset")
plt.xlabel("AttackType Label")
plt.ylabel("Number of Records")
plt.grid(True)
plt.tight_layout()
plt.show()

#shows class ratios
print(label_counts)
df["ts"] = pd.to_datetime(
    df["ts"],
    unit="s",
    utc=True
)

df_without_crap = df.drop(columns=["ts", "uid"])
print(df["ts"].head())

malicious, benign = df.groupby('label').get_group('malicious'), df.groupby('label').get_group('benign')

print(f"Malicious: {len(malicious)}")
print(f"Benign: {len(benign)}")

malicious.head()

benign.head()


def calculate_entropy(column, base=None):
    vc = pd.Series(column).value_counts(normalize=True, sort=False)
    base = np.e if base is None else base
    return -(vc * np.log(vc)/np.log(base)).sum()

def calculate_info_gain(df):
    info_gain_dict = {}
    for series_name, series in df.items():
            info_gain_dict[series_name] = mutual_info_classif(df, series, discrete_features=True)[0]
            info_gain_dict[series_name] = info_gain_dict[series_name] if not np.isnan(info_gain_dict[series_name]) else 0.0
    return info_gain_dict
