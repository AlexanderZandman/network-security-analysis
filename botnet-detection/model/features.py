import pandas as pd

DATA_PATH = "data/CTU-IoT-Malware-Capture-34-1.parquet"

# Attacks are grouped into three categories: scanning, attack, and C&C.
# Sidenote: C&C is command and control, which is the communication channel between the botnet and the botnet operator.
# So C&C is the most important to detect in our case, because this is a label for botnets
ATTACK = {
    "PartOfAHorizontalPortScan": "scanning", "C&C-PartOfAHorizontalPortScan": "scanning",
    "PartOfAHorizontalPortScan-Attack": "scanning", "Okiru": "scanning", "Okiru-Attack": "scanning",
    "DDoS": "attack", "Attack": "attack", "C&C-HeartBeat-Attack": "attack",
    "C&C": "c2", "C&C-HeartBeat": "c2", "C&C-Torii": "c2", "C&C-Mirai": "c2",
    "C&C-FileDownload": "c2", "C&C-HeartBeat-FileDownload": "c2", "FileDownload": "c2",
}

CATEGORICAL = ("proto", "history")

RANDOM_SEED = 68 - 1

def load_dataset(path=DATA_PATH):
    df = pd.read_parquet(path)

    return prepare_df(df)

def prepare_df(input_df):
    """
        Preprocess the dataset to clean up data points.
    """
    input_df = input_df.copy()
    # Set all labels to lowercase for consistency.
    input_df["label"] = input_df["label"].str.lower()

    # Tunnel parents are irrelevant and can be dropped.
    input_df =  input_df.drop('tunnel_parents', axis=1)

    # Convert timestamp to datetime
    input_df["ts"] = pd.to_datetime(
        input_df["ts"],
        unit="s",
        utc=True
    )

    # -----Deal with NaN values-----
    
    # Fields that can be NaN:
    #   service - string
    #   duration - numeric
    #   orig_bytes - numeric
    #   resp_bytes - numeric

    # Drop rows where service is missing and flow byte data is also missing
    drop_mask = (
        input_df[["service", "orig_bytes", "resp_bytes"]].isna().all(axis=1)
    )
    input_df = input_df.loc[~drop_mask]

    # Replace missing services with empty string
    input_df["service"] = input_df["service"].fillna("")

    # Replace NaN duration with 0. Assume 0 duration == single packet transmission
    input_df["duration"] = input_df["duration"].fillna(0)

    # Impute empty numeric values with mean value for matching service
    for col in ["orig_bytes", "resp_bytes"]:
        input_df[col] = input_df[col].fillna(input_df.groupby("service")[col].transform("mean"))

    return input_df

def undersample(input_df):
    """
        Undersample input dataset to balance out benign and malicious cases
    """

    benign = input_df[input_df["label"] == "benign"]
    malicious = input_df[input_df["label"] != "benign"]

    # undersample to length of benign samples
    malicious = malicious.sample(n=len(benign), random_state=RANDOM_SEED)

    balanced_df = (pd.concat([benign, malicious])
                        .sample(frac=1, random_state=RANDOM_SEED)
                        .reset_index(drop=True)
    )

    return balanced_df

def create_features_df(input_df):
    """
        Create a dataframe of features for a given type of classification model.
    """
    time_window = input_df["ts"].dt.floor("1min")
    src = input_df.groupby(["scenario", "id.orig_h", time_window])
    dst = input_df.groupby(["scenario", "id.resp_h", time_window])
    return pd.DataFrame({
        "proto": input_df["proto"].astype("category"),
        "duration": input_df["duration"],
        "orig_pkts": input_df["orig_pkts"],
        "resp_pkts": input_df["resp_pkts"],
        "orig_ip_bytes": input_df["orig_ip_bytes"],
        "resp_ip_bytes": input_df["resp_ip_bytes"],
        "is_single_pkt": (input_df["duration"] == 0).astype(int),
        "has_response": (input_df["resp_pkts"] > 0).astype(int),
        # "history": input_df["history"].astype("category"),
        # how many connections are made to the same destination host in a 10-minute time_window
        "dst_conn_cnt_in_time_window": dst["uid"].transform("size"),
        # how many unique source hosts are connecting to the same destination host in a 10-minute time_window
        "dst_uniq_src_ip_in_time_window": dst["id.orig_h"].transform("nunique"),
        # what the source host does in this 1-minute time window (for example scanning or propagation)
        "src_conn_cnt_in_time_window": src["uid"].transform("size"),
        "src_uniq_dst_ip_in_time_window": src["id.resp_h"].transform("nunique"),
        "src_uniq_dst_port_in_time_window": src["id.resp_p"].transform("nunique"),
        "src_no_reply_ratio_in_time_window": (input_df["resp_pkts"] == 0).groupby(
            [input_df["scenario"], input_df["id.orig_h"], time_window]).transform("mean"),
    })
