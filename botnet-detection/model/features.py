import pandas as pd

DATA_PATH = "../datasets/sample/iot23_sample.parquet"

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


def load_dataset(path=DATA_PATH):
    df = pd.read_parquet(path)

    return prepare_df(df)


def prepare_df(input_df):
    input_df = input_df.copy()
    input_df["label"] = input_df["label"].str.lower()

    # convert timestamp to datetime
    input_df["ts"] = pd.to_datetime(
        input_df["ts"],
        unit="s",
        utc=True
    )
    return input_df


def create_features_df(input_df):
    time_window = input_df["ts"].dt.floor("10min")
    src = input_df.groupby(["scenario", "id.orig_h", time_window])
    dst = input_df.groupby(["scenario", "id.resp_h", time_window])
    return pd.DataFrame({
        "proto": input_df["proto"].astype("category"),
        "duration": input_df["duration"],
        "orig_pkts": input_df["orig_pkts"],
        "resp_pkts": input_df["resp_pkts"],
        "orig_ip_bytes": input_df["orig_ip_bytes"],
        "resp_ip_bytes": input_df["resp_ip_bytes"],
        "is_single_pkt": input_df["duration"].isna().astype(int),
        "has_response": (input_df["resp_pkts"] > 0).astype(int),
        "history": input_df["history"].astype("category"),
        # how many connections are made to the same destination host in a 10-minute time_window
        "dst_conn_cnt_in_time_window": dst["uid"].transform("size"),
        # how many unique source hosts are connecting to the same destination host in a 10-minute time_window
        "dst_uniq_src_ip_in_time_window": dst["id.orig_h"].transform("nunique"),
        # what the source host does in this 10-minute time window (for example scanning or propagation)
        "src_conn_cnt_in_time_window": src["uid"].transform("size"),
        "src_uniq_dst_ip_in_time_window": src["id.resp_h"].transform("nunique"),
        "src_uniq_dst_port_in_time_window": src["id.resp_p"].transform("nunique"),
        "src_no_reply_ratio_in_time_window": (input_df["resp_pkts"] == 0).groupby(
            [input_df["scenario"], input_df["id.orig_h"], time_window]).transform("mean"),
    })
