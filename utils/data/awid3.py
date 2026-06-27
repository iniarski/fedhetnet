from os import path

import numpy as np
import pandas as pd

AWID3_CSV_CLASESS = {
    "Normal": 0,
    "SQL_Injection" : 4,
    "SSDP" : 4,
    "SDDP" : 4, # typo in csv
    "Evil_Twin" : 2,
    "Website_spoofing" : 4,
    "Deauth" : 1,
    "Disas": 1,
    '3.(Re)Assoc' : 1,   
    "Rogue_AP" : 2,   
    "Krack" : 2,
    "Kr00k" : 1,
    "SSH" : 4,
    "Botnet" : 4, 
    "Malware" : 4,
    np.nan : 0
}

AWID3_CSV_USED_COLUMNS = [
    "frame.time_delta",
    "frame.len",
    "radiotap.length",
    "wlan.fcs.bad_checksum",
    "radiotap.datarate",
    "radiotap.dbm_antsignal",
    "wlan.fc.type",
    "wlan.fc.subtype",
    "wlan.fc.ds",
    "wlan.fc.frag",
    "wlan.fc.retry",
    "wlan.fc.pwrmgt",
    "wlan.fc.moredata",
    "wlan.fc.protected",
    "wlan.fc.order",
    "wlan.duration",
    "wlan.seq",
    "Label",
]


def read_csv_to_df(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(
        filepath,
        on_bad_lines="skip",
    )

    df = df[AWID3_CSV_USED_COLUMNS]
    assert type(df) is pd.DataFrame

    df.replace({"?": 0}, inplace=True)

    df.fillna(0.0)
    return df


def convert_to_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[int]]:
    features = pd.DataFrame()
    labels = [AWID3_CSV_CLASESS[_class] for _class in df["Label"]]

    features["frame_len"] = df["frame.len"].astype(float) - df[
        "radiotap.length"
    ].astype(float)
    features["delta_time_us"] = df["frame.time_delta"].astype(float) * 1_000_000
    features["rt_flags"] = df["wlan.fcs.bad_checksum"].map(lambda x: 0x50 if x else 0x10)
    features["rt_flags"] = features["rt_flags"].astype(float)
    features["rt_ant_signal_dbm"] = df["radiotap.dbm_antsignal"].map(lambda x: -1 * int(x.split('-')[-1]) if type(x) is str else x).astype(float)
    features["rt_datarate_100kbps"] = df["radiotap.datarate"].astype(float) * 10
    features["wlan_duration"] = df["wlan.duration"].astype(float)
    for type_ in range(3):
        features[f"wlan_type_{type_}"] = (df["wlan.fc.type"] == type_).astype(float)
    for subtype in range(16):
        features[f"wlan_subtype_{subtype}"] = (df["wlan.fc.subtype"] == subtype).astype(
            float
        )
    features["wlan_ds"] = (
        df["wlan.fc.ds"]
        .apply(lambda x: int(x, 16) if type(x) is str else x)
        .astype(float)
    )
    features["wlan_frag"] = df["wlan.fc.frag"].astype(float)
    features["wlan_retry"] = df["wlan.fc.retry"].astype(float)
    features["wlan_pwrmgmt"] = df["wlan.fc.pwrmgt"].astype(float)
    features["wlan_moredata"] = df["wlan.fc.moredata"].astype(float)
    features["wlan_protected"] = df["wlan.fc.protected"].astype(float)
    features["wlan_order"] = df["wlan.fc.order"].astype(float)
    features["wlan_seq"] = df["wlan.seq"].astype(float)

    return features, labels


def save_to_npz(filepath: str, features: pd.DataFrame, labels: list[int], compressed = False) -> None:
    X = features.to_numpy(dtype=np.float16)
    y = np.array(labels, dtype=np.uint8)
    f16_max = np.finfo(np.float16).max
    
    X = np.nan_to_num(
        X,
        nan=0.0,
        posinf=f16_max,
        neginf=-f16_max
    )

    assert np.isfinite(X).all()

    if compressed:
        np.savez_compressed(filepath, X=X, y=y)
    else:
        np.savez(filepath, X=X, y=y)


def process_AWID3_csv(filepath: str, dest_dir: str) -> None:
    df = read_csv_to_df(filepath)
    df, labels = convert_to_features(df)
    filename = path.basename(filepath)
    target = path.join(dest_dir, filename)
    save_to_npz(target, df, labels)


if __name__ == "__main__":
    from os import listdir, mkdir
    from sys import argv

    filepath = argv[1]
    dest_dir = argv[2]

    if not path.exists(dest_dir):
        mkdir(dest_dir)

    if path.isdir(filepath):
        for file in listdir(filepath):
            if not path.isfile(path.join(filepath, file)):
                continue
            print(f"Processing {file}")
            process_AWID3_csv(path.join(filepath, file), dest_dir)

    else:
        process_AWID3_csv(filepath, dest_dir)
