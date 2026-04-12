from os import path

import numpy as np
import pandas as pd

AWID2_CSV_CLASESS = {
    "normal": 0,
    "flooding": 1,
    "impersonation": 2,
    "injection": 6,
    "amok": 1,
    "arp": 6,
    "authentication_request": 1,
    "beacon": 1,
    "cafe_latte": 2,
    "chop_chop": 6,
    "cts": 1,
    "deauthentication": 1,
    "dissassociation": 1,
    "evil_twin": 2,
    "fragmentation": 6,
    "hirte": 2,
    "power": 1,
    "probe_request": 1,
    "probe_response": 1,
    "rts": 1,
}

AWID2_CSV_COLUMN_NAMES = [
    "frame.interface_id",
    "frame.dlt",
    "frame.offset_shift",
    "frame.time_epoch",
    "frame.time_delta",
    "frame.time_delta_displayed",
    "frame.time_relative",
    "frame.len",
    "frame.cap_len",
    "frame.marked",
    "frame.ignored",
    "radiotap.version",
    "radiotap.pad",
    "radiotap.length",
    "radiotap.present.tsft",
    "radiotap.present.flags",
    "radiotap.present.rate",
    "radiotap.present.channel",
    "radiotap.present.fhss",
    "radiotap.present.dbm_antsignal",
    "radiotap.present.dbm_antnoise",
    "radiotap.present.lock_quality",
    "radiotap.present.tx_attenuation",
    "radiotap.present.db_tx_attenuation",
    "radiotap.present.dbm_tx_power",
    "radiotap.present.antenna",
    "radiotap.present.db_antsignal",
    "radiotap.present.db_antnoise",
    "radiotap.present.rxflags",
    "radiotap.present.xchannel",
    "radiotap.present.mcs",
    "radiotap.present.ampdu",
    "radiotap.present.vht",
    "radiotap.present.reserved",
    "radiotap.present.rtap_ns",
    "radiotap.present.vendor_ns",
    "radiotap.present.ext",
    "radiotap.mactime",
    "radiotap.flags.cfp",
    "radiotap.flags.preamble",
    "radiotap.flags.wep",
    "radiotap.flags.frag",
    "radiotap.flags.fcs",
    "radiotap.flags.datapad",
    "radiotap.flags.badfcs",
    "radiotap.flags.shortgi",
    "radiotap.datarate",
    "radiotap.channel.freq",
    "radiotap.channel.type.turbo",
    "radiotap.channel.type.cck",
    "radiotap.channel.type.ofdm",
    "radiotap.channel.type.2ghz",
    "radiotap.channel.type.5ghz",
    "radiotap.channel.type.passive",
    "radiotap.channel.type.dynamic",
    "radiotap.channel.type.gfsk",
    "radiotap.channel.type.gsm",
    "radiotap.channel.type.sturbo",
    "radiotap.channel.type.half",
    "radiotap.channel.type.quarter",
    "radiotap.dbm_antsignal",
    "radiotap.antenna",
    "radiotap.rxflags.badplcp",
    "wlan.fc.type_subtype",
    "wlan.fc.version",
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
    "wlan.ra",
    "wlan.da",
    "wlan.ta",
    "wlan.sa",
    "wlan.bssid",
    "wlan.frag",
    "wlan.seq",
    "wlan.bar.type",
    "wlan.ba.control.ackpolicy",
    "wlan.ba.control.multitid",
    "wlan.ba.control.cbitmap",
    "wlan.bar.compressed.tidinfo",
    "wlan.ba.bm",
    "wlan.fcs_good",
    "wlan_mgt.fixed.capabilities.ess",
    "wlan_mgt.fixed.capabilities.ibss",
    "wlan_mgt.fixed.capabilities.cfpoll.ap",
    "wlan_mgt.fixed.capabilities.privacy",
    "wlan_mgt.fixed.capabilities.preamble",
    "wlan_mgt.fixed.capabilities.pbcc",
    "wlan_mgt.fixed.capabilities.agility",
    "wlan_mgt.fixed.capabilities.spec_man",
    "wlan_mgt.fixed.capabilities.short_slot_time",
    "wlan_mgt.fixed.capabilities.apsd",
    "wlan_mgt.fixed.capabilities.radio_measurement",
    "wlan_mgt.fixed.capabilities.dsss_ofdm",
    "wlan_mgt.fixed.capabilities.del_blk_ack",
    "wlan_mgt.fixed.capabilities.imm_blk_ack",
    "wlan_mgt.fixed.listen_ival",
    "wlan_mgt.fixed.current_ap",
    "wlan_mgt.fixed.status_code",
    "wlan_mgt.fixed.timestamp",
    "wlan_mgt.fixed.beacon",
    "wlan_mgt.fixed.aid",
    "wlan_mgt.fixed.reason_code",
    "wlan_mgt.fixed.auth.alg",
    "wlan_mgt.fixed.auth_seq",
    "wlan_mgt.fixed.category_code",
    "wlan_mgt.fixed.htact",
    "wlan_mgt.fixed.chanwidth",
    "wlan_mgt.fixed.fragment",
    "wlan_mgt.fixed.sequence",
    "wlan_mgt.tagged.all",
    "wlan_mgt.ssid",
    "wlan_mgt.ds.current_channel",
    "wlan_mgt.tim.dtim_count",
    "wlan_mgt.tim.dtim_period",
    "wlan_mgt.tim.bmapctl.multicast",
    "wlan_mgt.tim.bmapctl.offset",
    "wlan_mgt.country_info.environment",
    "wlan_mgt.rsn.version",
    "wlan_mgt.rsn.gcs.type",
    "wlan_mgt.rsn.pcs.count",
    "wlan_mgt.rsn.akms.count",
    "wlan_mgt.rsn.akms.type",
    "wlan_mgt.rsn.capabilities.preauth",
    "wlan_mgt.rsn.capabilities.no_pairwise",
    "wlan_mgt.rsn.capabilities.ptksa_replay_counter",
    "wlan_mgt.rsn.capabilities.gtksa_replay_counter",
    "wlan_mgt.rsn.capabilities.mfpr",
    "wlan_mgt.rsn.capabilities.mfpc",
    "wlan_mgt.rsn.capabilities.peerkey",
    "wlan_mgt.tcprep.trsmt_pow",
    "wlan_mgt.tcprep.link_mrg",
    "wlan.wep.iv",
    "wlan.wep.key",
    "wlan.wep.icv",
    "wlan.tkip.extiv",
    "wlan.ccmp.extiv",
    "wlan.qos.tid",
    "wlan.qos.priority",
    "wlan.qos.eosp",
    "wlan.qos.ack",
    "wlan.qos.amsdupresent",
    "wlan.qos.buf_state_indicated",
    "wlan.qos.bit4",
    "wlan.qos.txop_dur_req",
    "wlan.qos.buf_state_indicated_",
    "data.len",
    "class",
]

AWID2_CSV_USED_COLUMNS = [
    "frame.time_delta",
    "frame.len",
    "radiotap.length",
    "radiotap.flags.cfp",
    "radiotap.flags.preamble",
    "radiotap.flags.wep",
    "radiotap.flags.frag",
    "radiotap.flags.fcs",
    "radiotap.flags.datapad",
    "radiotap.flags.badfcs",
    "radiotap.flags.shortgi",
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
    "class",
]


def read_csv_to_df(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(
        filepath,
        header=None,
        names=AWID2_CSV_COLUMN_NAMES,
        on_bad_lines="skip",
    )

    df = df[AWID2_CSV_USED_COLUMNS]
    assert type(df) is pd.DataFrame

    df.replace({"?": 0}, inplace=True)

    df.fillna(0.0)
    return df


def convert_to_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[int]]:
    features = pd.DataFrame()
    labels = [AWID2_CSV_CLASESS[_class] for _class in df["class"]]

    features["frame_len"] = df["frame.len"].astype(float) - df[
        "radiotap.length"
    ].astype(float)
    features["delta_time_us"] = df["frame.time_delta"].astype(float) * 1_000_000
    flag_cols = [
        "radiotap.flags.cfp",
        "radiotap.flags.preamble",
        "radiotap.flags.wep",
        "radiotap.flags.frag",
        "radiotap.flags.fcs",
        "radiotap.flags.datapad",
        "radiotap.flags.badfcs",
        "radiotap.flags.shortgi",
    ]
    features["rt_flags"] = sum(df[col].astype(int) * 2**i for i, col in enumerate(flag_cols))
    features["rt_flags"] = features["rt_flags"].astype(float)
    features["rt_ant_signal_dbm"] = df["radiotap.dbm_antsignal"].astype(float)
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


def save_to_npz(filepath: str, features: pd.DataFrame, labels: list[int]) -> None:
    X = features.to_numpy()
    y = np.array(labels)

    np.savez(filepath, X=X, y=y)


def process_awid2_csv(filepath: str, dest_dir: str) -> None:
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
            process_awid2_csv(path.join(filepath, file), dest_dir)

    else:
        process_awid2_csv(filepath, dest_dir)
