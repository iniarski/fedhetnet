import logging
import os
import random

import jax
import numpy as np
import shlex
from sklearn.model_selection import train_test_split

from lib5gpl2 import lib5gpl2py

from data.npzloader import CLASS_SAMPLING

def read_labeling_from_script(script_path: str) -> dict[str: lib5gpl2py.LabelingRules]:
    with open(script_path) as f:
        awid3_dir = ""
        pcaps_f_dir = ""
        pcaps_inz_dir = ""
        pcaps_3gpp_dir = ""
        rules = dict()

        for line in f:
            if line.startswith("$writer"):
                symbols = shlex.split(line)
                source_pcap = ""
                lr = lib5gpl2py.init_labeling_rules()
                for i, symbol in enumerate(symbols):
                    if symbol == '-i':
                        input_file = symbols[i + 1]
                        if input_file.startswith("${awid3_dir}"):
                            file = input_file.split('}')[1]
                            source_pcap = os.path.join(awid3_dir, file)
                        elif input_file.startswith("${pcaps_f_dir}"):
                            file = input_file.split('}')[1]
                            source_pcap = os.path.join(pcaps_f_dir, file)
                        elif input_file.startswith("${pcaps_inz_dir}"):
                            file = input_file.split('}')[1]
                            source_pcap = os.path.join(pcaps_inz_dir, file)
                        elif input_file.startswith("${pcaps_3gpp_dir}"):
                            file = input_file.split('}')[1]
                            source_pcap = os.path.join(pcaps_3gpp_dir, file)
                    elif symbol == '-f':
                        filter_string = symbols[i + 1]
                        lr.set_filter(filter_string)
                    elif symbol == '-n':
                        lr.label_all_normal = True
                    elif symbol == '-u':
                        lr.unprotected_only = True
                    elif symbol == '-P':
                        lr.pwrmgmt_only = True
                    elif symbol == '-m':
                        lr.malformed_only = True
                    elif symbol == '-I':
                        lr.ip_capture = True
                    elif symbol == '-l':
                        label = int(symbols[i + 1])
                        lr.attack_label = label
                    elif symbol == '-s':
                        index = int(symbols[i + 1])
                        lr.start_attack_index = index
                    elif symbol == '-e':
                        index = int(symbols[i + 1])
                        lr.end_attack_index = index

                if lr.validate():
                    rules[source_pcap] = lr
                else:
                    logging.error(f"Invalid labeling rules for file {source_pcap}")

            elif line.startswith("awid3_dir"):
                awid3_dir = line.split('"')[1]
            elif line.startswith("pcaps_f_dir"):
                pcaps_f_dir = line.split('"')[1]
            elif line.startswith("pcaps_inz_dir"):
                pcaps_inz_dir = line.split('"')[1]
            elif line.startswith("pcaps_3gpp_dir"):
                pcaps_3gpp_dir = line.split('"')[1]
        return rules


def print_labeling(lr: lib5gpl2py.LabelingRules) -> None:
    print(f"Filter srting: {lr.get_filter()}")
    print(f"Label all normal: {lr.label_all_normal}")
    print(f"Unprotected only: {lr.unprotected_only}")
    print(f"Power Mangement only: {lr.pwrmgmt_only}")
    print(f"Malformed only: {lr.malformed_only}")
    print(f"Ip capture: {lr.ip_capture}")
    print(f"Attack label: {lr.attack_label}")
    print(f"Attack start index: {lr.start_attack_index}")
    print(f"Attack end index: {lr.end_attack_index}")


def make_dataset(
        script_path: str,
        class_sampling: list[float] = CLASS_SAMPLING,
        batch_size: int = 256,
        selected_features: set[str] = set(),
        seed: int = 42,
        shuffle: bool = False,
        buffer_size: int = int(5e6),
        float_tokenization: bool = False,
        normal_only: bool = False,
        split: float = 0.8
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    n_classes = len(class_sampling)

    if any(x < 0 for x in class_sampling):
        raise Exception("Expected all elements of class sampling to be positive")
    if (split < 0 or split > 1):
        raise Exception("split should be in range (0, 1)")

    labeling_rules = read_labeling_from_script(script_path)
    tokens_by_class = [list() for _ in range(n_classes)]
    labels_by_class = [list() for _ in range(n_classes)]

    for file, labeling in labeling_rules.items():
        with lib5gpl2py.LabeledCapture(
            file, batch_size, labeling, selected_features, buffer_size=buffer_size, live_capture=False, float_tokenization=float_tokenization
        ) as cap:
            for _tokens, _labels, _ in cap:
                sample_class = np.max(_labels)

                if not normal_only or sample_class == 0:
                    tokens_by_class[sample_class].append(_tokens)
                    labels_by_class[sample_class].append(_labels)

    X_train, Y_train, X_test, Y_test = list(), list(), list(), list()
    for c in range(n_classes):
        if len(tokens_by_class[c]) == 0:
            continue

        tokens = tokens_by_class[c]
        labels = labels_by_class[c]
        x_train, x_test, y_train, y_test = train_test_split(tokens, labels, train_size=split, random_state=seed, shuffle=False)
        X_test.extend(x_test)
        Y_test.extend(y_test)

        sampling = class_sampling[c]
        repeats, rest = int(sampling), int(len(x_train) * sampling % 1)
        x_train_sampled, y_train_sampled = repeats * x_train, repeats * y_train

        if rest > 0:
            idx = list(range(len(x_train)))
            random.Random(seed).shuffle(idx)
            x_train_sampled.extend([x_train[i] for i in idx[:rest]])
            y_train_sampled.extend([y_train[i] for i in idx[:rest]])

        X_train.extend(x_train_sampled)
        Y_train.extend(y_train_sampled)

    if shuffle:
        idx = list(range(len(x_train)))
        random.Random(seed).shuffle(idx)
        X_train = [X_train[i] for i in idx]
        Y_train = [Y_train[i] for i in idx]

        idx = list(range(len(x_test)))
        random.Random(seed + 1).shuffle(idx)
        X_test = [X_test[i] for i in idx]
        Y_test = [Y_test[i] for i in idx]

    return X_train, Y_train, X_test, Y_test


if __name__ == '__main__':
    rules = read_labeling_from_script('scripts/make_ds_npz.sh')

    for file, lr in rules.items():
        print(f"File: {file}")
        print_labeling(lr)

    _, labels, _, _ = make_dataset('5gpl2/ml5g/byte_model/scripts/train_dataset.sh')
    labels = np.stack(labels)
    counts = np.bincount(labels, minlength=6)
    for i in range(6):
        print(f"Label {i}: {counts[i]}")
