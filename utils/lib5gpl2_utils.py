import logging
import os
import random

import jax
import numpy as np
import shlex
from sklearn.model_selection import train_test_split

from lib5gpl2 import lib5gpl2py

from utils.data.dataloader import CLASS_SAMPLING
N_FEATURES = 33

def read_labeling_from_script(script_path: str) -> tuple[dict[str: lib5gpl2py.LabelingRules], list[str]]:
    with open(script_path) as f:
        awid3_dir = ""
        pcaps_f_dir = ""
        pcaps_inz_dir = ""
        pcaps_3gpp_dir = ""
        rules = dict()
        np_files = []

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
            elif line.startswith("# read_numpy"):
                p = line.split(' ')[-1].removesuffix('\n')
                if os.path.isdir(p):
                    for f in os.listdir(p):
                        if f.endswith('.npz'):
                            np_files.append(os.path.join(p, f))
                elif os.path.isfile(p):
                    np_files.append(p)
                else:
                    print(f"Bad .npz path: {p}")
            elif line.startswith("awid3_dir"):
                awid3_dir = line.split('"')[1]
            elif line.startswith("pcaps_f_dir"):
                pcaps_f_dir = line.split('"')[1]
            elif line.startswith("pcaps_inz_dir"):
                pcaps_inz_dir = line.split('"')[1]
            elif line.startswith("pcaps_3gpp_dir"):
                pcaps_3gpp_dir = line.split('"')[1]
        print(f"Input pcap files: {len(rules)}, npz files: {len(np_files)}")
        return rules, np_files


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
        data_source: str | tuple,
        class_sampling: list[float] = CLASS_SAMPLING,
        batch_size: int = 256,
        selected_features: set[str] = set(),
        seed: int = 42,
        shuffle: bool = False,
        buffer_size: int = int(5e6),
        float_tokenization: bool = True,
        normal_only: bool = False,
        split: float = 0.8,
        n_features: int = N_FEATURES
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    n_classes = len(class_sampling)
    if any(x < 0 for x in class_sampling):
        raise Exception("Expected all elements of class sampling to be positive")
    if split < 0 or split > 1:
        raise Exception("split should be in range (0, 1)")

    if type(data_source) is str:
        labeling_rules, np_files = read_labeling_from_script(data_source)
    else:
        labeling_rules, np_files = data_source
    X_train, Y_train, X_test, Y_test = list(), list(), list(), list()

    def _sample_and_accumulate(tokens_by_class, labels_by_class):
        """Split and apply class sampling on one file's worth of data, then extend the output lists."""
        for c in range(n_classes):
            if len(tokens_by_class[c]) == 0:
                continue
            tokens = tokens_by_class[c]
            labels = labels_by_class[c]
            if len(tokens) == 1:
                x_train, x_test, y_train, y_test = tokens, [], labels, []
            else:
                x_train, x_test, y_train, y_test = train_test_split(
                    tokens, labels, train_size=split, random_state=seed, shuffle=False
                )
            X_test.extend(x_test)
            Y_test.extend(y_test)
            sampling = class_sampling[c]
            repeats, rest = int(sampling), int(len(x_train) * sampling)
            x_train_sampled = repeats * x_train
            y_train_sampled = repeats * y_train
            if rest > 0:
                idx = list(range(len(x_train)))
                random.Random(seed).shuffle(idx)
                x_train_sampled.extend([x_train[i] for i in idx[:rest]])
                y_train_sampled.extend([y_train[i] for i in idx[:rest]])
            X_train.extend(x_train_sampled)
            Y_train.extend(y_train_sampled)
    print("Processing pcap files")

    f16_max = np.finfo(np.float16).max


    for file, labeling in labeling_rules.items():
        tokens_by_class = [list() for _ in range(n_classes)]
        labels_by_class = [list() for _ in range(n_classes)]
        with lib5gpl2py.LabeledCapture(
            file, batch_size, labeling, buffer_size=buffer_size,
            live_capture=False, float_tokenization=float_tokenization
        ) as cap:
            for _tokens, _labels, _ in cap:
                if float_tokenization:
                    _tokens = np.clip(_tokens, -f16_max, f16_max).astype(np.float16)
                sample_class = np.max(_labels)
                if not normal_only or sample_class == 0:
                    tokens_by_class[sample_class].append(_tokens)
                    labels_by_class[sample_class].append(_labels)
        _sample_and_accumulate(tokens_by_class, labels_by_class)

    print("Finished processing pcap files, moving onto numpy")
    
    def _sample_and_accumulate_numpy(tokens_by_class, labels_by_class):
        for c in range(n_classes):
            if len(tokens_by_class[c]) == 0:
                continue
            tokens = tokens_by_class[c]
            labels = labels_by_class[c]
            if len(tokens) == 1:
                x_train, x_test, y_train, y_test = tokens, tokens[:0], labels, labels[:0]
            else:
                x_train, x_test, y_train, y_test = train_test_split(
                    tokens, labels, train_size=split, random_state=seed, shuffle=False
                )
            X_test.extend(x_test)
            Y_test.extend(y_test)

            sampling = class_sampling[c]
            repeats, rest = int(sampling), int(len(x_train) * (sampling % 1))

            # np.tile replaces list multiplication (repeats * x_train)
            tile_dims = (repeats,) + (1,) * (x_train.ndim - 1)
            x_train_sampled = np.tile(x_train, tile_dims)
            y_train_sampled = np.tile(y_train, (repeats,) + (1,) * (y_train.ndim - 1))

            if rest > 0:
                idx = list(range(len(x_train)))
                random.Random(seed).shuffle(idx)
                # np.concatenate replaces .extend on the sampled arrays
                x_train_sampled = np.concatenate([x_train_sampled, x_train[idx[:rest]]])
                y_train_sampled = np.concatenate([y_train_sampled, y_train[idx[:rest]]])

            # .extend on the outer lists still works — numpy iterates over axis 0
            X_train.extend(x_train_sampled)
            Y_train.extend(y_train_sampled)

    for np_file in np_files:
        data = np.load(np_file, mmap_mode='r')
        X, y = data['X'], data['y']
        if y.ndim == 1:
            n_batches = len(y) // batch_size
        else:
            n_batches = len(y)

        # --- first pass: count batches per class (no data loaded) ---
        class_counts = np.zeros(n_classes, dtype=np.int64)
        for i in range(n_batches):
            _labels = y[i] if y.ndim == 2 else y[i*batch_size:(i+1)*batch_size]
            sample_class = np.max(_labels)
            if not normal_only or sample_class == 0:
                class_counts[sample_class] += 1

        # --- pre-allocate one contiguous array per class ---
        tokens_by_class = [
            np.empty((class_counts[c], batch_size, n_features), dtype=X.dtype)
            for c in range(n_classes)
        ]
        labels_by_class = [
            np.empty((class_counts[c], batch_size), dtype=y.dtype)
            for c in range(n_classes)
        ]
        fill_idx = np.zeros(n_classes, dtype=np.int64)

        # --- second pass: fill pre-allocated arrays ---
        for i in range(n_batches):
            if y.ndim == 2:
                _tokens, _labels = X[i], y[i]
            else:
                _tokens = X[i*batch_size:(i+1)*batch_size]
                _labels = y[i*batch_size:(i+1)*batch_size]
            sample_class = np.max(_labels)
            if not normal_only or sample_class == 0:
                slot = fill_idx[sample_class]
                tokens_by_class[sample_class][slot] = _tokens  # copy into pre-alloc'd buffer
                labels_by_class[sample_class][slot] = _labels
                fill_idx[sample_class] += 1

        del data, X, y  # release mmap
        _sample_and_accumulate_numpy(tokens_by_class, labels_by_class)

    print("Finished processing numpy files")
    print(f"Train size: {len(X_train)}")
    print(f"Test size: {len(X_test)}")

    if shuffle:
        idx = list(range(len(X_train)))
        random.Random(seed).shuffle(idx)
        X_train = [X_train[i] for i in idx]
        Y_train = [Y_train[i] for i in idx]
        idx = list(range(len(X_test)))
        random.Random(seed + 1).shuffle(idx)
        X_test = [X_test[i] for i in idx]
        Y_test = [Y_test[i] for i in idx]

    return X_train, Y_train, X_test, Y_test

def per_file_split(script_path, n_clients=1, seed=42):
    labeling_rules, np_files = read_labeling_from_script(script_path)
    if n_clients == 1:
        return [(labeling_rules, np_files)]
    
    clients = [
        ({}, []) for _ in range(n_clients)
    ]
    
    random.seed(seed)
    for k, v in labeling_rules.items():
        i = random.randint(0, n_clients - 1)
        clients[i][0][k] = v
    for np_file in np_files:
        i = random.randint(0, n_clients - 1)
        clients[i][1].append(np_file)
        
    return clients
        

if __name__ == '__main__':
    rules, _ = read_labeling_from_script('scripts/make_ds_npz.sh')

    for file, lr in rules.items():
        print(f"File: {file}")
        print_labeling(lr)

    _, labels, _, _ = make_dataset('5gpl2/ml5g/byte_model/scripts/train_dataset.sh')
    labels = np.stack(labels)
    counts = np.bincount(labels, minlength=6)
    for i in range(6):
        print(f"Label {i}: {counts[i]}")
