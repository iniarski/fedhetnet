import os
from os import path
import pathlib

import numpy as np

from sklearn.model_selection import train_test_split


CLASS_SAMPLING = [0.1, 1.0, 3.0, 2.0, 2.0, 8.0, 1.0]


def sample_npz(filepath : str, class_sampling : list[float] = CLASS_SAMPLING, batch_size : int = 256):
    parent_dir = pathlib.Path(filepath).parent.resolve()
    filename = path.basename(filepath)
    target = path.join(parent_dir, "sampled_" + filename)

    data = np.load(filepath)
    X, y = data["X"], data["y"]
    X_sampled, y_sampled = list(), list()
    n_classes = len(class_sampling)
    n_features = X.shape[-1]

    features_by_class = [list() for _ in range(n_classes)]
    labels_by_class = [list() for _ in range(n_classes)]

    n_batches = len(y) // batch_size

    features = X[: n_batches * batch_size].reshape(-1, batch_size, n_features)
    labels = y[: n_batches * batch_size].reshape(-1, batch_size)

    for f, l in zip(features, labels):
        this_class = np.max(l)
        features_by_class[this_class].append(f)
        labels_by_class[this_class].append(l)

    for c in range(n_classes):
        if len(features_by_class[c]) == 0:
            continue

        features = features_by_class[c]
        labels = labels_by_class[c]
        sampling = class_sampling[c]

        repeats, rest = int(sampling), int(len(features) * sampling % 1)
        features_sampled = repeats * features
        labels_sampled = repeats * labels

        if rest > 0:
            idx = np.random.choice(np.arange(n_batches), rest)
            features_rest = [features[i] for i in idx]
            labels_rest = [labels[i] for i in idx]

            features_sampled.extend(features_rest)
            labels_sampled.extend(labels_rest)

        X_sampled.extend(features_sampled)
        y_sampled.extend(labels_sampled)

    np.savez(target, X=np.array(X_sampled), y=np.array(y_sampled))


def make_dataset(
        npz_path: str,
        sample_classes: bool = True,
        class_sampling: list[float] = CLASS_SAMPLING,
        batch_size: int = 256,
        seed: int = 42,
        shuffle: bool = False,
        normal_only: bool = False,
        split: float = 0.8
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    n_classes = len(class_sampling)

    if any(x < 0 for x in class_sampling):
        raise Exception("Expected all elements of class sampling to be positive")
    if (split < 0 or split > 1):
        raise Exception("split should be in range (0, 1)")

    features_by_class = [list() for _ in range(n_classes)]
    labels_by_class = [list() for _ in range(n_classes)]

    if path.isfile(npz_path):
        data = np.load(npz_path)

        X = data['X']
        y = data['y']

        if y.ndim == 1:
            n_batches = len(y) // batch_size

            X = X[: n_batches * batch_size].reshape(-1, batch_size, n_features)
            y = y[: n_batches * batch_size].reshape(-1, batch_size)

        for f, l in zip(X, y):
            _class = np.max(l)
            features_by_class[_class].append(f)
            labels_by_class[_class].append(l)
    else:
        files = [f for f in os.listdir(npz_path) if f.endswith('.npz')]
        for f in files:
            data = np.load(npz_path)

            X = data['X']
            y = data['y']

            if y.ndim == 1:
                n_batches = len(y) // batch_size

                X = X[: n_batches * batch_size].reshape(-1, batch_size, n_features)
                y = y[: n_batches * batch_size].reshape(-1, batch_size)

            for f, l in zip(X, y):
                _class = np.max(l)
                features_by_class[_class].append(f)
                labels_by_class[_class].append(l)

    X_train, Y_train, X_test, Y_test = list(), list(), list(), list()

    for c in range(n_classes):
        if len(features_by_class[c]) == 0:
            continue

        features = features_by_class[c]
        labels = labels_by_class[c]
        x_train, x_test, y_train, y_test = train_test_split(features, labels, train_size=split, random_state=seed, shuffle=False)
        X_test.extend(x_test)
        Y_test.extend(y_test)
        if sample_classes:
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
        else:
            X_train.extend(x_train)
            Y_train.extend(y_train)

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
    from sys import argv

    file = argv[1]

    sample_npz(file)
