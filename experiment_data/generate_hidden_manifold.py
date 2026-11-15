# Copyright 2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generate hidden-manifold datasets for Elivagar."""

from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split


def neural_net(x, W, v):
    """Transforms inputs via a single-layer neural network.

    Args:
        x (ndarray): input of shape (manifold_dimension,)
        W (ndarray): input-to-hidden weight matrix of shape (manifold_dimension, manifold_dimension)
        v (ndarray): hidden-to-output weight matrix of shape (manifold_dimension,)
    """
    return np.dot(v, np.tanh(W @ x) / np.sqrt(W.shape[0]))


def nonlinearity(X, biases):
    """Element-wise nonlinearity.

    Args:
        X (ndarray): inputs of shape (n_samples, n_features)
        biases (ndarray): biases of shape (n_features,)
    """
    return np.tanh(X - biases)


def generate_hidden_manifold_model(n_samples, n_features, manifold_dimension):
    """Data generation procedure for the 'hidden manifold model'."""
    # feature matrix F controls the embedding of the manifold
    F = np.random.normal(size=(manifold_dimension, n_features))

    # Gaussian matrix samples original inputs from the lower-dimensional manifold
    C = np.random.normal(size=(n_samples, manifold_dimension), loc=0, scale=1)

    # embed data, adding an element-wise nonlinearity
    biases = 2 * np.random.uniform(size=(n_features,)) - 1
    X = nonlinearity(C @ F / np.sqrt(manifold_dimension), biases)

    # define labels via a neural network
    W = np.random.normal(size=(manifold_dimension, manifold_dimension))
    v = np.random.normal(size=(manifold_dimension,))
    y = np.array([neural_net(c, W, v) for c in C])

    # post-process the labels to get balanced classes
    y = y - np.median(y)
    y = np.array([-1 if y_ < 0 else 1 for y_ in y])
    assert len(X[y == 1]) == n_samples // 2
    assert len(X[y == -1]) == n_samples // 2

    return X, y


def main():
    current_path = Path(__file__).parent.absolute()
    print(f"Current path: {current_path}")

    # Keep the original seed, dimension order, legacy RNG calls, and split
    # sequence so the 48- and 64-feature samples match the original generator.
    np.random.seed(3)
    manifold_dimension = 6
    n_samples = 400
    save_dims = [48, 64]

    # [原始代码，已停用] The old script created this directory and wrote
    # combined CSV files under hidden_manifold/.
    # import os
    # os.makedirs(current_path / "hidden_manifold", exist_ok=True)

    for n_features in [16, 32, 48, 64]:
        X, y = generate_hidden_manifold_model(n_samples, n_features, manifold_dimension)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1)

        if n_features not in save_dims:
            continue

        # [Elivagar] Keep all 320 samples from the original 80% training split:
        # combine its 288 train and 32 validation examples; do not save a val set.
        X_train = np.concatenate((X_train, X_val), axis=0)
        y_train = np.concatenate((y_train, y_val), axis=0)
        output_dir = current_path / f"HM-{n_features}"
        output_dir.mkdir(parents=True, exist_ok=True)
        for filename, values in (
            ("x_train.txt", X_train),
            ("y_train.txt", y_train),
            ("x_test.txt", X_test),
            ("y_test.txt", y_test),
        ):
            np.savetxt(output_dir / filename, values, fmt="%.18e")

        # [原始代码，已停用] Preserve the legacy CSV output for reference.
        # name_train = f"{current_path}/hidden_manifold/QML_Hidden_{n_features}d_train.csv"
        # np.savetxt(name_train, np.c_[X_train, y_train], delimiter=",")
        # name_val = f"{current_path}/hidden_manifold/QML_Hidden_{n_features}d_val.csv"
        # np.savetxt(name_val, np.c_[X_val, y_val], delimiter=",")
        # name_test = f"{current_path}/hidden_manifold/QML_Hidden_{n_features}d_test.csv"
        # np.savetxt(name_test, np.c_[X_test, y_test], delimiter=",")

        print(f"Generated hidden-manifold data with {n_features} features in {output_dir}.")


if __name__ == "__main__":
    main()
