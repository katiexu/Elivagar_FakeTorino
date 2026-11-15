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

"""Generate linearly separable datasets for Elivagar."""

from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split


def generate_linearly_separable(n_samples, n_features, margin):
    """Generate samples outside a margin around the all-ones hyperplane."""
    X = 2 * np.random.rand(2 * n_samples, n_features) - 1
    w = np.ones(n_features)
    labels = np.where(np.dot(X, w) > 0, -1, 1)
    outside_margin = np.abs(np.dot(X, w)) > margin
    X = X[outside_margin][:n_samples]
    labels = labels[outside_margin][:n_samples]
    return X, labels


def main():
    current_path = Path(__file__).parent.absolute()
    print(f"Current path: {current_path}")

    # Keep the original seed, dimension order, legacy RNG calls, and split
    # sequence so the 48- and 64-feature samples match the original generator.
    np.random.seed(42)
    n_samples = 400
    save_dims = [48, 64]

    # [原始代码，已停用] The old script created this unused directory and
    # wrote combined CSV files under linearly_separable/.
    # import os
    # os.makedirs(current_path / "hidden_manifold", exist_ok=True)
    # os.makedirs("linearly_separable", exist_ok=True)

    for n_features in [16, 32, 48, 64, 80]:
        margin = 0.02 * n_features
        X, y = generate_linearly_separable(n_samples, n_features, margin)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1)

        if n_features not in save_dims:
            continue

        # [Elivagar] Keep all 320 samples from the original 80% training split:
        # combine its 288 train and 32 validation examples; do not save a val set.
        X_train = np.concatenate((X_train, X_val), axis=0)
        y_train = np.concatenate((y_train, y_val), axis=0)
        output_dir = current_path / f"LS-{n_features}"
        output_dir.mkdir(parents=True, exist_ok=True)
        for filename, values in (
            ("x_train.txt", X_train),
            ("y_train.txt", y_train),
            ("x_test.txt", X_test),
            ("y_test.txt", y_test),
        ):
            np.savetxt(output_dir / filename, values, fmt="%.18e")

        # [原始代码，已停用] Preserve the legacy CSV output for reference.
        # name_train = f"{current_path}/linearly_separable/QML_Linear_{n_features}d_train.csv"
        # np.savetxt(name_train, np.c_[X_train, y_train], delimiter=",")
        # name_val = f"{current_path}/linearly_separable/QML_Linear_{n_features}d_val.csv"
        # np.savetxt(name_val, np.c_[X_val, y_val], delimiter=",")
        # name_test = f"{current_path}/linearly_separable/QML_Linear_{n_features}d_test.csv"
        # np.savetxt(name_test, np.c_[X_test, y_test], delimiter=",")

        print(f"Generated linearly separable data with {n_features} features in {output_dir}.")


if __name__ == "__main__":
    main()
