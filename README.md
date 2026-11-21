# Élivágar: Efficient Quantum Circuit Search for Classification

Élivágar is a training-free, device- and noise-aware framework for Quantum Circuit Search (QCS) for classification tasks.

## Setup

To setup the environment required to run Élivágar, first clone this repository:

```
git clone https://github.com/katiexu/Elivagar_FakeTorino.git
```

Then, create a new virtual environment, and activate it:

```
python -m venv elivagar_venv
elivagar_venv/Scripts/activate
```

Install all of the required packages via pip:

```
pip install -r requirements.txt
```

Now, install the Pytorch-Tensor-Train-Network package:

```
cd qtn_vqc/Pytorch-Tensor-Train-Network
python -m pip install --no-deps .
cd ../..
```

`--no-deps` installs the local `tc` package without changing the versions pinned by `requirements.txt`.

Setup should now be complete.

## Example usage

As an example, we can use Élivágar to search for a circuit for the Moons dataset, targeting the IBM-Osaka device. To do so, we first generate candidate circuits for the target dataset-device combination:

```
./scripts/moons/elivagar/circ_gen_osaka.sh
```

Next, we compute the Clifford Noise Resilience (CNR) of each of the generated candidate circuits to estimate circuit noise robustness:

```
./scripts/moons/elivagar/cnr_osaka.sh
```

We then compute the representation capacity (RepCap) of candidate circuits, to estimate circuit performance:

```
./scripts/moons/elivagar/repcap_osaka.sh
```

Next, we can compute composite scores using the CNR and RepCap scores for each circuit, select the circuits to be trained, and train them on the Moons dataset:

```
./scripts/moons/elivagar/train_osaka.sh
```

Finally, we can evaluate the trained circuits on a withheld test set from the Moons dataset to check circuit performance:

```
./scripts/moons/elivagar/eval_osaka.sh
```

## Running with IBM fake backends

The legacy `elivagar` environment can continue to use the versions in `requirements.txt` for circuit generation, CNR, RepCap, training, and inference. Exporting a FakeBackend snapshot requires newer Qiskit packages, so do that once in a separate environment rather than installing them into `elivagar`:

```
conda create -n fake-torino python=3.10 -y
conda activate fake-torino
python -m pip install qiskit-ibm-runtime==0.52.0 qiskit-aer==0.17.2
```

From the repository root, export the local FakeTorino hardware and noise snapshot (no IBM account is required):

```
python extract_fake_torino.py --output fake_torino_properties.json
```

The resulting `fake_torino_properties.json` is a static calibration snapshot, not live device data. In the `elivagar` environment, `--device_name ibm_FakeTorino` resolves to this file automatically; use `--fake_backend_json <path>` if the snapshot is stored elsewhere.

CNR now restricts the FakeBackend noise model to the physical qubits in each circuit's mapping by default, retaining the mapped qubits' gate and readout errors. The full MNIST-2 run (2,500 circuits, 32 Clifford replicas) took about **1 hour**, compared with an estimated **20 hours** with the full 133-qubit noise model. Add `--full_noise_model` to the `compute_clifford_nr.py` command to use the full model for comparison; the `cnr.txt` and score output formats and locations remain unchanged.

### MNIST-2 Élivágar workflow

Run the five scripts in order from the repository root, in the `elivagar` environment:

```bash
bash scripts/mnist_2/elivagar/circ_gen_FakeTorino.sh
bash scripts/mnist_2/elivagar/cnr_FakeTorino.sh
bash scripts/mnist_2/elivagar/repcap_FakeTorino.sh
bash scripts/mnist_2/elivagar/train_FakeTorino.sh
bash scripts/mnist_2/elivagar/eval_FakeTorino.sh
```

Circuit generation uses the snapshot's hardware properties. CNR and noisy inference build an `AerSimulator` from its noise model; neither runs a `FakeTorino` backend object or connects to IBM hardware. RepCap and model training are classical/device-independent stages, although training uses the CNR and RepCap scores. RepCap computes reduced density matrices directly from statevectors, so `qutip` is no longer required.

### MNIST-4 digits 0–3 Élivágar workflow

The `mnist_4_0123` dataset contains all MNIST examples labeled 0, 1, 2, or 3 (28,911 total). The generator combines the original MNIST training and test files, then creates a stratified 80:20 split by default; consequently, it does not preserve the original official train/test boundary. To regenerate the four text files from the standard IDX files in `experiment_data/MNIST/raw`, run this from the repository root:

```bash
python experiment_data/make_mnist_4_0123.py
```

The output is written to `experiment_data/mnist_4_0123/` (23,129 training and 5,782 test examples with the default seed and ratio). The existing preprocessing produces 16 features per image and two-column bipolar labels. The split ratio and random seed can be changed with `--train_ratio` and `--seed`.

Run the five FakeTorino scripts in order from the repository root, in the `elivagar` environment:

```bash
bash scripts/mnist_4/elivagar/circ_gen_FakeTorino.sh
bash scripts/mnist_4/elivagar/cnr_FakeTorino.sh
bash scripts/mnist_4/elivagar/repcap_FakeTorino.sh
bash scripts/mnist_4/elivagar/train_FakeTorino.sh
bash scripts/mnist_4/elivagar/eval_FakeTorino.sh
```

These scripts save their circuits and results under `experiments/mnist_4_0123/40_params/ibm_FakeTorino/elivagar/`. RepCap samples 16 examples per class (64 total) for each circuit; training uses the full training split. The evaluation script follows the original MNIST-4 setting and samples 2,000 test examples rather than all 5,782. Remove `--num_test_samples 2000` from that script to evaluate the full test split. In the full run, training took about 9 h 12 min (measured from its log). Estimates from output-file timestamps are about 2 h 22 min for CNR, 43 min for RepCap, and 4 h for evaluation. The Osaka-named scripts are retained unchanged for the original workflow.

### MNIST-10 NLL FakeTorino workflow

The `mnist_10_6_nll` dataset uses 36 features per image, 10 measured qubits/classes, and integer class labels from 0 to 9. Its NPY files contain 50,000 training and 10,000 test examples. Run the five FakeTorino scripts from the repository root in order:

```bash
bash scripts/mnist_10/elivagar/circ_gen_FakeTorino.sh
bash scripts/mnist_10/elivagar/cnr_FakeTorino.sh
bash scripts/mnist_10/elivagar/repcap_FakeTorino.sh
bash scripts/mnist_10/elivagar/train_FakeTorino.sh
bash scripts/mnist_10/elivagar/eval_FakeTorino.sh
```

The scripts use the existing `mnist_10_6_nll` NPY dataset and save outputs under `experiments/mnist_10_6_nll/72_params/ibm_FakeTorino/elivagar/`. Training uses NLL for 50 epochs and, by default, all 50,000 training examples; add `--num_train_samples 512` to the training command for a smaller pilot. Evaluation samples 400 test examples per trial. CNR and noisy inference use the FakeTorino snapshot through `AerSimulator`, not a live IBM backend. Runtime depends heavily on CPU performance and the generated circuits; measure the pilot before scheduling the full 2,500-circuit workflow.

### Linearly separable and hidden-manifold workflows

`LS-48`, `LS-64`, `HM-48`, and `HM-64` are four independent binary-classification tasks. Generate their data with `python experiment_data/generate_linearly_separable.py` and `python experiment_data/generate_hidden_manifold.py`. The scripts keep the original random streams and write only the 48- and 64-feature datasets to `experiment_data/{LS,HM}-{48,64}/`. Each task has 320 training and 80 test examples, and the original validation split is merged into training. The 48-feature tasks use 12 logical qubits, and the 64-feature tasks use 16. The registered circuit configurations use 60/160 trainable parameters respectively (the 16-qubit tasks were raised from 80 to 160), one data repetition, one measured qubit, and 48/64 embedding gates.

Each task has a separate, five-stage FakeTorino workflow. Run its scripts from the repository root in order:

| Task | Circuit generation | CNR | RepCap | Train | Noisy evaluation |
|---|---|---|---|---|---|
| LS-48 | `bash scripts/LS-48/elivagar/circ_gen_FakeTorino.sh` | `bash scripts/LS-48/elivagar/cnr_FakeTorino.sh` | `bash scripts/LS-48/elivagar/repcap_FakeTorino.sh` | `bash scripts/LS-48/elivagar/train_FakeTorino.sh` | `bash scripts/LS-48/elivagar/eval_FakeTorino.sh` |
| LS-64 | `bash scripts/LS-64/elivagar/circ_gen_FakeTorino.sh` | `bash scripts/LS-64/elivagar/cnr_FakeTorino.sh` | `bash scripts/LS-64/elivagar/repcap_FakeTorino.sh` | `bash scripts/LS-64/elivagar/train_FakeTorino.sh` | `bash scripts/LS-64/elivagar/eval_FakeTorino.sh` |
| HM-48 | `bash scripts/HM-48/elivagar/circ_gen_FakeTorino.sh` | `bash scripts/HM-48/elivagar/cnr_FakeTorino.sh` | `bash scripts/HM-48/elivagar/repcap_FakeTorino.sh` | `bash scripts/HM-48/elivagar/train_FakeTorino.sh` | `bash scripts/HM-48/elivagar/eval_FakeTorino.sh` |
| HM-64 | `bash scripts/HM-64/elivagar/circ_gen_FakeTorino.sh` | `bash scripts/HM-64/elivagar/cnr_FakeTorino.sh` | `bash scripts/HM-64/elivagar/repcap_FakeTorino.sh` | `bash scripts/HM-64/elivagar/train_FakeTorino.sh` | `bash scripts/HM-64/elivagar/eval_FakeTorino.sh` |

All four tasks generate their own 2,500-circuit candidate pool and compute their own CNR results; no circuit pool is shared between LS and HM. CNR uses 16 Clifford replicas. RepCap uses 16 samples from each class and 32 parameter samples. Training uses batch size 32, saves final full-training-set metrics, and runs for 200 epochs on 12 qubits or 100 epochs on 16 qubits; `--use_gpu` enables CUDA for RepCap and training when available. The CNR and noisy-evaluation scripts select Aer `statevector` to avoid density-matrix memory growth at 12/16 qubits. Evaluation uses the full 80-sample test split.

Measured full runs on a 32-core CPU with an RTX 4080 took about 9.7–9.9 hours for each 12-qubit task. Each 16-qubit task with 160 parameters took about 61 hours, and CNR accounted for about 47 hours of that time. RepCap and training ran on the GPU; CNR and noisy evaluation used Aer `statevector`. Output directories are `experiments/<task>/<60|160>_params/ibm_FakeTorino/elivagar/`.

### Trial outputs and result export

Training writes the selected circuits to `search_100_ibm_FakeTorino/trial_N/`. Each trial contains the circuit files (`gates.txt`, `gate_params.txt`, bounds files, and `qubit_mapping.txt`) and one `run_K/model.pt` per training run. The trial also stores noiseless test metrics in `accs.txt` and `val_losses.txt`, plus noisy test metrics in `noise_sim/ibm_FakeTorino/*_inference_only.txt`, with one line per run. With `--save_train_metrics`, `train_accs.txt` and `train_losses.txt` store each final model's metrics on the full training set; these values are not the per-batch values shown in the training log.

To collect all trials and runs into `results/results_<task>.csv`, run:

```bash
python export_trial_results.py --search_dir experiments/<task>/<N>_params/ibm_FakeTorino/elivagar/search_100_ibm_FakeTorino
```
