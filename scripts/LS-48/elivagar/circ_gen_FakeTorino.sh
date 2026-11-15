#!/bin/bash

# [LS-48 FakeBackend] Generate this independent 12-qubit candidate-circuit pool from the FakeTorino snapshot.
python generate_device_aware_circuits.py --target_dataset LS-48 --num_circs 2500 --device_name ibm_FakeTorino --save_dir ./experiments/LS-48/60_params/ibm_FakeTorino/elivagar --temp 0.5 --add_rotations
