#!/bin/bash

# [HM-64 FakeBackend] Generate this independent 16-qubit candidate-circuit pool from the FakeTorino snapshot.
python generate_device_aware_circuits.py --target_dataset HM-64 --num_circs 2500 --device_name ibm_FakeTorino --save_dir ./experiments/HM-64/80_params/ibm_FakeTorino/elivagar --temp 0.5 --add_rotations
