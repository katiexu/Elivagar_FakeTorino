#!/bin/bash

# [LS-64 FakeBackend] Use 16 Clifford replicas and Aer statevector simulation for the 16-qubit circuits.
python compute_clifford_nr.py --dataset LS-64 --num_circs 2500 --num_cdcs 16 --device_name ibm_FakeTorino --use_qubit_mapping --save_cnr_scores --encoding_type angle --circs_dir ./experiments/LS-64/80_params/ibm_FakeTorino/elivagar --aer_method statevector
