#!/bin/bash

# [HM-48 FakeBackend] Use 16 Clifford replicas and Aer statevector simulation for the 12-qubit circuits.
python compute_clifford_nr.py --dataset HM-48 --num_circs 2500 --num_cdcs 16 --device_name ibm_FakeTorino --use_qubit_mapping --save_cnr_scores --encoding_type angle --circs_dir ./experiments/HM-48/60_params/ibm_FakeTorino/elivagar --aer_method statevector
