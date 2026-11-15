#!/bin/bash

# [HM-48 FakeBackend] Noisy FakeTorino-snapshot inference over all 80 test samples per trained run.
python circuit_inference.py --dataset HM-48 --num_circs 25 --num_runs_per_circ 5 --encoding_type angle --device_name ibm_FakeTorino --circs_dir ./experiments/HM-48/60_params/ibm_FakeTorino/elivagar/search_100_ibm_FakeTorino --results_save_dir noise_sim --transpiler_opt_level 1 --circ_prefix trial --use_qubit_mapping --qubit_mapping_filename qubit_mapping.txt --aer_method statevector
