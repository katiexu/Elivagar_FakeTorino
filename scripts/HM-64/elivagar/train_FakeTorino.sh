#!/bin/bash

# [HM-64 training] Use 100 epochs for the 16-qubit task; save final full-training-set metrics and use CUDA when available.
python train_elivagar_circuits.py --dataset HM-64 --circs_dir ./experiments/HM-64/80_params/ibm_FakeTorino/elivagar --device_name ibm_FakeTorino --encoding_type angle --noise_importance 0.25 --num_data_for_rep_cap 32 --num_params_for_rep_cap 32 --num_cdcs 16 --num_circs 2500 --num_candidates_per_circ 100 --num_runs_per_circ 5 --num_epochs 100 --batch_size 32 --learning_rate 0.01 --save_train_metrics --use_gpu
