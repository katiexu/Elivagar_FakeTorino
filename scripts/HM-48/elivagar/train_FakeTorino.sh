#!/bin/bash

# [HM-48 training] Keep 200 epochs for the 12-qubit task; save final full-training-set metrics and use CUDA when available.
python train_elivagar_circuits.py --dataset HM-48 --circs_dir ./experiments/HM-48/60_params/ibm_FakeTorino/elivagar --device_name ibm_FakeTorino --encoding_type angle --noise_importance 0.25 --num_data_for_rep_cap 32 --num_params_for_rep_cap 32 --num_cdcs 16 --num_circs 2500 --num_candidates_per_circ 100 --num_runs_per_circ 5 --num_epochs 200 --batch_size 32 --learning_rate 0.01 --save_train_metrics --use_gpu
