#!/bin/bash

# [HM-64 RepCap] RepCap is device-independent; use this task's circuit pool, 16 samples per class, and CUDA when available.
python compute_rep_cap.py --dataset HM-64 --num_circs 2500 --encoding_type angle --circs_dir ./experiments/HM-64/160_params/ibm_FakeTorino/elivagar --num_samples_per_class 16 --num_param_samples 32 --use_gpu
