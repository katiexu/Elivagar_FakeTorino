#!/bin/bash

# [HM-48 RepCap] RepCap is device-independent; use this task's circuit pool, 16 samples per class, and CUDA when available.
python compute_rep_cap.py --dataset HM-48 --num_circs 2500 --encoding_type angle --circs_dir ./experiments/HM-48/60_params/ibm_FakeTorino/elivagar --num_samples_per_class 16 --num_param_samples 32 --use_gpu
