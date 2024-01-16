#!/bin/bash

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

taskset -c 1 python profile_features_elasticc.py
