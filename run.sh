#!/bin/bash

# run.sh

# --
# Setup environment

conda create -y -n td_env python=3.7
conda activate td_env

pip install numpy
pip install scipy

# --
# Test

python test.py --dim 4096 --k 512

python test.py --dim 8192 --k 1024

python test.py --dim 16284 --k 1024