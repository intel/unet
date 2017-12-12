#!/bin/sh

# Activate the correct Tensorflow environment (conda)
source activate tf
# Run the distributed tensorflow
stdbuf -oL numactl -p 1 python $1train_dist.py --job_name=$2 --task_index=$3 > $1training.log
