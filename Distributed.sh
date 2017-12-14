#!/bin/sh
# ----------------------------------------------------------------------------
# Copyright 2017 Intel
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

# We assume that there has been a conda environment setup named 'tf'.
# To create a new conda environment:
# conda create -n tf -c intel python=2 pip numpy
# source activate tf
# pip install https://anaconda.org/intel/tensorflow/1.4.0/download/tensorflow-1.4.0-cp27-cp27mu-linux_x86_64.whl
# pip install keras
# You'll also need whatever other packages your script relies on (e.g. opencv, simpleITK, h5py)
# pip install h5py opencv-python simpleITK tqdm

# Activate the correct Tensorflow environment (conda)
source activate tf
# Run the distributed tensorflow
# We flush messages immediately rather than buffering them.
# All messages go to the local training.log file
if [ $2 = "localhost" ]; then
	stdbuf -oL numactl -p 1 python $1train_dist.py --job_name="ps" --task_index=0 > $1training.log
else
	stdbuf -oL numactl -p 1 python $1train_dist.py --job_name="worker" --task_index=$3 > $1training.log
fi


