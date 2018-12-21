#!/usr/bin/env bash

# -*- coding: utf-8 -*-
#
# Copyright (c) 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: EPL-2.0
#

# You'll need to download the Decathlon dataset.
# You can get the data here:  http://medicaldecathlon.com/
# Currently it is on Google Drive:
#   https://drive.google.com/drive/folders/1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--2
# Download Task01_BrainTumour.tar
# Then untar to the Decathlon directory.

# tar -xvf  Task01_BrainTumour.tar
#

DECATHLON_DIR="../../data/decathlon/Task01_BrainTumour/"
HDF5_DIR="../../data/decathlon/"
IMG_SIZE=128
MODEL_OUTPUT_DIR="./output/"

echo "Converting Decathlon raw data to HDF5 file."
# Run Python script to convert to a single HDF5 file
python convert_raw_to_hdf5.py --data_path $DECATHLON_DIR \
        --save_path $HDF5_DIR --resize=$IMG_SIZE

echo "Run U-Net training on BraTS Decathlon dataset"
# Run training script
# The settings.py file contains the model training.
python train.py --data_path $HDF5_DIR/$IMG_SIZEx$IMG_SIZE \
        --output_path $MODEL_OUTPUT_DIR
