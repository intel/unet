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
# Download the tar file "Task01_BrainTumour.tar"
# Then untar to the Decathlon directory.

# tar -xvf  Task01_BrainTumour.tar
#

DECATHLON_DIR=${1:-"../../data/decathlon/Task01_BrainTumour/"}
HDF5_DIR=${2:-"../../data/decathlon/"}
IMG_SIZE=${3:-128}
MODEL_OUTPUT_DIR=${4:-"./output/"}
MODEL_OUTPUT_FILENAME=${5:-"decathlon_brats.h5"}

clear
echo "Script to train Decathlon Brain Tumor Segmentation (BraTS) U-Net model"
echo "======================================================================"
echo "You need to download the dataset from http://medicaldecathlon.com"
echo "Download the tar file 'Task01_BrainTumour.tar'"
echo "Then extract the data from the tar file"
echo "tar -xvf Task01_BrainTumour.tar"
echo "Make sure to change the DECATHLON_DIR variable in this script"
echo " to wherever you untarred the dataset."

echo "*****************************************"
echo "Step 1 of 3: Convert raw data to HDF5 file"
echo "*****************************************"

echo "Converting Decathlon raw data to HDF5 file."
# Run Python script to convert to a single HDF5 file
python convert_raw_to_hdf5.py --data_path $DECATHLON_DIR \
        --output_filename $MODEL_OUTPUT_FILENAME \
        --save_path $HDF5_DIR --resize=$IMG_SIZE

echo "*****************************************"
echo "Step 2 of 3: Train U-Net on dataset"
echo "*****************************************"

echo "Run U-Net training on BraTS Decathlon dataset"
# Run training script
# The settings.py file contains the model training.
python train.py --data_path ${HDF5_DIR}/${IMG_SIZE}x${IMG_SIZE} \
        --data_filename $MODEL_OUTPUT_FILENAME \
        --output_path $MODEL_OUTPUT_DIR

echo "*****************************************"
echo "Step 3 of 3: Run sample inference script"
echo "*****************************************"

python run_inference.py --model $MODEL_OUTPUT_DIR/unet_model_for_inference.hdf5 \
        --data_path $HDF5_DIR/${IMG_SIZE}x${IMG_SIZE}/$MODEL_OUTPUT_FILENAME
