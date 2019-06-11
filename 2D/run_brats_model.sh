#!/usr/bin/env bash
#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Intel Corporation
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
if [ "$1" == "-h" ]; then
  echo "Usage: `basename $0` [DECATHLON_DIR] [SUBSET_DIR] [IMG_SIZE]" \
       " [MODEL_OUTPUT_DIR] [INFERENCE_FILENAME]"
  echo "Trains a U-Net model based on the Decathlon Brain Tumor" \
       "Segmentation (BraTS) dataset found at: "
  echo "http://medicaldecathlon.com"
  echo " "
  exit 0
fi

DECATHLON_DIR=${1:-"../../data/decathlon"}
SUBSET_DIR=${2:-"Task01_BrainTumour"}
IMG_SIZE=${3:-144}  # This should be a multiple of 16
MODEL_OUTPUT_DIR=${4:-"./output"}
INFERENCE_FILENAME=${6:-"unet_model_for_decathlon.hdf5"}

MODEL_OUTPUT_FILENAME=${SUBSET_DIR}".h5"

NUM_EPOCHS=20  # Number of epochs to train
LEARNING_RATE=0.0001  # 0.00005  Adam optimizer

# 32 feature maps is preferable, but uses about 16 GB of memory.
# You can reduce this to 16 feature maps in order for the
# training to run using less memory.
# You can attempt to turn knobs on feature maps, learning rate,
# and batch size to get better model performance.
FEATURE_MAPS=32  # Number of feature maps in model

clear
echo "Script to train Decathlon Brain Tumor Segmentation (BraTS) U-Net model"
echo "======================================================================"
echo "You need to download the dataset from http://medicaldecathlon.com"
echo "Download the tar file 'Task01_BrainTumour.tar'"
echo "Then extract the data from the tar file"
echo "tar -xvf Task01_BrainTumour.tar"
echo "Make sure to change the DECATHLON_DIR variable in this script"
echo " to wherever you untarred the dataset."

if [[ ! -f ${DECATHLON_DIR}/${SUBSET_DIR}/dataset.json ]] ; then
    echo " "
    echo "ERROR:"
    echo "File '${DECATHLON_DIR}/${SUBSET_DIR}/dataset.json' " \
         "is not there, aborting."
    echo "Please download the Decathlon dataset, extract it, " \
         "and point this script to that directory."
    exit
fi

echo " "
echo "******************************************"
echo "Step 1 of 3: Convert raw data to HDF5 file"
echo "******************************************"

echo "Converting Decathlon raw data to HDF5 file."
# Run Python script to convert to a single HDF5 file
# Resize should be a multiple of 16 because of the way the
# max pooling and upsampling works in U-Net. The rule is
# 2^n where n is the number of max pooling/upsampling concatenations.
python convert_raw_to_hdf5.py --data_path $DECATHLON_DIR/${SUBSET_DIR} \
        --output_filename $MODEL_OUTPUT_FILENAME \
        --save_path $DECATHLON_DIR --resize=$IMG_SIZE

echo " "
echo "***********************************"
echo "Step 2 of 3: Train U-Net on dataset"
echo "***********************************"

echo "Run U-Net training on BraTS Decathlon dataset"
# Run training script
# The settings.py file contains the model training.
python train.py \
        --epochs $NUM_EPOCHS  \
        --learningrate $LEARNING_RATE \
        --data_path $DECATHLON_DIR/${IMG_SIZE}x${IMG_SIZE} \
        --data_filename $MODEL_OUTPUT_FILENAME \
        --output_path $MODEL_OUTPUT_DIR \
        --inference_filename $INFERENCE_FILENAME \
        --featuremaps $FEATURE_MAPS \
        --print_model \
        --keras_api \
        --use_upsampling \
        --use_augmentation \
        --use_dropout

echo " "
echo "****************************************"
echo "Step 3 of 3: Run sample inference script"
echo "****************************************"

python plot_inference_examples.py  \
        --data_path $DECATHLON_DIR/${IMG_SIZE}x${IMG_SIZE} \
        --data_filename $MODEL_OUTPUT_FILENAME \
        --output_path $MODEL_OUTPUT_DIR \
        --inference_filename $INFERENCE_FILENAME
