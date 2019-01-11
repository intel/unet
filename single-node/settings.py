#!/usr/bin/env python
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

import psutil  # pip install psutil
import os

DATA_PATH = os.path.join("../../data/decathlon/144x144/")
DATA_FILENAME = "Task01_BrainTumour.h5"
OUT_PATH = os.path.join("./output/")
INFERENCE_FILENAME="unet_model_for_decathlon.hdf5"

EPOCHS = 30  # Number of epochs to train

"""
If the batch size is too small, then training is unstable.
I believe this is because we are using 2D slicewise model.
There are more slices without tumor than with tumor in the
dataset so the data may become imbalanced if we choose too
small of a batch size. There are, of course, many ways
to handle imbalance training samples, but if we have
enough memory, it is easiest just to select a sufficiently
large batch size to make sure we have a few slices with
tumors in each batch.
"""
BATCH_SIZE = 128

# Using Adam optimizer
LEARNING_RATE = 0.0001 #0.00005  # 0.00005
WEIGHT_DICE_LOSS = 0.9  # Combined loss weight for dice versus BCE

FEATURE_MAPS = 32 # 32 is a good number, but requires about 16 GB of memory
PRINT_MODEL = True  # Print the model

# CPU specific parameters for multi-threading.
# These can help take advantage of multi-core CPU systems
# and significantly boosts training speed with MKL-DNN TensorFlow.
BLOCKTIME = 1
NUM_INTER_THREADS = 1
# Default is to use the number of physical cores available

# Figure out how many physical cores we have available
# Set floor to at least 2 threads
NUM_INTRA_THREADS = max(len(psutil.Process().cpu_affinity()),2)

CHANNELS_FIRST = False
USE_KERAS_API = True   # If true, then use Keras API. Otherwise, use tf.keras
# 28 DEC 2018: tf.keras has some bugs in the use of HDF5 and with the custom
# loss function. Recommend to use Keras API when in doubt.
USE_UPSAMPLING = True  # If true, then use bilinear interpolation. Otherwise, transposed convolution
USE_AUGMENTATION = False # Use data augmentation during training
USE_DROPOUT = False  # Use spatial dropout in model
