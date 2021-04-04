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
import psutil
import os

DATA_PATH=os.path.join("/data/medical_decathlon/Task01_BrainTumour")
OUT_PATH = os.path.join("./output/")
INFERENCE_FILENAME = "2d_unet_decathlon"

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
LEARNING_RATE = 0.0001  # 0.00005
WEIGHT_DICE_LOSS = 0.85  # Combined loss weight for dice versus BCE

FEATURE_MAPS = 16
PRINT_MODEL = True  # Print the model

# CPU specific parameters for multi-threading.
# These can help take advantage of multi-core CPU systems
# and significantly boosts training speed with MKL-DNN TensorFlow.
BLOCKTIME = 0
NUM_INTER_THREADS = 1
# Default is to use the number of physical cores available

# Figure out how many physical cores we have available
# Minimum of either the CPU affinity or the number of physical cores
import multiprocessing
NUM_INTRA_THREADS = min(len(psutil.Process().cpu_affinity()), psutil.cpu_count(logical=False))

CROP_DIM=128  # Crop height and width to this size
SEED=816      # Random seed
TRAIN_TEST_SPLIT=0.80 # The train/test split

CHANNELS_FIRST = False
USE_UPSAMPLING = False
USE_AUGMENTATION = True  # Use data augmentation during training
USE_DROPOUT = True  # Use spatial dropout in model
USE_PCONV = False  # If True, Partial Convolution based padding will be used. See https://arxiv.org/pdf/1811.11718.pdf
