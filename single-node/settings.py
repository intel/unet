#!/usr/bin/env python
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

import psutil
import os

DATA_PATH = os.path.join("../../data/decathlon/144x144/")
DATA_FILENAME = "Task01_BrainTumour.h5"
OUT_PATH = os.path.join("./output/")
INFERENCE_FILENAME="unet_model_for_inference.hdf5"

EPOCHS = 30
BATCH_SIZE = 128  # If the batch size is to small, then training is unstable
LEARNING_RATE = 0.00005  # 0.00005
PRINT_MODEL = True

BLOCKTIME = 1
NUM_INTER_THREADS = 1
# Default is to use the number of physical cores available
NUM_INTRA_THREADS = psutil.cpu_count(logical=False)

CHANNELS_FIRST = False
USE_KERAS_API = True   # If true, then use Keras API. Otherwise, use tf.keras
USE_UPSAMPLING = True  # If true, then use bilinear interpolation. Otherwise, transposed convolution
