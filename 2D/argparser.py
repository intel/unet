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

"""
This module just reads parameters from the command line.
"""

import argparse
import settings    # Use the custom settings.py file for default parameters
import os

parser = argparse.ArgumentParser(
    description="Trains 2D U-Net model (Keras/TF) on BraTS dataset.",
    add_help=True, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--data_path", default=settings.DATA_PATH,
                    help="the path to the data")
parser.add_argument("--data_filename", default=settings.DATA_FILENAME,
                    help="the HDF5 data filename")
parser.add_argument("--output_path", default=settings.OUT_PATH,
                    help="the folder to save the model and checkpoints")
parser.add_argument("--inference_filename", default=settings.INFERENCE_FILENAME,
                    help="the Keras inference model filename")
parser.add_argument("--use_upsampling",
                    help="use upsampling instead of transposed convolution",
                    action="store_true", default=settings.USE_UPSAMPLING)
parser.add_argument("--num_threads", type=int,
                    default=settings.NUM_INTRA_THREADS,
                    help="the number of threads")
parser.add_argument(
    "--num_inter_threads",
    type=int,
    default=settings.NUM_INTER_THREADS,
    help="the number of intraop threads")
parser.add_argument("--batch_size", type=int, default=settings.BATCH_SIZE,
                    help="the batch size for training")
parser.add_argument(
    "--blocktime",
    type=int,
    default=settings.BLOCKTIME,
    help="blocktime")
parser.add_argument("--epochs", type=int, default=settings.EPOCHS,
                    help="number of epochs to train")
parser.add_argument(
    "--learningrate",
    type=float,
    default=settings.LEARNING_RATE,
    help="learningrate")
parser.add_argument(
    "--weight_dice_loss",
    type=float,
    default=settings.WEIGHT_DICE_LOSS,
    help="Weight for the Dice loss compared to crossentropy")
parser.add_argument(
    "--featuremaps",
    type=int,
    default=settings.FEATURE_MAPS,
    help="How many feature maps in the model.")
parser.add_argument(
    "--keras_api",
    help="use keras instead of tf.keras",
    action="store_true",
    default=settings.USE_KERAS_API)
parser.add_argument("--channels_first", help="use channels first data format",
                    action="store_true", default=settings.CHANNELS_FIRST)
parser.add_argument("--print_model", help="print the model",
                    action="store_true", default=settings.PRINT_MODEL)
parser.add_argument("--use_dropout", help="add spatial dropout layers 3/4",
                    action="store_true", default=settings.USE_DROPOUT)
parser.add_argument("--use_augmentation", help="use data augmentation on training images",
                    action="store_true", default=settings.USE_AUGMENTATION)

args = parser.parse_args()

# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Get rid of the AVX, SSE warnings

os.environ["KMP_BLOCKTIME"] = str(args.blocktime)
os.environ["KMP_AFFINITY"] = "granularity=fine,compact,1,0"

os.environ["OMP_NUM_THREADS"] = str(args.num_threads)
os.environ["INTRA_THREADS"] = str(args.num_threads)
os.environ["INTER_THREADS"] = str(args.num_inter_threads)
os.environ["KMP_SETTINGS"] = "0"  # Show the settings at runtime
