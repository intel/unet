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

import os
import argparse
import psutil

num_data_loaders = 2
num_prefetched_batches = 3

parser = argparse.ArgumentParser(
    description="Train 3D U-Net model", add_help=True,
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--bz",
                    type=int,
                    default=16,
                    help="Batch size")
parser.add_argument("--patch_height",
                    type=int,
                    default=144,
                    help="Size of the 3D patch height")
parser.add_argument("--patch_width",
                    type=int,
                    default=144,
                    help="Size of the 3D patch width")
parser.add_argument("--patch_depth",
                    type=int,
                    default=144,
                    help="Size of the 3D patch depth")
parser.add_argument("--lr",
                    type=float,
                    default=0.01,
                    help="Learning rate")
parser.add_argument("--featuremaps",
                    type=int,
                    default=16,
                    help="Number of feature maps in the first convolutional layer")
parser.add_argument("--train_test_split",
                    type=float,
                    default=0.85,
                    help="Train/test split (0-1)")
parser.add_argument("--validate_test_split",
                    type=float,
                    default=0.50,
                    help="Valdiation/test split (0-1)")
parser.add_argument("--epochs",
                    type=int,
                    default=40,
                    help="Number of epochs")
parser.add_argument("--intraop_threads",
                    type=int,
                    default=max(
                        len(psutil.Process().cpu_affinity())-num_data_loaders, 2),
                    help="Number of intraop threads")
parser.add_argument("--keras_api",
                    help="use keras instead of tf.keras",
                    action="store_true",
                    default=True)
parser.add_argument("--interop_threads",
                    type=int,
                    default=1,
                    help="Number of interop threads")
parser.add_argument("--blocktime",
                    type=int,
                    default=1,
                    help="Block time for CPU threads")
parser.add_argument("--number_input_channels",
                    type=int,
                    default=1,
                    help="Number of input channels")
parser.add_argument("--print_model",
                    action="store_true",
                    default=False,
                    help="Print the summary of the model layers")
parser.add_argument("--use_upsampling",
                    action="store_true",
                    default=False,
                    help="Use upsampling instead of transposed convolution")
datapath = "../../data/decathlon/Task01_BrainTumour/"
parser.add_argument("--data_path",
                    default=datapath,
                    help="Root directory for Medical Decathlon dataset")
parser.add_argument("--saved_model",
                    default="./saved_model/3d_unet_decathlon.hdf5",
                    help="Save model to this path")
parser.add_argument("--random_seed",
                    default=816,
                    help="Random seed")

args = parser.parse_args()

args.num_data_loaders = num_data_loaders
args.num_prefetched_batches = num_prefetched_batches
