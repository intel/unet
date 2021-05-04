#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2020 Intel Corporation
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

import settings
import argparse

parser = argparse.ArgumentParser(
    description="Train 3D U-Net model", add_help=True,
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--data_path",
                    default=settings.DATA_PATH,
                    help="Root directory for BraTS dataset")
parser.add_argument("--csv_list",
                    default=settings.CSV_LIST,
                    help="CSV file with image and label filenames")
parser.add_argument("--epochs",
                    type=int,
                    default=settings.EPOCHS,
                    help="Number of epochs")
parser.add_argument("--saved_model_name",
                    default=settings.SAVED_MODEL_NAME,
                    help="Save model to this path")
parser.add_argument("--batch_size",
                    type=int,
                    default=settings.BATCH_SIZE,
                    help="Training batch size")
parser.add_argument("--tile_height",
                    type=int,
                    default=settings.TILE_HEIGHT,
                    help="Size of the 3D patch height")
parser.add_argument("--tile_width",
                    type=int,
                    default=settings.TILE_WIDTH,
                    help="Size of the 3D patch width")
parser.add_argument("--tile_depth",
                    type=int,
                    default=settings.TILE_DEPTH,
                    help="Size of the 3D patch depth")
parser.add_argument("--number_input_channels",
                    type=int,
                    default=settings.NUMBER_INPUT_CHANNELS,
                    help="Number of input channels")
parser.add_argument("--number_output_classes",
                    type=int,
                    default=settings.NUMBER_OUTPUT_CLASSES,
                    help="Number of output classes/channels")
parser.add_argument("--train_test_split",
                    type=float,
                    default=settings.TRAIN_TEST_SPLIT,
                    help="Train/test split (0-1)")
parser.add_argument("--validate_test_split",
                    type=float,
                    default=settings.VALIDATE_TEST_SPLIT,
                    help="Validation/test split (0-1)")
parser.add_argument("--print_model",
                    action="store_true",
                    default=settings.PRINT_MODEL,
                    help="Print the summary of the model layers")
parser.add_argument("--filters",
                    type=int,
                    default=settings.FILTERS,
                    help="Number of filters in the first convolutional layer")
parser.add_argument("--use_upsampling",
                    action="store_true",
                    default=settings.USE_UPSAMPLING,
                    help="Use upsampling instead of transposed convolution")
parser.add_argument("--random_seed",
                    default=settings.RANDOM_SEED,
                    help="Random seed for determinism")

args = parser.parse_args()
