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

"""
This module just reads parameters from the command line.
"""

import argparse
import settings    # Use the custom settings.py file for default parameters
import os

parser = argparse.ArgumentParser(
    description="2D U-Net model (Keras/TF) on BraTS Decathlon dataset.",
    add_help=True, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--data_path", default=settings.DATA_PATH,
                    help="The path to the Medical Decathlon directory")
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
parser.add_argument("--num_inter_threads", type=int,
                    default=settings.NUM_INTER_THREADS,
                    help="the number of intraop threads")
parser.add_argument("--batch_size", type=int, default=settings.BATCH_SIZE,
                    help="the batch size for training")
parser.add_argument("--split", type=float, default=settings.TRAIN_TEST_SPLIT,
                    help="Train/testing split for the data")
parser.add_argument("--seed", type=int, default=settings.SEED,
                    help="Seed for random number generation")
parser.add_argument("--crop_dim", type=int, default=settings.CROP_DIM,
                    help="Size to crop images (square, in pixels). If -1, then no cropping.")
parser.add_argument("--blocktime", type=int,
                    default=settings.BLOCKTIME,
                    help="blocktime")
parser.add_argument("--epochs", type=int,
                    default=settings.EPOCHS,
                    help="number of epochs to train")
parser.add_argument("--learningrate", type=float,
                    default=settings.LEARNING_RATE,
                    help="learningrate")
parser.add_argument("--weight_dice_loss", type=float,
                    default=settings.WEIGHT_DICE_LOSS,
                    help="Weight for the Dice loss compared to crossentropy")
parser.add_argument("--featuremaps", type=int,
                    default=settings.FEATURE_MAPS,
                    help="How many feature maps in the model.")
parser.add_argument("--use_pconv", help="use partial convolution based padding",
                    action="store_true",
                    default=settings.USE_PCONV)
parser.add_argument("--channels_first", help="use channels first data format",
                    action="store_true", default=settings.CHANNELS_FIRST)
parser.add_argument("--print_model", help="print the model",
                    action="store_true",
                    default=settings.PRINT_MODEL)
parser.add_argument("--use_dropout",
                    default=settings.USE_DROPOUT,
                    help="add spatial dropout layers 3/4",
                    action="store_true",
                    )
parser.add_argument("--use_augmentation",
                    default=settings.USE_AUGMENTATION,
                    help="use data augmentation on training images",
                    action="store_true")
parser.add_argument("--output_pngs",
                    default="inference_examples",
                    help="the directory for the output prediction pngs")
parser.add_argument("--input_filename",
                    help="Name of saved TensorFlow model directory",
                    default=os.path.join(settings.OUT_PATH,settings.INFERENCE_FILENAME))

args = parser.parse_args()


