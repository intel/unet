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
Takes a trained model and performs inference on a few validation examples.
"""
import os
import sys
import numpy as np
import tensorflow as tf
import keras as K
import argparse

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(
    description="Inference example for trained 2D U-Net model on BraTS.",
    add_help=True, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--inference_filename", default="../output/unet_model_for_decathlon.hdf5",
                    help="the Keras inference model filename")

args = parser.parse_args()

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Get rid of the AVX, SSE warnings
num_threads = 10
num_inter_threads = 1

os.environ["OMP_NUM_THREADS"] = str(num_threads)
os.environ["KMP_BLOCKTIME"] = "1"
os.environ["KMP_AFFINITY"] = "granularity=thread,compact,1,0"

"""
For best CPU speed set the number of intra and inter threads
to take advantage of multi-core systems.
See https://github.com/intel/mkl-dnn
"""
CONFIG = tf.ConfigProto(intra_op_parallelism_threads=num_threads,
                        inter_op_parallelism_threads=num_inter_threads)

SESS = tf.Session(config=CONFIG)

K.backend.set_session(SESS)

def calc_dice(y_true, y_pred, smooth=1.):
    """
    Sorensen Dice coefficient
    """
    y_true = np.round(y_true)
    y_pred = np.round(y_pred)
    numerator = 2.0 * np.sum(y_true * y_pred) + smooth
    denominator = np.sum(y_true) + np.sum(y_pred) + smooth
    coef = numerator / denominator

    return coef


def plot_results(model, img, msk, img_no, png_directory):
    """
    Calculate the Dice and plot the predicted masks for image # img_no
    """
    pred_mask = model.predict(img)

    dice_score = calc_dice(pred_mask, msk)

    print("Dice score for Image #{} = {:.4f}".format(img_no,
                                                     dice_score))

    plt.figure(figsize=(15, 15))
    plt.subplot(1, 3, 1)
    plt.imshow(img[0, :, :, 0], cmap="bone", origin="lower")
    plt.axis("off")
    plt.title("MRI Input", fontsize=20)
    plt.subplot(1, 3, 2)
    plt.imshow(msk[0, :, :, 0], origin="lower")
    plt.axis("off")
    plt.title("Ground truth", fontsize=20)
    plt.subplot(1, 3, 3)
    plt.imshow(pred_mask[0, :, :, 0], origin="lower")
    plt.axis("off")
    plt.title("Prediction\nDice = {:.4f}".format(dice_score), fontsize=20)

    plt.tight_layout()

    png_name = os.path.join(png_directory, "pred{}.png".format(img_no))
    plt.savefig(png_name, bbox_inches="tight", pad_inches=0)
    print("Saved png file to {}".format(png_name))


if __name__ == "__main__":

    print("TensorFlow version: {}".format(tf.__version__))
    print("Intel MKL-DNN is enabled = {}".format(tf.pywrap_tensorflow.IsMklEnabled()))

    # Load data
    # You can create this Numpy datafile by running the create_validation_sample.py script
    sample_datafile = os.path.join("data", "validation_data.npz")
    try:
        data_file = np.load(sample_datafile)
    except IOError:
        print("Can't find {}. Please run `python create_validation_sample.py` to generate the sample datafile.".format(
            sample_datafile))
        sys.exit()

    imgs_validation = data_file["imgs_validation"]
    msks_validation = data_file["msks_validation"]
    img_indicies = data_file["indicies_validation"]

    print("Using Keras model: {}".format(args.inference_filename))

    # Load model
    model = K.models.load_model(args.inference_filename, compile=False)

    # Create output directory for images
    png_directory = "inference_examples_keras"
    if not os.path.exists(png_directory):
        os.makedirs(png_directory)

    # Plot some results
    # The plots will be saved to the png_directory
    for idx, img_index in enumerate(img_indicies):
        plot_results(model, imgs_validation[[idx], ],
                     msks_validation[[idx], ],
                     img_index, png_directory)
