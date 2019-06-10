#!/usr/bin/env python
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
Takes a trained model and performs inference on a few validation examples.
"""
from model import unet
import os

import numpy as np
import tensorflow as tf
import keras as K
import settings
import argparse
import h5py

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")


parser = argparse.ArgumentParser(
    description="Inference example for trained 2D U-Net model on BraTS.",
    add_help=True, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--data_path", default=settings.DATA_PATH,
                    help="the path to the data")
parser.add_argument("--data_filename", default=settings.DATA_FILENAME,
                    help="the HDF5 data filename")
parser.add_argument("--output_path", default=settings.OUT_PATH,
                    help="the folder to save the model and checkpoints")
parser.add_argument("--inference_filename", default=settings.INFERENCE_FILENAME,
                    help="the Keras inference model filename")

parser.add_argument("--output_pngs", default="inference_examples",
                    help="the directory for the output prediction pngs")

parser.add_argument("--intraop_threads", default=settings.NUM_INTRA_THREADS,
                    type=int, help="Number of intra-op-parallelism threads")
parser.add_argument("--interop_threads", default=settings.NUM_INTER_THREADS,
                    type=int, help="Number of inter-op-parallelism threads")

args = parser.parse_args()

# Optimize CPU threads for TensorFlow
CONFIG = tf.ConfigProto(
    inter_op_parallelism_threads=args.interop_threads,
    intra_op_parallelism_threads=args.intraop_threads)

SESS = tf.Session(config=CONFIG)
K.backend.set_session(SESS)


def calc_dice(target, prediction, smooth=0.01):
    """
    Sorenson Dice
    \frac{  2 \times \left | T \right | \cap \left | P \right |}{ \left | T \right | +  \left | P \right |  }
    where T is ground truth (target) mask and P is the prediction mask
    """
    prediction = np.round(prediction)

    numerator = 2.0 * np.sum(target * prediction) + smooth
    denominator = np.sum(target) + np.sum(prediction) + smooth
    coef = numerator / denominator

    return coef


def calc_soft_dice(target, prediction, smooth=0.01):
    """
    Sorensen (Soft) Dice coefficient - Don't round preictions
    """
    numerator = 2.0 * np.sum(target * prediction) + smooth
    denominator = np.sum(target) + np.sum(prediction) + smooth
    coef = numerator / denominator

    return coef


def plot_results(model, imgs_validation, msks_validation,
                 img_no, png_directory):
    """
    Calculate the Dice and plot the predicted masks for image # img_no
    """

    img = imgs_validation[[img_no], ]
    msk = msks_validation[[img_no], ]

    pred_mask = model.predict(img)

    plt.figure(figsize=(10, 10))
    plt.subplot(1, 3, 1)
    plt.imshow(img[0, :, :, 0], cmap="bone", origin="lower")
    plt.title("MRI")
    plt.axis("off")
    plt.subplot(1, 3, 2)
    plt.imshow(msk[0, :, :, 0], origin="lower")
    plt.title("Ground Truth")
    plt.axis("off")
    plt.subplot(1, 3, 3)
    plt.imshow(pred_mask[0, :, :, 0], origin="lower")
    plt.title("Prediction\n(Dice = {:.4f})".format(calc_dice(msk, pred_mask)))
    plt.axis("off")

    png_filename = os.path.join(png_directory, "pred_{}.png".format(img_no))
    plt.savefig(png_filename, bbox_inches="tight", pad_inches=0)
    print("Dice {:.4f}, Soft Dice {:.4f}, Saved png file to: {}".format(
        calc_dice(msk, pred_mask), calc_soft_dice(msk, pred_mask), png_filename))


if __name__ == "__main__":

    data_filename = os.path.join(args.data_path, args.data_filename)
    model_filename = os.path.join(args.output_path, args.inference_filename)

    # Load data
    df = h5py.File(data_filename, "r")
    imgs_testing = df["imgs_testing"]
    msks_testing = df["msks_testing"]
    files_testing = df["testing_input_files"]

    # Load model
    unet_model = unet()
    model = unet_model.load_model(model_filename)

    # Create output directory for images
    png_directory = "inference_examples"
    if not os.path.exists(png_directory):
        os.makedirs(png_directory)

    # Plot some results
    # The plots will be saved to the png_directory
    # Just picking some random samples.
    indicies_testing = [40, 61, 102, 210, 371,
                        400, 1093, 2222, 3540, 4485,
                        5566, 5675, 6433]


    for idx in indicies_testing:
        plot_results(model, imgs_testing, msks_testing,
                     idx, args.output_pngs)
