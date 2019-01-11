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


def calc_dice(y_true, y_pred, smooth=1.):
    """
    Sorensen Dice coefficient
    """
    numerator = 2.0 * np.sum(y_true * y_pred) + smooth
    denominator = np.sum(y_true) + np.sum(y_pred) + smooth
    coef = numerator / denominator

    return coef

def dice_coef(y_true, y_pred, axis=(1, 2), smooth=1.):
    """
    Sorenson (Soft) Dice
    \frac{  2 \times \left | T \right | \cap \left | P \right |}{ \left | T \right | +  \left | P \right |  }
    where T is ground truth mask and P is the prediction mask
    """
    intersection = tf.reduce_sum(y_true * y_pred, axis=axis)
    union = tf.reduce_sum(y_true + y_pred, axis=axis)
    numerator = tf.constant(2.) * intersection + smooth
    denominator = union + smooth
    coef = numerator / denominator

    return tf.reduce_mean(coef)


def dice_coef_loss(target, prediction, axis=(1, 2), smooth=1.):
    """
    Sorenson (Soft) Dice loss
    Using -log(Dice) as the loss since it is better behaved.
    Also, the log allows avoidance of the division which
    can help prevent underflow when the numbers are very small.
    """
    intersection = tf.reduce_sum(prediction * target, axis=axis)
    p = tf.reduce_sum(prediction, axis=axis)
    t = tf.reduce_sum(target, axis=axis)
    numerator = tf.reduce_mean(intersection + smooth)
    denominator = tf.reduce_mean(t + p + smooth)
    dice_loss = -tf.log(2.*numerator) + tf.log(denominator)

    return dice_loss


def combined_dice_ce_loss(y_true, y_pred, axis=(1, 2), smooth=1.,
                          weight=0.9):
    """
    Combined Dice and Binary Cross Entropy Loss
    """
    return weight*dice_coef_loss(y_true, y_pred, axis, smooth) + \
        (1-weight)*K.losses.binary_crossentropy(y_true, y_pred)


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

    # Load data
    # You can create this Numpy datafile by running the create_validation_sample.py script
    try:
    	data_file = np.load("validation_data.npz")
    except IOError:
        print("Can't find {}. Please run `python create_validation_sample.py` to generate the sample datafile.".format("validation_data.npz"))
        sys.exit()
        
    imgs_validation = data_file["imgs_validation"]
    msks_validation = data_file["msks_validation"]
    img_indicies = data_file["indicies_validation"]

    print("Using Keras model: {}".format(args.inference_filename))
    
    # Load model
    model = K.models.load_model(args.inference_filename, custom_objects={
        "combined_dice_ce_loss": combined_dice_ce_loss,
        "dice_coef_loss": dice_coef_loss,
        "dice_coef": dice_coef})

    # Create output directory for images
    png_directory = "inference_examples_keras"
    if not os.path.exists(png_directory):
        os.makedirs(png_directory)

    # Plot some results
    # The plots will be saved to the png_directory
    for idx, img_index in enumerate(img_indicies):
    	plot_results(model, imgs_validation[[idx],],
    		     msks_validation[[idx],],
                 img_index, png_directory)
