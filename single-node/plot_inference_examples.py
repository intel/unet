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


# """
# BEGIN - Limit Tensoflow to only use specific GPU
# """
# import os

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"   # 0,1,2,3
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" # Supress Tensforflow debug messages

# # """
# # END - Limit Tensoflow to only use specific GPU
# # """

import os

import numpy as np
import tensorflow as tf
from tensorflow import keras as K
import settings
import argparse
import h5py

import matplotlib.pyplot as plt

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

args = parser.parse_args()

def calc_dice(y_true, y_pred, smooth = 1.):
    """
    Sorensen Dice coefficient
    """
    numerator = 2.0 * np.sum(y_true * y_pred) + smooth
    denominator = np.sum(y_true) + np.sum(y_pred) + smooth
    coef = numerator / denominator

    return coef

def plot_results(model, imgs_test, msks_test, img_no, png_directory):
    """
    Calculate the Dice and plot the predicted masks for image # img_no
    """

    img = imgs_test[[img_no],]
    msk = msks_test[[img_no],]
    pred_mask = model.predict(img)

    dice_score = calc_dice(pred_mask, msk)
    
    print("Dice score for Image #{} = {:.4f}".format(img_no,
                            dice_score))

    plt.figure(figsize=(15,15));
    plt.subplot(1,3,1);
    plt.imshow(img[0,:,:,2], cmap="bone");
    plt.axis("off");
    plt.title("MRI Input");
    plt.subplot(1,3,2);
    plt.imshow(msk[0,:,:,0]);
    plt.axis("off");
    plt.title("Ground truth");
    plt.subplot(1,3,3);
    plt.imshow(pred_mask[0,:,:,0]);
    plt.axis("off");
    plt.title("Prediction\nDice = {:.4f}".format(dice_score));

    plt.tight_layout();
    
    png_name = os.path.join(png_directory, "pred{}.png".format(img_no))
    plt.savefig(png_name, bbox_inches="tight", pad_inches=0)
    print("Saved png file to {}".format(png_name))

if __name__ == "__main__":

    data_fn = os.path.join(args.data_path, args.data_filename)
    model_fn = os.path.join(args.output_path, args.inference_filename)

    # Load data
    df = h5py.File(data_fn, "r")
    imgs_test = df["imgs_test"]
    msks_test = df["msks_test"]

    # Load model
    model = K.models.load_model(model_fn)

    # Create output directory for images
    png_directory = "inference_examples"
    if not os.path.exists(png_directory):
        os.makedirs(png_directory)
        
    plot_results(model, imgs_test, msks_test, 400, png_directory) # Image #400
    plot_results(model, imgs_test, msks_test, 1100, png_directory) # Image #1100
    plot_results(model, imgs_test, msks_test, 5673, png_directory) 
    plot_results(model, imgs_test, msks_test, 6433, png_directory)
    plot_results(model, imgs_test, msks_test, 9003, png_directory)
    plot_results(model, imgs_test, msks_test, 10591, png_directory)
