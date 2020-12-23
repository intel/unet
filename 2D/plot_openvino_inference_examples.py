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
import os

import numpy as np
import tensorflow as tf
import time
from tensorflow import keras as K
import settings
import argparse
from dataloader import DatasetGenerator

from openvino.inference_engine import IECore

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")


parser = argparse.ArgumentParser(
    description="TensorFlow Inference example for trained 2D U-Net model on BraTS.",
    add_help=True, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--data_path", default=settings.DATA_PATH,
                    help="the path to the data")
parser.add_argument("--output_path", default=settings.OUT_PATH,
                    help="the folder to save the model and checkpoints")
parser.add_argument("--inference_filename", default=settings.INFERENCE_FILENAME,
                    help="the TensorFlow inference model filename")
parser.add_argument("--use_pconv",help="use partial convolution based padding",
                    action="store_true",
                    default=settings.USE_PCONV)
parser.add_argument("--output_pngs", default="inference_examples",
                    help="the directory for the output prediction pngs")

parser.add_argument("--intraop_threads", default=settings.NUM_INTRA_THREADS,
                    type=int, help="Number of intra-op-parallelism threads")
parser.add_argument("--interop_threads", default=settings.NUM_INTER_THREADS,
                    type=int, help="Number of inter-op-parallelism threads")
parser.add_argument("--crop_dim", default=settings.CROP_DIM,
                    type=int, help="Crop dimension for images")
parser.add_argument("--seed", default=settings.SEED,
                    type=int, help="Random seed")

args = parser.parse_args()

def test_intel_tensorflow():
    """
    Check if Intel version of TensorFlow is installed
    """
    import tensorflow as tf
    
    print("We are using Tensorflow version {}".format(tf.__version__))
           
    major_version = int(tf.__version__.split(".")[0])
    if major_version >= 2:
       from tensorflow.python import _pywrap_util_port
       print("Intel-optimizations (DNNL) enabled:", _pywrap_util_port.IsMklEnabled())
    else:
       print("Intel-optimizations (DNNL) enabled:", tf.pywrap_tensorflow.IsMklEnabled()) 

test_intel_tensorflow()


def calc_dice(target, prediction, smooth=0.0001):
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


def calc_soft_dice(target, prediction, smooth=0.0001):
    """
    Sorensen (Soft) Dice coefficient - Don't round preictions
    """
    numerator = 2.0 * np.sum(target * prediction) + smooth
    denominator = np.sum(target) + np.sum(prediction) + smooth
    coef = numerator / denominator

    return coef


def plot_results(ds, idx, png_directory, exec_net, input_layer_name, output_layer_name):
    
    dt = ds.get_dataset().take(1).as_numpy_iterator()  # Get some examples (use different seed for different samples)

    plt.figure(figsize=(10,10))

    for img, msk in dt:

        plt.subplot(1, 3, 1)
        plt.imshow(img[idx, :, :, 0], cmap="bone", origin="lower")
        plt.title("MRI {}".format(idx), fontsize=20)

        plt.subplot(1, 3, 2)
        plt.imshow(msk[idx, :, :], cmap="bone", origin="lower")
        plt.title("Ground truth", fontsize=20)

        plt.subplot(1, 3, 3)

        print("Index {}: ".format(idx), end="")

        # Predict using the OpenVINO model
        # NOTE: OpenVINO expects channels first for input and output
        # So we transpose the input and output
        start_time = time.time()
        res = exec_net.infer({input_layer_name: np.transpose(img[[idx]], [0,3,1,2])})
        prediction = np.transpose(res[output_layer_name], [0,2,3,1])    
        print("Elapsed time = {:.4f} msecs, ".format(1000.0*(time.time()-start_time)), end="")
        
        plt.imshow(prediction[0,:,:,0], cmap="bone", origin="lower")
        dice_coef = calc_dice(msk[idx], prediction)
        plt.title("Prediction\nDice = {:.4f}".format(dice_coef), fontsize=20)

        print("Dice coefficient = {:.4f}, ".format(dice_coef), end="")
        
        save_name = os.path.join(png_directory, "prediction_openvino_{}.png".format(idx))
        print("Saved as: {}".format(save_name))
        plt.savefig(save_name)

if __name__ == "__main__":

    model_filename = os.path.join(args.output_path, args.inference_filename)

    ds_testing = DatasetGenerator(os.path.join(args.data_path, "testing/*.npz"), 
                              crop_dim=args.crop_dim, 
                              batch_size=128, 
                              augment=False, 
                              seed=args.seed)
    
    path_to_xml_file = "{}/FP32/{}.xml".format(args.output_path, args.inference_filename)
    path_to_bin_file = "{}/FP32/{}.bin".format(args.output_path, args.inference_filename)

    ie = IECore()
    net = ie.read_network(model=path_to_xml_file, weights=path_to_bin_file)

    input_layer_name = next(iter(net.input_info))
    output_layer_name = next(iter(net.outputs))
    print("Input layer name = {}\nOutput layer name = {}".format(input_layer_name, output_layer_name))

    exec_net = ie.load_network(network=net, device_name="CPU", num_requests=1)

    # Create output directory for images
    png_directory = args.output_pngs
    if not os.path.exists(png_directory):
        os.makedirs(png_directory)

    # Plot some results
    # The plots will be saved to the png_directory
    # Just picking some random samples.
    indicies_testing = [11,17,23,56,89,101,119]

    for idx in indicies_testing:
        plot_results(ds_testing, idx, png_directory, exec_net, input_layer_name, output_layer_name)
