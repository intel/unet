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
#os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Get rid of the AVX, SSE warnings

import numpy as np
import tensorflow as tf
import time
from tensorflow import keras as K
import settings
import argparse
from dataloader import DatasetGenerator, get_decathlon_filelist

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
parser.add_argument("--blocktime", type=int,
                    default=settings.BLOCKTIME,
                    help="blocktime")
parser.add_argument("--intraop_threads", default=settings.NUM_INTRA_THREADS,
                    type=int, help="Number of intra-op-parallelism threads")
parser.add_argument("--interop_threads", default=settings.NUM_INTER_THREADS,
                    type=int, help="Number of inter-op-parallelism threads")
parser.add_argument("--crop_dim", default=settings.CROP_DIM,
                    type=int, help="Crop dimension for images")
parser.add_argument("--seed", default=settings.SEED,
                    type=int, help="Random seed")
parser.add_argument("--split", type=float, default=settings.TRAIN_TEST_SPLIT,
                    help="Train/testing split for the data")
parser.add_argument("--BF16", help="auto mixed precision",
                    action="store_true")
parser.add_argument("--OMP", help="openMP thread settings",
                    action="store_true")   

args = parser.parse_args()


def test_oneDNN():
    import tensorflow as tf

    import os

    def get_mkl_enabled_flag():

        mkl_enabled = False
        major_version = int(tf.__version__.split(".")[0])
        minor_version = int(tf.__version__.split(".")[1])
        if major_version >= 2:
            if minor_version < 5:
                from tensorflow.python import _pywrap_util_port
            elif minor_version >= 9:

                from tensorflow.python.util import _pywrap_util_port
                onednn_enabled = int(os.environ.get('TF_ENABLE_ONEDNN_OPTS', '1'))

            else:
                from tensorflow.python.util import _pywrap_util_port
                onednn_enabled = int(os.environ.get('TF_ENABLE_ONEDNN_OPTS', '0'))
            mkl_enabled = _pywrap_util_port.IsMklEnabled() or (onednn_enabled == 1)
        else:
            mkl_enabled = tf.pywrap_tensorflow.IsMklEnabled()
        return mkl_enabled

    print ("We are using Tensorflow version", tf.__version__)
    print("oneDNN enabled :", get_mkl_enabled_flag())
test_oneDNN()


if args.OMP:
    # If hyperthreading is enabled, then use
    os.environ["KMP_AFFINITY"] = "granularity=thread,compact,1,0"

    # If hyperthreading is NOT enabled, then use
    #os.environ["KMP_AFFINITY"] = "granularity=thread,compact"

    os.environ["KMP_BLOCKTIME"] = str(args.blocktime)
    os.environ["OMP_NUM_THREADS"] = str(args.intraop_threads)
    os.environ["KMP_SETTINGS"] = "0"  # Show the settings at runtime

else:
    os.environ["INTRA_THREADS"] = str(args.intraop_threads)
    os.environ["INTER_THREADS"] = str(args.interop_threads)



def set_itex_amp(amp_target, device):
    # set configure for auto mixed precision.
    import intel_extension_for_tensorflow as itex
    print("intel_extension_for_tensorflow {}".format(itex.__version__))

    auto_mixed_precision_options = itex.AutoMixedPrecisionOptions()
    if amp_target=="BF16":
        auto_mixed_precision_options.data_type = itex.BFLOAT16
    else:
        auto_mixed_precision_options.data_type = itex.FLOAT16

    graph_options = itex.GraphOptions(auto_mixed_precision_options=auto_mixed_precision_options)
    # enable auto mixed precision.
    graph_options.auto_mixed_precision = itex.ON

    config = itex.ConfigProto(graph_options=graph_options)
    # set GPU backend.
    print(config)
    backend = device
    itex.set_backend(backend, config)

    print("Set itex for AMP (auto_mixed_precision, {}_FP32) with backend {}".format(amp_target, backend))

if args.BF16:
  print("set itex amp")
  set_itex_amp( amp_target="BF16", device="cpu" )




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
    Sorensen (Soft) Dice coefficient - Don't round predictions
    """
    numerator = 2.0 * np.sum(target * prediction) + smooth
    denominator = np.sum(target) + np.sum(prediction) + smooth
    coef = numerator / denominator

    return coef


def plot_results(ds, batch_num, png_directory):
    
    plt.figure(figsize=(10,10))

    img, msk = next(ds.ds)

    idx = np.argmax(np.sum(np.sum(msk[:,:,:,0], axis=1), axis=1)) # find the slice with the largest tumor

    plt.subplot(1, 3, 1)
    plt.imshow(img[idx, :, :, 0], cmap="bone", origin="lower")
    plt.title("MRI {}".format(idx), fontsize=20)

    plt.subplot(1, 3, 2)
    plt.imshow(msk[idx, :, :], cmap="bone", origin="lower")
    plt.title("Ground truth", fontsize=20)

    plt.subplot(1, 3, 3)

    print("Index {}: ".format(idx), end="")
    
    # Predict using the TensorFlow model
    start_time = time.time()
    prediction = model.predict(img[[idx]])
    print("Elapsed time = {:.4f} msecs, ".format(1000.0*(time.time()-start_time)), end="")
    
    plt.imshow(prediction[0,:,:,0], cmap="bone", origin="lower")
    dice_coef = calc_dice(msk[idx], prediction)
    print("Dice coefficient = {:.4f}, ".format(dice_coef), end="")
    plt.title("Prediction\nDice = {:.4f}".format(dice_coef), fontsize=20)

    save_name = os.path.join(png_directory, "prediction_tf_{}_{}.png".format(batch_num, idx))
    print("Saved as: {}".format(save_name))
    plt.savefig(save_name)
        
if __name__ == "__main__":

    model_filename_fp32 = os.path.join(args.output_path, "2d_unet_decathlon")
    model_filename_bf16 = os.path.join(args.output_path, "2d_unet_decathlon_bf16")

    if(os.path.exists(model_filename_fp32)):
        model_filename= model_filename_fp32
    elif(os.path.exists(model_filename_bf16)):
        model_filename= model_filename_bf16
    else:
        print("Please train the model first: exiting")
        exit()



    trainFiles, validateFiles, testFiles = get_decathlon_filelist(data_path=args.data_path, seed=args.seed, split=args.split)

    ds_test = DatasetGenerator(testFiles, batch_size=128, crop_dim=[args.crop_dim,args.crop_dim], augment=False, seed=args.seed)

    # Load model
    if args.use_pconv:
        from model_pconv import unet
        unet_model = unet(use_pconv=True)
    else:
        from model import unet
        unet_model = unet()
        
    
    model = unet_model.load_model(model_filename)

    # Create output directory for images
    png_directory = args.output_pngs
    if not os.path.exists(png_directory):
        os.makedirs(png_directory)

    for batchnum in range(10):
        plot_results(ds_test, batchnum, png_directory)
