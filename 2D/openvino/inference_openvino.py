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

from openvino.inference_engine import IENetwork, IECore
from time import time
import logging as log
import numpy as np
from argparse import ArgumentParser
import os
import sys

"""
OpenVINO Python Inference Script
This will load the OpenVINO version of the model (IR)
and perform inference on a few validation samples
from the Decathlon dataset.

You'll need the extension library to handle the Resize_Bilinear operations.

python inference_openvino.py -l ${INTEL_OPENVINO_DIR}/inference_engine/lib/intel64/libcpu_extension_avx2.so

"""

def dice_score(y_true, y_pred, smooth=1.):
    """
    Sorensen Dice coefficient
    """
    y_true = np.round(y_true)
    y_pred = np.round(y_pred)
    numerator = 2.0 * np.sum(y_true * y_pred) + smooth
    denominator = np.sum(y_true) + np.sum(y_pred) + smooth
    coef = numerator / denominator

    return coef

def plot_predictions(predictions, input_data, label_data, img_indicies, args):
    """
    Plot the predictions with matplotlib and save to png files
    """
    png_directory = "inference_examples_openvino"
    if not os.path.exists(png_directory):
        os.makedirs(png_directory)

    import matplotlib.pyplot as plt

    # Processing output blob
    log.info("Plotting the predictions and saving to png files. Please wait...")
    number_imgs = predictions.shape[0]
    num_rows_per_image = args.rows_per_image
    row = 0

    for idx in range(number_imgs):

        if row == 0:
            plt.figure(figsize=(15, 15))

        plt.subplot(num_rows_per_image, 3, 1+row*3)
        plt.imshow(input_data[idx, 0, :, :], cmap="bone", origin="lower")
        plt.axis("off")
        if row == 0:
            plt.title("MRI")

        plt.subplot(num_rows_per_image, 3, 2+row*3)
        plt.imshow(label_data[idx, 0, :, :], origin="lower")
        plt.axis("off")
        if row == 0:
            plt.title("Ground truth")

        plt.subplot(num_rows_per_image, 3, 3+row*3)
        plt.imshow(predictions[idx, 0, :, :], origin="lower")
        plt.axis("off")
        if row == 0:
            plt.title("Prediction")

        plt.tight_layout()

        if (row == (num_rows_per_image-1)) or (idx == (number_imgs-1)):

            if num_rows_per_image == 1:
                fileidx = "pred{}.png".format(img_indicies[idx])
            else:
                fileidx = "pred_group{}".format(idx // num_rows_per_image)
            filename = os.path.join(png_directory, fileidx)
            plt.savefig(filename,
                        bbox_inches="tight", pad_inches=0)
            print("Saved file: {}".format(filename))
            row = 0
        else:
            row += 1


def load_data():
    """
    Modify this to load your data and labels
    """

    # Load data
    # You can create this Numpy datafile by running the create_validation_sample.py script
    data_file = np.load("data/validation_data.npz")
    imgs_validation = data_file["imgs_validation"]
    msks_validation = data_file["msks_validation"]
    img_indicies = data_file["indicies_validation"]

    """
    OpenVINO uses channels first tensors (NCHW).
    TensorFlow usually does channels last (NHWC).
    So we need to transpose the axes.
    """
    input_data = imgs_validation.transpose((0, 3, 1, 2))
    msks_data = msks_validation.transpose((0, 3, 1, 2))

    return input_data, msks_data, img_indicies


def load_model(model, fp16=False):
    """
    Load the OpenVINO model.
    """
    log.info("Loading U-Net model to the plugin")

    model_xml = model
    model_bin = os.path.splitext(model_xml)[0] + ".bin"

    return model_xml, model_bin


def print_stats(exec_net, input_data, n_channels, batch_size, input_blob, out_blob, args):
    """
    Prints layer by layer inference times.
    Good for profiling which ops are most costly in your model.
    """

    # Start sync inference
    log.info("Starting inference ({} iterations)".format(args.number_iter))
    infer_time = []

    for i in range(args.number_iter):
        t0 = time()
        res = exec_net.infer(
            inputs={input_blob: input_data[0:batch_size, :n_channels]})
        infer_time.append((time() - t0) * 1000)

    average_inference = np.average(np.asarray(infer_time))
    log.info("Average running time of one batch: {:.5f} ms".format(
        average_inference))
    log.info("Images per second = {:.3f}".format(
        batch_size * 1000.0 / average_inference))

    perf_counts = exec_net.requests[0].get_perf_counts()
    log.info("Performance counters:")
    log.info("{:<70} {:<15} {:<15} {:<15} {:<10}".format("name",
                                                         "layer_type",
                                                         "exec_type",
                                                         "status",
                                                         "real_time, us"))
    for layer, stats in perf_counts.items():
        log.info("{:<70} {:<15} {:<15} {:<15} {:<10}".format(layer,
                                                             stats["layer_type"],
                                                             stats["exec_type"],
                                                             stats["status"],
                                                             stats["real_time"]))


def build_argparser():

    parser = ArgumentParser()
    parser.add_argument("-number_iter", "--number_iter",
                        help="Number of iterations", default=5, type=int)
    parser.add_argument("-l", "--cpu_extension",
                        help="MKLDNN (CPU)-targeted custom layers. "
                             "Absolute path to a shared library with "
                             "the kernels impl.", type=str)
    parser.add_argument("-pp", "--plugin_dir", help="Path to a plugin folder",
                        type=str, default=None)
    parser.add_argument("-d", "--device",
                        help="Specify the target device to infer on; "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)", default="CPU",
                        type=str)
    parser.add_argument("-plot", "--plot", help="Plot results",
                        default=False, action="store_true")
    parser.add_argument("-rows_per_image", "--rows_per_image",
                        help="Number of rows per plot (when -plot = True)",
                        default=4, type=int)
    parser.add_argument("-stats", "--stats", help="Plot the runtime statistics",
                        default=False, action="store_true")
    parser.add_argument("-m", "--model", help="OpenVINO model filename",
                        default="../openvino_models/FP32/saved_model.xml")
    return parser


def main():

    log.basicConfig(format="[ %(levelname)s ] %(message)s",
                    level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()

    ie = IECore()
    if args.cpu_extension and "CPU" in args.device:
        ie.add_extension(args.cpu_extension, "CPU")


    # Read IR
    model_xml, model_bin = load_model(args.model, args.device=="MYRIAD")
    log.info("Loading network files:\n\t{}\n\t{}".format(model_xml, model_bin))
    net = IENetwork(model=model_xml, weights=model_bin)

    if "CPU" in args.device:
        supported_layers = ie.query_network(net, "CPU")
        not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
        if len(not_supported_layers) != 0:
            log.error("Following layers are not supported by the plugin for specified device {}:\n {}".
                      format(args.device, ', '.join(not_supported_layers)))
            log.error("Please try to specify cpu extensions library path in sample's command line parameters using -l "
                      "or --cpu_extension command line argument")
            sys.exit(1)

    """
    Ask OpenVINO for input and output tensor names and sizes
    """
    input_blob = next(iter(net.inputs))  # Name of the input layer
    out_blob = next(iter(net.outputs))   # Name of the output layer

    # Load data
    input_data, label_data, img_indicies = load_data()

    batch_size = 1
    n_channels = input_data.shape[1]
    height = input_data.shape[2]
    width = input_data.shape[3]

    # Reshape the OpenVINO network to accept the different image input shape
    # NOTE: This only works for some models (e.g. fully convolutional)
    net.reshape({input_blob:(batch_size,n_channels,height,width)})
    batch_size, n_channels, height, width = net.inputs[input_blob].shape
    batch_size, n_out_channels, height_out, width_out = net.outputs[out_blob].shape

    # Loading model to the plugin
    log.info("Loading model to the plugin")
    exec_net = ie.load_network(network=net, device_name=args.device)
    del net

    if args.stats:
        # Print the latency and throughput for inference
        print_stats(exec_net, input_data, n_channels,
                    batch_size, input_blob, out_blob, args)

    """
    OpenVINO inference code
    input_blob is the name (string) of the input tensor in the graph
    out_blob is the name (string) of the output tensor in the graph
    Essentially, this looks exactly like a feed_dict for TensorFlow inference
    """
    # Go through the sample validation dataset to plot predictions
    predictions = np.zeros((img_indicies.shape[0], n_out_channels,
                            height_out, width_out))

    for idx in range(0, img_indicies.shape[0], batch_size):

        res = exec_net.infer(inputs={input_blob:
                                     input_data[idx:(idx+batch_size),
                                                :n_channels]})

        # Save the predictions to array
        predictions[idx:(idx+batch_size), ] = res[out_blob]

    if idx != (len(img_indicies)-1):  # Partial batch left in data
        log.info("Partial batch left over in dataset.")

    """
    Evaluate model with Dice metric
    """
    for idx in range(img_indicies.shape[0]):
        dice = dice_score(predictions[idx, 0, :, :], label_data[idx, 0, :, :])
        log.info("Image #{}: Dice score = {:.4f}".format(
            img_indicies[idx], dice))

    if args.plot:
        plot_predictions(predictions, input_data,
                         label_data, img_indicies, args)

    del exec_net


if __name__ == '__main__':
    sys.exit(main() or 0)
