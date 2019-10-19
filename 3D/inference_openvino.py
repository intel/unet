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

import sys
import os
import csv

import numpy as np
import logging as log
from time import time
from openvino.inference_engine import IENetwork, IECore

import tensorflow as tf
import keras as K

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import nibabel as nib

from tqdm import tqdm

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Get rid of the AVX, SSE warnings

"""
OpenVINO Python Inference Script
This will load the OpenVINO version of the model (IR)
and perform inference on a few validation samples
from the Decathlon dataset.

"""

def dice_score(pred, truth):
    """
    Sorensen Dice score
    Measure of the overlap between the prediction and ground truth masks
    """
    numerator = np.sum(np.round(pred) * truth) * 2.0
    denominator = np.sum(np.round(pred)) + np.sum(truth)

    return numerator / denominator

def crop_img(img, msk, crop_dim, n_channels, n_out_channels):
    """
    Crop the image and mask
    """

    number_of_dimensions = len(crop_dim)

    slices = []

    for idx in range(number_of_dimensions):  # Go through each dimension

        cropLen = crop_dim[idx]
        imgLen = img.shape[idx]

        start = (imgLen-cropLen)//2

        slices.append(slice(start, start+cropLen))

    # No slicing along channels
    slices_img = slices.copy()
    slices_msk = slices.copy()

    slices_img.append(slice(0, n_channels))
    slices_msk.append(slice(0, n_out_channels))

    return img[tuple(slices_img)], msk[tuple(slices_msk)]

def z_normalize_img(img):
    """
    Normalize the image so that the mean value for each image
    is 0 and the standard deviation is 1.
    """
    for channel in range(img.shape[-1]):

        img_temp = img[..., channel]
        img_temp = (img_temp - np.mean(img_temp)) / np.std(img_temp)

        img[..., channel] = img_temp

    return img

def load_data(imgFile, mskFile, crop_dim, n_channels, n_out_channels, openVINO_order=True):
    """
    Modify this to load your data and labels
    """

    imgs = np.empty((len(imgFile),*crop_dim,n_channels))
    msks = np.empty((len(mskFile),*crop_dim,n_out_channels))
    fileIDs = []

    for idx in range(len(imgFile)):

        img_temp = np.array(nib.load(imgFile[idx]).dataobj)
        msk = np.array(nib.load(mskFile[idx]).dataobj)

        if n_channels == 1:
            img = img_temp[:, :, :, [0]]  # FLAIR channel
        else:
            img = img_temp

        # Add channels to mask
        msk[msk > 0] = 1.0
        msk = np.expand_dims(msk, -1)


        # Crop the image to the input size
        img, msk = crop_img(img, msk, crop_dim, n_channels, n_out_channels)

        # z-normalize the pixel values
        img = z_normalize_img(img)

        fileIDs.append(os.path.basename(imgFile[idx]))

        imgs[idx] = img
        msks[idx] = msk

    if openVINO_order:
        imgs = imgs.transpose((0, 4, 1, 2, 3))
        msks = msks.transpose((0, 4, 1, 2, 3))

    return imgs, msks, fileIDs


def load_model(model_xml, fp16=False):
    """
    Load the OpenVINO model.
    """
    log.info("Loading U-Net model to the plugin")

    model_bin = os.path.splitext(model_xml)[0] + ".bin"

    return model_xml, model_bin


def print_stats(exec_net, input_data, n_channels, batch_size, input_blob, out_blob, args):
    """
    Prints layer by layer inference times.
    Good for profiling which ops are most costly in your model.
    """

    # Start sync inference
    log.info("Starting inference ({} iterations)".format(args.number_iter))
    log.info("Number of input channels = {}".format(n_channels))
    log.info("Input data shape = {}".format(input_data.shape))
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

    parser = ArgumentParser(description="Performs inference using OpenVINO. Compares to Keras model.",
                            formatter_class=ArgumentDefaultsHelpFormatter)
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
    parser.add_argument("-stats", "--stats", help="Plot the runtime statistics",
                        default=False, action="store_true")
    parser.add_argument("-plot", "--plot", help="Plot the predictions",
                        default=False, action="store_true")
    parser.add_argument("--csv_file",
                        default="test.csv",
                        help="CSV list of files to test")
    parser.add_argument("--openvino_model", type=str, help="The saved OpenVINO XML file",
                        default="./openvino_models/FP32/3d_unet_decathlon.xml")
    parser.add_argument("--keras_model", type=str, help="Keras model filename",
                        default="./saved_model/3d_unet_decathlon.hdf5")
    return parser

def read_csv_file(filename):
    """
    Read the CSV file with the image and mask filenames
    """
    imgFiles = []
    mskFiles = []
    with open(filename, "rt") as f:
        data = csv.reader(f)
        for row in data:
            if len(row) > 0:
                imgFiles.append(row[0])
                mskFiles.append(row[1])

    return imgFiles, mskFiles, len(imgFiles)


def main():

    log.basicConfig(format="[ %(levelname)s ] %(message)s",
                    level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()

    log.info(args)

    log.info("Loading test data from file: {}".format(args.csv_file))

    ie = IECore()
    if args.cpu_extension and "CPU" in args.device:
        ie.add_extension(args.cpu_extension, "CPU")


    # Read IR
    model_xml, model_bin = load_model(args.openvino_model, args.device=="MYRIAD")
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
    batch_size, n_channels, height, width, depth = net.inputs[input_blob].shape
    batch_size, n_out_channels, height_out, width_out, depth_out = net.outputs[out_blob].shape
    crop_dim = [height, width, depth]
    """
    Read the CSV file with the filenames of the images and masks
    """
    imgFiles, mskFiles, num_imgs = read_csv_file(args.csv_file)


    """
    Load the data for OpenVINO
    """
    input_data, label_data_ov, img_indicies = load_data(imgFiles, mskFiles,
                crop_dim, n_channels, n_out_channels, openVINO_order=True)


    # Reshape the OpenVINO network to accept the different image input shape
    # NOTE: This only works for some models (e.g. fully convolutional)
    batch_size = 1
    n_channels = input_data.shape[1]
    height = input_data.shape[2]
    width = input_data.shape[3]
    depth = input_data.shape[4]

    net.reshape({input_blob:(batch_size,n_channels,height,width,depth)})
    batch_size, n_channels, height, width, depth = net.inputs[input_blob].shape
    batch_size, n_out_channels, height_out, width_out, depth_out = net.outputs[out_blob].shape

    log.info("The network inputs are:")
    for idx, input_layer in enumerate(net.inputs.keys()):
        log.info("{}: {}, shape = {} [N,C,H,W,D]".format(idx,input_layer,net.inputs[input_layer].shape))

    log.info("The network outputs are:")
    for idx, output_layer in enumerate(net.outputs.keys()):
        log.info("{}: {}, shape = {} [N,C,H,W,D]".format(idx,output_layer,net.outputs[output_layer].shape))

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
    predictions_ov = np.zeros((num_imgs, n_out_channels,
                            depth_out, height_out, width_out))

    log.info("Starting OpenVINO inference")
    ov_times = []
    for idx in tqdm(range(0, num_imgs)):

        start_time = time()

        res = exec_net.infer(inputs={input_blob: input_data[[idx],:n_channels]})

        ov_times.append(time() - start_time)

        predictions_ov[idx, ] = res[out_blob]

        #print("{}, {}".format(imgFiles[idx], dice_score(res[out_blob],label_data_ov[idx])))


    log.info("Finished OpenVINO inference")

    del exec_net

    """
    Load the data for Keras
    """
    input_data, label_data_keras, img_indicies = load_data(imgFiles, mskFiles,
                        crop_dim, n_channels, n_out_channels,
                        openVINO_order=False)

    # Load OpenVINO model for inference
    model = K.models.load_model(args.keras_model, compile=False)

    # Inference only Keras
    K.backend._LEARNING_PHASE = tf.constant(0)
    K.backend.set_learning_phase(False)
    K.backend.set_learning_phase(0)
    K.backend.set_image_data_format("channels_last")

    predictions_keras = np.zeros((num_imgs,
                            height_out, width_out, depth_out, n_out_channels))

    log.info("Starting Keras inference")
    keras_times = []
    for idx in tqdm(range(num_imgs)):

        start_time = time()
        res = model.predict(input_data[[idx],...,:n_channels])

        keras_times.append(time() - start_time)

        #print("{}, {}".format(imgFiles[idx], dice_score(res,label_data_keras[idx])))

        predictions_keras[idx] = res

    log.info("Finished Keras inference")

    save_directory = "predictions_openvino"
    try:
        os.stat(save_directory)
    except:
        os.mkdir(save_directory)

    """
    Evaluate model with Dice metric
    """
    out_channel = 0
    for idx in tqdm(range(num_imgs)):

        filename = os.path.splitext(os.path.splitext(img_indicies[idx])[0])[0]
        img = input_data[idx,...,:n_channels]
        ground_truth = label_data_keras[idx, :, :, :, out_channel]

        # Transpose the OpenVINO prediction back to NCHWD (to be consistent with Keras)
        pred_ov = np.transpose(predictions_ov, [0,2,3,4,1])[idx, :, :, :, out_channel]
        pred_keras = predictions_keras[idx, :, :, :, out_channel]

        dice_ov = dice_score(pred_ov, ground_truth)
        dice_keras = dice_score(pred_keras, ground_truth)


        img_nib = nib.Nifti1Image(img, np.eye(4))
        img_nib.to_filename(os.path.join(save_directory,
                                      "{}_img.nii.gz".format(filename)))

        msk_nib = nib.Nifti1Image(ground_truth, np.eye(4))
        msk_nib.to_filename(os.path.join(save_directory,
                                      "{}_msk.nii.gz".format(filename)))

        pred_ov_nib = nib.Nifti1Image(pred_ov, np.eye(4))
        pred_ov_nib.to_filename(os.path.join(save_directory,
                                       "{}_pred_ov.nii.gz".format(filename)))

        log.info("Image file {}: OpenVINO Dice score = {:f}, "
            "Keras/TF Dice score = {:f}, Maximum absolute pixel difference OV versus Keras/TF = {:.2e}".format(
            img_indicies[idx], dice_ov, dice_keras, np.mean(np.abs(pred_ov - pred_keras))))

    log.info("Average inference time: \n"
             "OpenVINO = {} seconds (s.d. {})\n "
             "Keras/TF = {} seconds (s.d. {})\n".format(np.mean(ov_times),
             np.std(ov_times),
             np.mean(keras_times),
             np.std(keras_times)))
    log.info("Raw OpenVINO inference times = {} seconds".format(ov_times))
    log.info("Raw Keras inference times = {} seconds".format(keras_times))


if __name__ == '__main__':
    sys.exit(main() or 0)
