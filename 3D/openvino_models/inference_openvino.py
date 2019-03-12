#!/usr/bin/env python
"""
 Copyright (c) 2019 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

"""
OpenVINO Python Inference Script
This will load the OpenVINO version of the model (IR)
and perform inference on a few validation samples
from the Decathlon dataset.

"""

import sys
import os
from argparse import ArgumentParser
import numpy as np
import logging as log
from time import time
from openvino.inference_engine import IENetwork, IEPlugin

import sys
sys.path.append("..") # Adds higher directory to python modules path.
from dataloader import DataGenerator

def dice_score(pred, truth):
    """
    Sorensen Dice score
    Measure of the overlap between the prediction and ground truth masks
    """
    numerator = np.sum(pred * truth) * 2.0 + 1.0
    denominator = np.sum(pred) + np.sum(truth) + 1.0

    return numerator / denominator

def load_data(args):
    """
    Modify this to load your data and labels
    """

    validation_data_params = {"dim": (args.patch_dim, args.patch_dim, args.patch_dim),
                              "batch_size": 1,
                              "n_in_channels": args.number_input_channels,
                              "n_out_channels": 1,
                              "train_test_split": args.train_test_split,
                              "augment": False,
                              "shuffle": False,
                              "seed": args.random_seed}
    validation_generator = DataGenerator(False, args.data_path,
                                         **validation_data_params)

    # for batch_idx in tqdm(range(validation_generator.num_batches),
    #                       desc="Predicting on batch"):

    batch_idx = 0
    imgs, msks = validation_generator.get_batch(batch_idx)
    fileIDs = validation_generator.get_batch_fileIDs(batch_idx)

    """
    OpenVINO uses channels first tensors (NCHWD).
    TensorFlow usually does channels last (NHWDC).
    So we need to transpose the axes.
    """
    imgs = imgs.transpose((0,4,1,2,3))
    msks = msks.transpose((0,4,1,2,3))

    return imgs, msks, fileIDs

def load_model(fp16=False):
    """
    Load the OpenVINO model.
    """
    log.info("Loading U-Net model to the plugin")

    if fp16:  # Floating point 16 is for Myriad X
        model_xml = "./FP16/3d_unet_decathlon.xml"
    else:     # FP32 for most devices
        model_xml = "./FP32/3d_unet_decathlon.xml"

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
        res = exec_net.infer(inputs={input_blob: input_data[0:batch_size,:n_channels]})
        infer_time.append((time() - t0) * 1000)

    average_inference = np.average(np.asarray(infer_time))
    log.info("Average running time of one batch: {:.5f} ms".format(average_inference))
    log.info("Images per second = {:.3f}".format(batch_size * 1000.0 / average_inference))

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
    parser.add_argument("-stats", "--stats", help="Plot the runtime statistics",
                        default=False, action="store_true")
    parser.add_argument("--patch_dim",
                        type=int,
                        default=144,
                        help="Size of the 3D patch")
    parser.add_argument("--train_test_split",
                        type=float,
                        default=0.85,
                        help="Train test split (0-1)")
    parser.add_argument("--number_input_channels",
                        type=int,
                        default=1,
                        help="Number of input channels")
    datapath = "../../../data/decathlon/Task01_BrainTumour/"
    parser.add_argument("--data_path",
                        default=datapath,
                        help="Root directory for Medical Decathlon dataset")
    parser.add_argument("--random_seed",
                        default=816,
                        help="Random seed")

    return parser

def main():

    log.basicConfig(format="[ %(levelname)s ] %(message)s",
                    level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()

    # Plugin initialization for specified device and
    #     load extensions library if specified
    plugin = IEPlugin(device=args.device, plugin_dirs=args.plugin_dir)
    if args.cpu_extension and "CPU" in args.device:
        plugin.add_cpu_extension(args.cpu_extension)

    # Read IR
    # If using MYRIAD then we need to load FP16 model version
    model_xml, model_bin = load_model(args.device == "MYRIAD")

    log.info("Loading network files:\n\t{}\n\t{}".format(model_xml, model_bin))
    net = IENetwork(model=model_xml, weights=model_bin)
    #net = IENetwork.from_ir(model=model_xml, weights=model_bin) # Old API

    """
    This code checks to see if all of the graphs in the IR are
    compatible with OpenVINO. If not, then you'll need to probably
    try to load in an extension library from ${INTEL_CVSDK_DIR}/inference_engine/lib
    """
    if "CPU" in plugin.device:
        supported_layers = plugin.get_supported_layers(net)
        not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
        if len(not_supported_layers) != 0:
            log.error("Following layers are not supported by the plugin "
                      " for specified device {}:\n {}".
                      format(plugin.device, ", ".join(not_supported_layers)))
            log.error("Please try to specify cpu extensions library path "
                      "in sample's command line parameters using -l "
                      "or --cpu_extension command line argument")
            log.error("On CPU this is usually -l ${INTEL_CVSDK_DIR}/inference_engine/lib/centos_7.4/intel64/libcpu_extension_avx2.so")
            log.error("You may need to build the OpenVINO samples directory for this library to be created on your system.")
            log.error("e.g. bash ${INTEL_CVSDK_DIR}/inference_engine/samples/build_samples.sh will trigger the library to be built.")
            log.error("Replace 'centos_7.4' with the pathname on your computer e.g. ('ubuntu_16.04')")
            sys.exit(1)

    assert len(net.inputs.keys()) == 1, "Sample supports only single input topologies"
    assert len(net.outputs) == 1, "Sample supports only single output topologies"

    """
    Ask OpenVINO for input and output tensor names and sizes
    """
    input_blob = next(iter(net.inputs))  # Name of the input layer
    out_blob = next(iter(net.outputs))   # Name of the output layer

    batch_size, n_channels, height, width, depth = net.inputs[input_blob].shape
    batch_size, n_out_channels, height_out, width_out, depth_out = net.outputs[out_blob].shape
    net.batch_size = batch_size

    log.info("Batch size = {}".format(batch_size))

    # Load data
    input_data, label_data, img_indicies = load_data(args)

    # Loading model to the plugin
    exec_net = plugin.load(network=net)
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
    predictions = np.zeros((input_data.shape[0], n_out_channels,
                            height_out, width_out, depth_out))

    for idx in range(0, input_data.shape[0], batch_size):
        log.info("{} started".format(idx))
        res = exec_net.infer(inputs={input_blob:
                                     input_data[idx:(idx+batch_size),
                                     :n_channels]})
        log.info("{} finished".format(idx))
        # Save the predictions to array
        predictions[idx:(idx+batch_size),] = res[out_blob]

    if idx != (len(img_indicies)-1):  # Partial batch left in data
        log.info("Partial batch left over in dataset.")

    """
    Evaluate model with Dice metric
    """
    for idx in range(img_indicies.shape[0]):
        dice = dice_score(predictions[idx,0,:,:,:], label_data[idx,0,:,:,:])
        log.info("Image #{}: Dice score = {:.4f}".format(img_indicies[idx], dice))

    if args.plot:
        plot_predictions(predictions, input_data, label_data, img_indicies, args)

    del exec_net
    del plugin

if __name__ == '__main__':
    sys.exit(main() or 0)
