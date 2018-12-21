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

import os
import argparse

import tensorflow as tf

#import keras as K
from tensorflow import keras as K

parser = argparse.ArgumentParser(
    description="Converts a Keras model to frozen TensorFlow protobuf model.",
    add_help=True, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--input_model", "-m",
                    default=os.path.join("output",
                    "unet_model_upsampling_for_inference.hdf5"),
                    type=str, help="Path to Keras model to be converted.")
parser.add_argument("--output_directory",
                    type=str,
                    default="frozen_tensorflow_model",
                    help="Output directory for frozen TF model.")

argv = parser.parse_args()

def export_keras_to_tf(input_model, output_dir):
    """
    Load the Keras model. Use the TF graph conversion utilities
    to freeze the graph and save it as a protobuf file.
    """

    print("Loading Keras model: ", input_model)

    keras_model = K.models.load_model(input_model)

    """
    Tell Keras that we want to remove the training ops and
    just do inference.
    """
    K.backend.set_learning_phase(0)
    K.backend.set_image_data_format("channels_last")
    print("Setting to channels last data format (NHWC).")

    input_node_names = []
    for idx in range(len(keras_model.inputs)):
        input_node_names.append(keras_model.inputs[idx].name)

    predictions = []
    prediction_node_names = []

    for idx in range(len(keras_model.outputs)):
        prediction_node_names.append("output_node" + str(idx))
        predictions.append(tf.identity(keras_model.outputs[idx],
                         name=prediction_node_names[idx]))

    sess = K.backend.get_session()

    constant_graph = tf.compat.v1.graph_util.convert_variables_to_constants(sess,
                     sess.graph.as_graph_def(), prediction_node_names)
    infer_graph = tf.compat.v1.graph_util.remove_training_nodes(constant_graph)

    out_filename = os.path.splitext(os.path.basename(input_model))[0] + ".pb"

    tf.io.write_graph(infer_graph, output_dir, out_filename, as_text=True)

    print("Saved as TF frozen model to: ",
          os.path.join(output_dir, out_filename))

    return prediction_node_names, input_node_names

def main():

    input_model = argv.input_model

    prediction_node_names, input_node_names = export_keras_to_tf(input_model,
                argv.output_directory)

    print("Input nodes are :", input_node_names)
    print("Output nodes are:", prediction_node_names)


if __name__ == "__main__":
  main()
