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

import os
import sys
import argparse
import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io
from keras import backend as K
from keras.models import load_model
import shutil

def setKerasOptions():
    K._LEARNING_PHASE = tf.constant(0)
    K.set_learning_phase(False)
    K.set_learning_phase(0)
    K.set_image_data_format("channels_last")

def getInputParameters():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_model", "-m", required=True,
                        type=str, help="Path to Keras model.")

    return parser


def export_keras_to_tf(input_model, output_model):
    print("Loading Keras model: ", input_model)

    keras_model = load_model(input_model, compile=False)

    print(keras_model.summary())

    num_output = len(keras_model.outputs)
    predictions = [None] * num_output
    prediction_node_names = [None] * num_output

    for i in range(num_output):
        prediction_node_names[i] = "output_node" + str(i)
        predictions[i] = tf.identity(keras_model.outputs[i],
                                     name=prediction_node_names[i])

    sess = K.get_session()

    constant_graph = graph_util.convert_variables_to_constants(sess,
                                                               sess.graph.as_graph_def(), prediction_node_names)
    infer_graph = graph_util.remove_training_nodes(constant_graph)

    # Write protobuf of frozen model
    frozen_dir = "./tf_protobuf/"
    shutil.rmtree(frozen_dir, ignore_errors=True) # Remove existing directory
    graph_io.write_graph(infer_graph, frozen_dir, output_model, as_text=False)

    print("Input shape {}".format(keras_model.inputs[0].shape))

    pb_filename = os.path.join(frozen_dir, output_model)
    print("Frozen TensorFlow model written to: {}".format(pb_filename))
    print("Convert this to OpenVINO by running:\n")
    print("source /opt/intel/openvino/bin/setupvars.sh")
    print("python $INTEL_OPENVINO_DIR/deployment_tools/model_optimizer/mo_tf.py \\")
    print("       --input_model {} \\".format(pb_filename))

    shape_string = "[1"
    for idx in range(len(keras_model.inputs[0].shape[1:])):
        shape_string += ",{}".format(keras_model.inputs[0].shape[idx+1])
    shape_string += "]"

    print("       --input_shape {} \\".format(shape_string))
    print("       --output_dir openvino_models/FP32/ \\")
    print("       --data_type FP32")

    return prediction_node_names

def main():
    argv = getInputParameters().parse_args()

    input_model = argv.input_model

    # Change filename to protobuf extension
    base = os.path.basename(input_model)
    output_model = os.path.splitext(base)[0] + ".pb"

    prediction_node_names = export_keras_to_tf(input_model, output_model)

    print("Ouput nodes are:", prediction_node_names)
    print("Saved as TF frozen model to: ", output_model)


if __name__ == "__main__":
    main()
