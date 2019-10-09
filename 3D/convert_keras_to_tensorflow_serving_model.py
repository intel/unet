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

import shutil
from tensorflow.contrib.session_bundle import exporter
import keras as K
import tensorflow as tf
from model import unet
import os
import argparse
from argparser import args
if args.keras_api:
    import keras as K
else:
    from tensorflow import keras as K

parser = argparse.ArgumentParser()
parser.add_argument("--input_filename",
                    help="Name of saved Keras model (e.g. model.h5)",
                    default=os.path.join("saved_model", "3d_unet_decathlon.hdf5"))
parser.add_argument("--output_directory",
                    help="Directory where to save the TensorFlow Serving Protobuf Model",
                    default="saved_3dunet_model_protobuf")

args = parser.parse_args()

sess = K.backend.get_session()

print("Loading saved Keras model.")

"""
If there are other custom loss and metric functions you'll need to specify them
and add them to the dictionary below.
"""
unet_model = unet(channels_last = True)  # channels first or last
model = K.models.load_model(args.input_filename, custom_objects=unet_model.custom_objects)

print("Freezing the graph.")
K.backend.set_learning_phase(0)

signature = tf.saved_model.signature_def_utils.predict_signature_def(
    inputs={'input': model.input}, outputs={'output': model.output})

shutil.rmtree(args.output_directory, ignore_errors=True)

print("Saving the model to directory {}".format(args.output_directory))

builder = tf.saved_model.builder.SavedModelBuilder(args.output_directory)
builder.add_meta_graph_and_variables(
    sess=sess,
    tags=[tf.saved_model.tag_constants.SERVING],
    signature_def_map={
        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
            signature
    })
builder.save()
print("TensorFlow protobuf version of model is saved.")
