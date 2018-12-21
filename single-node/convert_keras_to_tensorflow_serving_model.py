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
from tensorflow.contrib.session_bundle import exporter
import keras
import tensorflow as tf

import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_filename",
                    help="Name of saved Keras model (e.g. model.h5)",
                    default=os.path.join(".", "unet2d_keras_model_upsampling_for_training.hdf5"))
parser.add_argument("--output_directory",
                    help="Directory where to save the TensorFlow Serving Protobuf Model",
                    default="saved_2dunet_model_protobuf")

args = parser.parse_args()


def dice_coef(y_true, y_pred, smooth=1.0):
    intersection = tf.reduce_sum(y_true * y_pred, axis=(1, 2, 3))
    union = tf.reduce_sum(y_true + y_pred, axis=(1, 2, 3))
    numerator = tf.constant(2.) * intersection + smooth
    denominator = union + smooth
    coef = numerator / denominator
    return tf.reduce_mean(coef)


def dice_coef_loss(y_true, y_pred, smooth=1.0):

    y_true_f = keras.backend.flatten(y_true)
    y_pred_f = keras.backend.flatten(y_pred)
    intersection = keras.backend.sum(y_true_f * y_pred_f)
    loss = -keras.backend.log(2.0 * intersection + smooth) + \
        keras.backend.log((keras.backend.sum(y_true_f) +
                           keras.backend.sum(y_pred_f) + smooth))

    return loss


sess = keras.backend.get_session()

print("Loading saved Keras model.")

"""
If there are other custom loss and metric functions you'll need to specify them
and add them to the dictionary below.
"""
model = keras.models.load_model(args.input_filename, custom_objects={
                                "dice_coef": dice_coef, "dice_coef_loss": dice_coef_loss})


print("Freezing the graph.")
keras.backend.set_learning_phase(0)

signature = tf.saved_model.signature_def_utils.predict_signature_def(
    inputs={'input': model.input}, outputs={'output': model.output})

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
