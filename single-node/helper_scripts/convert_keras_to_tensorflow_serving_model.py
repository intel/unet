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
from tensorflow.contrib.session_bundle import exporter
import keras
import tensorflow as tf

import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_filename",
                    help="Name of saved Keras model (e.g. model.h5)",
                    default="output/unet_model_for_decathlon.hdf5")
parser.add_argument("--output_directory",
                    help="Directory where to save the TensorFlow Serving Protobuf Model",
                    default="saved_2dunet_model_protobuf")

args = parser.parse_args()

def dice_coef(y_true, y_pred, axis=(1, 2), smooth=1.):
    """
    Sorenson (Soft) Dice
    2 * |TP| / |T|*|P|
    where T is ground truth mask and P is the prediction mask
    """
    intersection = tf.reduce_sum(y_true * y_pred, axis=axis)
    union = tf.reduce_sum(y_true + y_pred, axis=axis)
    numerator = tf.constant(2.) * intersection + smooth
    denominator = union + smooth
    coef = numerator / denominator

    return tf.reduce_mean(coef)


def dice_coef_loss(target, prediction, axis=(1, 2), smooth=1.):
    """
    Sorenson (Soft) Dice loss
    Using -log(Dice) as the loss since it is better behaved.
    Also, the log allows avoidance of the division which
    can help prevent underflow when the numbers are very small.
    """
    intersection = tf.reduce_sum(prediction * target, axis=axis)
    p = tf.reduce_sum(prediction, axis=axis)
    t = tf.reduce_sum(target, axis=axis)
    numerator = tf.reduce_mean(intersection + smooth)
    denominator = tf.reduce_mean(t + p + smooth)
    dice_loss = -tf.log(2.*numerator) + tf.log(denominator)

    return dice_loss


def combined_dice_ce_loss(y_true, y_pred, axis=(1, 2), smooth=1., weight=.9):
    """
    Combined Dice and Binary Cross Entropy Loss
    """
    return weight*dice_coef_loss(y_true, y_pred, axis, smooth) + \
        (1-weight)*keras.losses.binary_crossentropy(y_true, y_pred)

sess = keras.backend.get_session()

print("Loading saved Keras model.")

"""
If there are other custom loss and metric functions you'll need to specify them
and add them to the dictionary below.
"""
model = keras.models.load_model(args.input_filename, custom_objects={
                                "dice_coef": dice_coef,
                                "combined_dice_ce_loss": combined_dice_ce_loss,
                                "dice_coef_loss": dice_coef_loss})


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

print("Model input name = ", model.input.op.name)
print("Model input shape = ", model.input.shape)
print("Model output name = ", model.output.op.name)
print("Model output shape = ", model.output.shape)

from distutils.sysconfig import get_python_lib
packages_directory=get_python_lib()

print()
print()
print("="*30)
print("To freeze this model, run the commands:")
print("mkdir frozen_model")
print("python {}/tensorflow/python/tools/freeze_graph.py "
       "--input_saved_model_dir {} "
       "--output_node_names {} "
       "--output_graph frozen_model/saved_model_frozen.pb".format(packages_directory,
       args.output_directory, model.output.op.name))
