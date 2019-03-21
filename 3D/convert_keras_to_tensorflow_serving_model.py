#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Intel Corporation
#
# This program is free software: you can redistribute it and/or modify  
# it under the terms of the GNU General Public License as published by  
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but 
# WITHOUT ANY WARRANTY; without even the implied warranty of 
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU 
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License 
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#

import shutil
from tensorflow.contrib.session_bundle import exporter
import keras
import tensorflow as tf

import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_filename",
                    help="Name of saved Keras model (e.g. model.h5)",
                    default=os.path.join("saved_model", "3d_unet_decathlon.hdf5"))
parser.add_argument("--output_directory",
                    help="Directory where to save the TensorFlow Serving Protobuf Model",
                    default="saved_3dunet_model_protobuf")

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


def sensitivity(target, prediction, axis=(1, 2, 3), smooth=1e-5):

    intersection = tf.reduce_sum(prediction * target, axis=axis)
    coef = (intersection + smooth) / \
        (tf.reduce_sum(prediction, axis=axis) + smooth)
    return tf.reduce_mean(coef)


def specificity(target, prediction, axis=(1, 2, 3), smooth=1e-5):

    intersection = tf.reduce_sum(prediction * target, axis=axis)
    coef = (intersection + smooth) / \
        (tf.reduce_sum(prediction, axis=axis) + smooth)
    return tf.reduce_mean(coef)


sess = keras.backend.get_session()

print("Loading saved Keras model.")

"""
If there are other custom loss and metric functions you'll need to specify them
and add them to the dictionary below.
"""
model = keras.models.load_model(args.input_filename, custom_objects={
                                "sensitivity": sensitivity, "specificity": specificity,
                                "dice_coef": dice_coef, "dice_coef_loss": dice_coef_loss})


print("Freezing the graph.")
keras.backend.set_learning_phase(0)

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
