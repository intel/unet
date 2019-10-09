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
                    help="Directory where to save the TensorFlow Checkpoint",
                    default="saved_3dunet_model_checkpoint")

args = parser.parse_args()

sess = K.backend.get_session()

print("Loading saved Keras model.")


"""
If there are other custom loss and metric functions you'll need to specify them
and add them to the dictionary below.
"""
unet_model = unet(channels_last = True)  # channels first or last
model = K.models.load_model(args.input_filename, custom_objects=unet_model.custom_objects)


print("Saving the model to directory {}".format(args.output_directory))

saver = tf.train.Saver()

# Create directory if it doesn't exist
try:
    os.stat(args.output_directory)
except:
    os.mkdir(args.output_directory)

save_path = saver.save(sess, os.path.join(
    args.output_directory, "unet_model.ckpt"))
print("Checkpoint saved in path: {}".format(save_path))
