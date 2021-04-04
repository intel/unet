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

"""
This module contains all of the model definition code.
You can try custom models by modifying the code here.
"""

from argparser import args
import os
import time
import shutil
import settings

import tensorflow as tf  # conda install -c anaconda tensorflow

from tensorflow import keras as K

from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io

from libs.pconv_layer import PConv2D


class unet(object):
    """
    2D U-Net model class
    """

    def __init__(self, channels_first=settings.CHANNELS_FIRST,
                 fms=settings.FEATURE_MAPS,
                 output_path=settings.OUT_PATH,
                 inference_filename=settings.INFERENCE_FILENAME,
                 blocktime=settings.BLOCKTIME,
                 num_threads=settings.NUM_INTRA_THREADS,
                 learning_rate=settings.LEARNING_RATE,
                 weight_dice_loss=settings.WEIGHT_DICE_LOSS,
                 num_inter_threads=settings.NUM_INTRA_THREADS,
                 use_upsampling=settings.USE_UPSAMPLING,
                 use_dropout=settings.USE_DROPOUT,
                 print_model=settings.PRINT_MODEL,
                 use_pconv=False):

        self.channels_first = channels_first
        if self.channels_first:
            """
            Use NCHW format for data
            """
            self.concat_axis = 1
            self.data_format = "channels_first"

        else:
            """
            Use NHWC format for data
            """
            self.concat_axis = -1
            self.data_format = "channels_last"

        self.fms = fms  # 32 or 16 depending on your memory size

        self.learningrate = learning_rate
        self.weight_dice_loss = weight_dice_loss

        print("Data format = " + self.data_format)
        K.backend.set_image_data_format(self.data_format)

        self.output_path = output_path
        self.inference_filename = inference_filename


        self.metrics = [self.dice_coef, self.soft_dice_coef]

        self.loss = self.dice_coef_loss
        #self.loss = self.combined_dice_ce_loss

        self.optimizer = K.optimizers.Adam(lr=self.learningrate)

        self.custom_objects = {
            "combined_dice_ce_loss": self.combined_dice_ce_loss,
            "dice_coef_loss": self.dice_coef_loss,
            "dice_coef": self.dice_coef,
            "soft_dice_coef": self.soft_dice_coef}

        if use_pconv:
            self.custom_objects.update({'PConv2D': PConv2D})

        self.blocktime = blocktime
        self.num_threads = num_threads
        self.num_inter_threads = num_inter_threads

        self.use_upsampling = use_upsampling
        self.use_dropout = use_dropout
        self.print_model = print_model

    def dice_coef(self, target, prediction, axis=(1, 2), smooth=0.0001):
        """
        Sorenson Dice
        \frac{  2 \times \left | T \right | \cap \left | P \right |}{ \left | T \right | +  \left | P \right |  }
        where T is ground truth mask and P is the prediction mask
        """
        prediction = K.backend.round(prediction)  # Round to 0 or 1

        intersection = tf.reduce_sum(target * prediction, axis=axis)
        union = tf.reduce_sum(target + prediction, axis=axis)
        numerator = tf.constant(2.) * intersection + smooth
        denominator = union + smooth
        coef = numerator / denominator

        return tf.reduce_mean(coef)

    def soft_dice_coef(self, target, prediction, axis=(1, 2), smooth=0.0001):
        """
        Sorenson (Soft) Dice  - Don't round the predictions
        \frac{  2 \times \left | T \right | \cap \left | P \right |}{ \left | T \right | +  \left | P \right |  }
        where T is ground truth mask and P is the prediction mask
        """

        intersection = tf.reduce_sum(target * prediction, axis=axis)
        union = tf.reduce_sum(target + prediction, axis=axis)
        numerator = tf.constant(2.) * intersection + smooth
        denominator = union + smooth
        coef = numerator / denominator

        return tf.reduce_mean(coef)

    def dice_coef_loss(self, target, prediction, axis=(1, 2), smooth=0.0001):
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
        dice_loss = -tf.math.log(2.*numerator) + tf.math.log(denominator)

        return dice_loss

    def combined_dice_ce_loss(self, target, prediction, axis=(1, 2), smooth=0.0001):
        """
        Combined Dice and Binary Cross Entropy Loss
        """
        return self.weight_dice_loss*self.dice_coef_loss(target, prediction, axis, smooth) + \
            (1-self.weight_dice_loss)*K.losses.binary_crossentropy(target, prediction)

    def unet_model(self, imgs_shape, msks_shape,
                   dropout=0.2,
                   final=False):
        """
        U-Net Model
        ===========
        Based on https://arxiv.org/abs/1505.04597
        The default uses UpSampling2D (bilinear interpolation) in
        the decoder path. The alternative is to use Transposed
        Convolution.
        """

        if not final:
            if self.use_upsampling:
                print("Using UpSampling2D")
            else:
                print("Using Transposed Deconvolution")

        num_chan_in = imgs_shape[self.concat_axis]
        num_chan_out = msks_shape[self.concat_axis]

        # You can make the network work on variable input height and width
        # if you pass None as the height and width
#         if self.channels_first:
#             self.input_shape = [num_chan_in, None, None]
#         else:
#             self.input_shape = [None, None, num_chan_in]

        self.input_shape = imgs_shape

        self.num_input_channels = num_chan_in

        inputs = K.layers.Input(self.input_shape, name="MRImages")

        # Convolution parameters
        params = dict(kernel_size=(3, 3), activation="relu",
                      padding="same",
                      kernel_initializer="he_uniform")

        # Transposed convolution parameters
        params_trans = dict(kernel_size=(2, 2), strides=(2, 2),
                            padding="same")

        encodeA = PConv2D(name="encodeAa", filters=self.fms, **params)(inputs)
        encodeA = PConv2D(name="encodeAb", filters=self.fms, **params)(encodeA)
        poolA = K.layers.MaxPooling2D(name="poolA", pool_size=(2, 2))(encodeA)

        encodeB = PConv2D(name="encodeBa", filters=self.fms*2, **params)(poolA)
        encodeB = PConv2D(
            name="encodeBb", filters=self.fms*2, **params)(encodeB)
        poolB = K.layers.MaxPooling2D(name="poolB", pool_size=(2, 2))(encodeB)

        encodeC = PConv2D(name="encodeCa", filters=self.fms*4, **params)(poolB)
        if self.use_dropout:
            encodeC = K.layers.SpatialDropout2D(dropout)(encodeC)
        encodeC = PConv2D(
            name="encodeCb", filters=self.fms*4, **params)(encodeC)

        poolC = K.layers.MaxPooling2D(name="poolC", pool_size=(2, 2))(encodeC)

        encodeD = PConv2D(name="encodeDa", filters=self.fms*8, **params)(poolC)
        if self.use_dropout:
            encodeD = K.layers.SpatialDropout2D(dropout)(encodeD)
        encodeD = PConv2D(
            name="encodeDb", filters=self.fms*8, **params)(encodeD)

        poolD = K.layers.MaxPooling2D(name="poolD", pool_size=(2, 2))(encodeD)

        encodeE = PConv2D(
            name="encodeEa", filters=self.fms*16, **params)(poolD)
        encodeE = PConv2D(
            name="encodeEb", filters=self.fms*16, **params)(encodeE)

        if self.use_upsampling:
            up = K.layers.UpSampling2D(name="upE", size=(2, 2))(encodeE)
        else:
            up = K.layers.Conv2DTranspose(name="transconvE", filters=self.fms*8,
                                          **params_trans)(encodeE)
        concatD = K.layers.concatenate(
            [up, encodeD], axis=self.concat_axis, name="concatD")

        decodeC = PConv2D(
            name="decodeCa", filters=self.fms*8, **params)(concatD)
        decodeC = PConv2D(
            name="decodeCb", filters=self.fms*8, **params)(decodeC)

        if self.use_upsampling:
            up = K.layers.UpSampling2D(name="upC", size=(2, 2))(decodeC)
        else:
            up = K.layers.Conv2DTranspose(name="transconvC", filters=self.fms*4,
                                          **params_trans)(decodeC)
        concatC = K.layers.concatenate(
            [up, encodeC], axis=self.concat_axis, name="concatC")

        decodeB = PConv2D(
            name="decodeBa", filters=self.fms*4, **params)(concatC)
        decodeB = PConv2D(
            name="decodeBb", filters=self.fms*4, **params)(decodeB)

        if self.use_upsampling:
            up = K.layers.UpSampling2D(name="upB", size=(2, 2))(decodeB)
        else:
            up = K.layers.Conv2DTranspose(name="transconvB", filters=self.fms*2,
                                          **params_trans)(decodeB)
        concatB = K.layers.concatenate(
            [up, encodeB], axis=self.concat_axis, name="concatB")

        decodeA = PConv2D(
            name="decodeAa", filters=self.fms*2, **params)(concatB)
        decodeA = PConv2D(
            name="decodeAb", filters=self.fms*2, **params)(decodeA)

        if self.use_upsampling:
            up = K.layers.UpSampling2D(name="upA", size=(2, 2))(decodeA)
        else:
            up = K.layers.Conv2DTranspose(name="transconvA", filters=self.fms,
                                          **params_trans)(decodeA)
        concatA = K.layers.concatenate(
            [up, encodeA], axis=self.concat_axis, name="concatA")

        convOut = PConv2D(name="convOuta", filters=self.fms, **params)(concatA)
        convOut = PConv2D(name="convOutb", filters=self.fms, **params)(convOut)

        prediction = K.layers.Conv2D(name="PredictionMask",
                                     filters=num_chan_out, kernel_size=(1, 1),
                                     activation="sigmoid")(convOut)

        model = K.models.Model(inputs=[inputs], outputs=[
                               prediction], name="2DUNet_pconv_decathlon_brats")

        optimizer = self.optimizer

        if final:
            model.trainable = False
        else:

            model.compile(optimizer=optimizer,
                          loss=self.loss,
                          metrics=self.metrics)

            if self.print_model:
                model.summary()

        return model

    def get_callbacks(self):
        """
        Define any callbacks for the training
        """

        model_filename = os.path.join(
            self.output_path, self.inference_filename)

        print("Writing model to '{}'".format(model_filename))

        # Save model whenever we get better validation loss
        model_checkpoint = K.callbacks.ModelCheckpoint(model_filename,
                                                       verbose=1,
                                                       monitor="val_loss",
                                                       save_best_only=True)

        directoryName = "unet_block{}_inter{}_intra{}".format(self.blocktime,
                                                              self.num_threads,
                                                              self.num_inter_threads)

        # Tensorboard callbacks
        if (self.use_upsampling):
            tensorboard_filename = os.path.join(self.output_path,
                                                "keras_tensorboard_upsampling/{}".format(
                                                    directoryName))
        else:
            tensorboard_filename = os.path.join(self.output_path,
                                                "keras_tensorboard_transposed/{}".format(
                                                    directoryName))

        tensorboard_checkpoint = K.callbacks.TensorBoard(
            log_dir=tensorboard_filename,
            write_graph=True, write_images=True)

        early_stopping = K.callbacks.EarlyStopping(patience=5, restore_best_weights=True)

        return model_filename, [model_checkpoint, early_stopping, tensorboard_checkpoint]

    def evaluate_model(self, model_filename, ds_validation):
        """
        Evaluate the best model on the validation dataset
        """

        model = K.models.load_model(
            model_filename, custom_objects=self.custom_objects)

        print("Evaluating model on test dataset. Please wait...")
        metrics = model.evaluate(
            ds_validation,
            batch_size=self.batch_size,
            verbose=1)

        for idx, metric in enumerate(metrics):
            print("Test dataset {} = {:.4f}".format(
                model.metrics_names[idx], metric))

    def create_model(self, imgs_shape, msks_shape,
                     dropout=0.2,
                     final=False):
        """
        If you have other models, you can try them here
        """
        return self.unet_model(imgs_shape, msks_shape,
                               dropout=dropout,
                               final=final)

    def load_model(self, model_filename):
        """
        Load a model from Keras file
        """
        return K.models.load_model(model_filename, custom_objects=self.custom_objects)

    def print_openvino_mo_command(self, model_filename, input_shape):
        """
        Prints the command for the OpenVINO model optimizer step
        """
        model = self.load_model(model_filename)

        print("Convert the TensorFlow model to OpenVINO by running:\n")
        print("source /opt/intel/openvino/bin/setupvars.sh")
        print("python $INTEL_OPENVINO_DIR/deployment_tools/model_optimizer/mo_tf.py \\")
        print("       --saved_model_dir {} \\".format(model_filename))

        shape_string = "[1"
        for idx in range(len(input_shape)):
            shape_string += ",{}".format(input_shape[idx])
        shape_string += "]"

        print("       --input_shape {} \\".format(shape_string))
        print("       --model_name {} \\".format(self.inference_filename))
        print("       --output_dir {} \\".format(os.path.join(self.output_path, "FP32")))
        print("       --data_type FP32\n\n")
    
