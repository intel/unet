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

from argparser import args
if args.keras_api:
    import keras as K
else:
    from tensorflow import keras as K


class unet(object):

    def __init__(self, use_upsampling=False, learning_rate=0.001,
                 n_cl_in=1, n_cl_out=1, feature_maps = 16,
                 dropout=0.2, print_summary=False,
                 channels_last = True):

        self.channels_last = channels_last
        if channels_last:
            self.concat_axis = -1
            self.data_format = "channels_last"

        else:
            self.concat_axis = 1
            self.data_format = "channels_first"

        #print("Data format = " + self.data_format)
        K.backend.set_image_data_format(self.data_format)

        self.fms = feature_maps # 16 or 32 feature maps in the first convolutional layer

        self.use_upsampling = use_upsampling
        self.dropout = dropout
        self.print_summary = print_summary
        self.n_cl_in = n_cl_in
        self.n_cl_out = n_cl_out

        # self.loss = self.dice_coef_loss
        self.loss = self.combined_dice_ce_loss

        self.learning_rate = learning_rate
        self.optimizer = K.optimizers.Adam(lr=self.learning_rate)

        self.metrics= [self.dice_coef, self.soft_dice_coef, "accuracy",
                 self.sensitivity, self.specificity]

        self.custom_objects = {
            "combined_dice_ce_loss": self.combined_dice_ce_loss,
            "dice_coef_loss": self.dice_coef_loss,
            "dice_coef": self.dice_coef,
            "soft_dice_coef": self.soft_dice_coef,
            "sensitivity": self.sensitivity,
            "specificity": self.specificity}

        self.model = self.unet_3d()

    def dice_coef(self, target, prediction, axis=(1, 2, 3), smooth=0.01):
        """
        Sorenson Dice
        \frac{  2 \times \left | T \right | \cap \left | P \right |}{ \left | T \right | +  \left | P \right |  }
        where T is ground truth mask and P is the prediction mask
        """
        prediction = tf.round(prediction)  # Round to 0 or 1

        intersection = tf.reduce_sum(target * prediction, axis=axis)
        union = tf.reduce_sum(target + prediction, axis=axis)
        numerator = tf.constant(2.) * intersection + smooth
        denominator = union + smooth
        coef = numerator / denominator

        return tf.reduce_mean(coef)

    def soft_dice_coef(self, target, prediction, axis=(1, 2, 3), smooth=0.01):
        """
        Sorenson (Soft) Dice - Don't round predictions
        \frac{  2 \times \left | T \right | \cap \left | P \right |}{ \left | T \right | +  \left | P \right |  }
        where T is ground truth mask and P is the prediction mask
        """
        intersection = tf.reduce_sum(target * prediction, axis=axis)
        union = tf.reduce_sum(target + prediction, axis=axis)
        numerator = tf.constant(2.) * intersection + smooth
        denominator = union + smooth
        coef = numerator / denominator

        return tf.reduce_mean(coef)


    def dice_coef_loss(self, target, prediction, axis=(1, 2, 3), smooth=0.1):
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


    def combined_dice_ce_loss(self, target, prediction, axis=(1, 2, 3),
                              smooth=0.1, weight=0.7):
        """
        Combined Dice and Binary Cross Entropy Loss
        """
        return weight*self.dice_coef_loss(target, prediction, axis, smooth) + \
            (1-weight)*K.losses.binary_crossentropy(target, prediction)


    def unet_3d(self):
        """
        3D U-Net
        """
        def ConvolutionBlock(x, name, fms, params):
            """
            Convolutional block of layers
            Per the original paper this is back to back 3D convs
            with batch norm and then ReLU.
            """

            x = K.layers.Conv3D(filters=fms, **params, name=name+"_conv0")(x)
            x = K.layers.BatchNormalization(name=name+"_bn0")(x)
            x = K.layers.Activation("relu", name=name+"_relu0")(x)

            x = K.layers.Conv3D(filters=fms, **params, name=name+"_conv1")(x)
            x = K.layers.BatchNormalization(name=name+"_bn1")(x)
            x = K.layers.Activation("relu", name=name)(x)

            return x

        if self.channels_last:
            input_shape = [None, None, None, self.n_cl_in]
        else:
            input_shape = [self.n_cl_in, None, None, None]

        inputs = K.layers.Input(shape=input_shape,
                                name="MRImages")

        params = dict(kernel_size=(3, 3, 3), activation=None,
                      padding="same", data_format=self.data_format,
                      kernel_initializer="he_uniform")

        # Transposed convolution parameters
        params_trans = dict(data_format=self.data_format,
                            kernel_size=(2, 2, 2), strides=(2, 2, 2),
                            padding="same")


        # BEGIN - Encoding path
        encodeA = ConvolutionBlock(inputs, "encodeA", self.fms, params)
        poolA = K.layers.MaxPooling3D(name="poolA", pool_size=(2, 2, 2))(encodeA)

        encodeB = ConvolutionBlock(poolA, "encodeB", self.fms*2, params)
        poolB = K.layers.MaxPooling3D(name="poolB", pool_size=(2, 2, 2))(encodeB)

        encodeC = ConvolutionBlock(poolB, "encodeC", self.fms*4, params)
        poolC = K.layers.MaxPooling3D(name="poolC", pool_size=(2, 2, 2))(encodeC)

        encodeD = ConvolutionBlock(poolC, "encodeD", self.fms*8, params)
        poolD = K.layers.MaxPooling3D(name="poolD", pool_size=(2, 2, 2))(encodeD)

        encodeE = ConvolutionBlock(poolD, "encodeE", self.fms*16, params)
        # END - Encoding path

        # BEGIN - Decoding path
        if self.use_upsampling:
            up = K.layers.UpSampling3D(name="upE", size=(2, 2, 2),
                                       interpolation="bilinear")(encodeE)
        else:
            up = K.layers.Conv3DTranspose(name="transconvE", filters=self.fms*8,
                                          **params_trans)(encodeE)
        concatD = K.layers.concatenate(
            [up, encodeD], axis=self.concat_axis, name="concatD")

        decodeC = ConvolutionBlock(concatD, "decodeC", self.fms*8, params)

        if self.use_upsampling:
            up = K.layers.UpSampling3D(name="upC", size=(2, 2, 2),
                                       interpolation="bilinear")(decodeC)
        else:
            up = K.layers.Conv3DTranspose(name="transconvC", filters=self.fms*4,
                                          **params_trans)(decodeC)
        concatC = K.layers.concatenate(
            [up, encodeC], axis=self.concat_axis, name="concatC")

        decodeB = ConvolutionBlock(concatC, "decodeB", self.fms*4, params)

        if self.use_upsampling:
            up = K.layers.UpSampling3D(name="upB", size=(2, 2, 2),
                                       interpolation="bilinear")(decodeB)
        else:
            up = K.layers.Conv3DTranspose(name="transconvB", filters=self.fms*2,
                                          **params_trans)(decodeB)
        concatB = K.layers.concatenate(
            [up, encodeB], axis=self.concat_axis, name="concatB")

        decodeA = ConvolutionBlock(concatB, "decodeA", self.fms*2, params)

        if self.use_upsampling:
            up = K.layers.UpSampling3D(name="upA", size=(2, 2, 2),
                                       interpolation="bilinear")(decodeA)
        else:
            up = K.layers.Conv3DTranspose(name="transconvA", filters=self.fms,
                                          **params_trans)(decodeA)
        concatA = K.layers.concatenate(
            [up, encodeA], axis=self.concat_axis, name="concatA")

        # END - Decoding path

        convOut = ConvolutionBlock(concatA, "convOut", self.fms, params)

        prediction = K.layers.Conv3D(name="PredictionMask",
                                     filters=self.n_cl_out, kernel_size=(1, 1, 1),
                                     data_format=self.data_format,
                                     activation="sigmoid")(convOut)

        model = K.models.Model(inputs=[inputs], outputs=[prediction])

        if self.print_summary:
            model.summary()

        return model


    def sensitivity(self, target, prediction, axis=(1, 2, 3), smooth=0.0001):
        """
        Sensitivity
        """
        prediction = tf.round(prediction)

        intersection = tf.reduce_sum(prediction * target, axis=axis)
        coef = (intersection + smooth) / (tf.reduce_sum(target,
                                                        axis=axis) + smooth)
        return tf.reduce_mean(coef)


    def specificity(self, target, prediction, axis=(1, 2, 3), smooth=0.0001):
        """
        Specificity
        """
        prediction = tf.round(prediction)

        intersection = tf.reduce_sum(prediction * target, axis=axis)
        coef = (intersection + smooth) / (tf.reduce_sum(prediction,
                                                        axis=axis) + smooth)
        return tf.reduce_mean(coef)
