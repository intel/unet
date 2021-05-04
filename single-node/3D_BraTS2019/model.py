#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2020 Intel Corporation
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
from argparser import args
import tensorflow as tf
from tensorflow import keras as K


def dice_coef(target, prediction, axis=(1, 2, 3), smooth=0.0001):
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


def soft_dice_coef(target, prediction, axis=(1, 2, 3), smooth=0.0001):
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


def dice_loss(target, prediction, axis=(1, 2, 3), smooth=0.0001):
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


def unet_3d(input_dim, filters=args.filters,
            number_output_classes=args.number_output_classes,
            use_upsampling=args.use_upsampling,
            concat_axis=-1, model_name=args.saved_model_name):
    """
    3D U-Net
    """

    def ConvolutionBlock(x, name, filters, params):
        """
        Convolutional block of layers
        Per the original paper this is back to back 3D convs
        with batch norm and then ReLU.
        """

        x = K.layers.Conv3D(filters=filters, **params, name=name+"_conv0")(x)
        x = K.layers.BatchNormalization(name=name+"_bn0")(x)
        x = K.layers.Activation("relu", name=name+"_relu0")(x)

        x = K.layers.Conv3D(filters=filters, **params, name=name+"_conv1")(x)
        x = K.layers.BatchNormalization(name=name+"_bn1")(x)
        x = K.layers.Activation("relu", name=name)(x)

        return x

    inputs = K.layers.Input(shape=input_dim, name="MRImages")

    params = dict(kernel_size=(3, 3, 3), activation=None,
                  padding="same",
                  kernel_initializer="he_uniform")

    # Transposed convolution parameters
    params_trans = dict(kernel_size=(2, 2, 2), strides=(2, 2, 2),
                        padding="same",
                        kernel_initializer="he_uniform")

    # BEGIN - Encoding path
    encodeA = ConvolutionBlock(inputs, "encodeA", filters, params)
    poolA = K.layers.MaxPooling3D(name="poolA", pool_size=(2, 2, 2))(encodeA)

    encodeB = ConvolutionBlock(poolA, "encodeB", filters*2, params)
    poolB = K.layers.MaxPooling3D(name="poolB", pool_size=(2, 2, 2))(encodeB)

    encodeC = ConvolutionBlock(poolB, "encodeC", filters*4, params)
    poolC = K.layers.MaxPooling3D(name="poolC", pool_size=(2, 2, 2))(encodeC)

    encodeD = ConvolutionBlock(poolC, "encodeD", filters*8, params)
    poolD = K.layers.MaxPooling3D(name="poolD", pool_size=(2, 2, 2))(encodeD)

    encodeE = ConvolutionBlock(poolD, "encodeE", filters*16, params)
    # END - Encoding path

    # BEGIN - Decoding path
    if use_upsampling:
        up = K.layers.UpSampling3D(name="upE", size=(2, 2, 2))(encodeE)
    else:
        up = K.layers.Conv3DTranspose(name="transconvE", filters=filters*8,
                                      **params_trans)(encodeE)
    concatD = K.layers.concatenate(
        [up, encodeD], axis=concat_axis, name="concatD")

    decodeC = ConvolutionBlock(concatD, "decodeC", filters*8, params)

    if use_upsampling:
        up = K.layers.UpSampling3D(name="upC", size=(2, 2, 2))(decodeC)
    else:
        up = K.layers.Conv3DTranspose(name="transconvC", filters=filters*4,
                                      **params_trans)(decodeC)
    concatC = K.layers.concatenate(
        [up, encodeC], axis=concat_axis, name="concatC")

    decodeB = ConvolutionBlock(concatC, "decodeB", filters*4, params)

    if use_upsampling:
        up = K.layers.UpSampling3D(name="upB", size=(2, 2, 2))(decodeB)
    else:
        up = K.layers.Conv3DTranspose(name="transconvB", filters=filters*2,
                                      **params_trans)(decodeB)
    concatB = K.layers.concatenate(
        [up, encodeB], axis=concat_axis, name="concatB")

    decodeA = ConvolutionBlock(concatB, "decodeA", filters*2, params)

    if use_upsampling:
        up = K.layers.UpSampling3D(name="upA", size=(2, 2, 2))(decodeA)
    else:
        up = K.layers.Conv3DTranspose(name="transconvA", filters=filters,
                                      **params_trans)(decodeA)
    concatA = K.layers.concatenate(
        [up, encodeA], axis=concat_axis, name="concatA")

    # END - Decoding path

    convOut = ConvolutionBlock(concatA, "convOut", filters, params)

    prediction = K.layers.Conv3D(name="PredictionMask",
                                 filters=number_output_classes,
                                 kernel_size=(1, 1, 1),
                                 activation="sigmoid")(convOut)

    model = K.models.Model(inputs=[inputs], outputs=[prediction],
                           name=model_name)

    if args.print_model:
        model.summary()

    return model
