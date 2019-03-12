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

"""
This module contains all of the model definition code.
You can try custom models by modifying the code here.
"""

from argparser import args
import os
import time

import tensorflow as tf # conda install -c anaconda tensorflow

if args.keras_api:
    import keras as K
else:
    from tensorflow import keras as K

if args.channels_first:
    """
    Use NCHW format for data
    """
    concat_axis = 1
    data_format = "channels_first"

else:
    """
    Use NHWC format for data
    """
    concat_axis = -1
    data_format = "channels_last"


print("Data format = " + data_format)
K.backend.set_image_data_format(data_format)


def dice_coef(y_true, y_pred, axis=(1, 2), smooth=1.):
    """
    Sorenson (Soft) Dice
    \frac{  2 \times \left | T \right | \cap \left | P \right |}{ \left | T \right | +  \left | P \right |  }
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


def combined_dice_ce_loss(y_true, y_pred, axis=(1, 2), smooth=1.,
                          weight=args.weight_dice_loss):
    """
    Combined Dice and Binary Cross Entropy Loss
    """
    return weight*dice_coef_loss(y_true, y_pred, axis, smooth) + \
        (1-weight)*K.losses.binary_crossentropy(y_true, y_pred)


def unet_model(imgs_shape, msks_shape,
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
        if args.use_upsampling:
            print("Using UpSampling2D")
        else:
            print("Using Transposed Deconvolution")

    num_chan_out = msks_shape[-1]

    inputs = K.layers.Input(imgs_shape[1:], name="MRImages")

    # Convolution parameters
    params = dict(kernel_size=(3, 3), activation="relu",
                  padding="same", data_format=data_format,
                  kernel_initializer="he_uniform")

    # Transposed convolution parameters
    params_trans = dict(data_format=data_format,
                        kernel_size=(2, 2), strides=(2, 2),
                        padding="same")

    fms = args.featuremaps  #32 or 16 depending on your memory size

    encodeA = K.layers.Conv2D(name="encodeAa", filters=fms, **params)(inputs)
    encodeA = K.layers.Conv2D(name="encodeAb", filters=fms, **params)(encodeA)
    poolA = K.layers.MaxPooling2D(name="poolA", pool_size=(2, 2))(encodeA)

    encodeB = K.layers.Conv2D(name="encodeBa", filters=fms*2, **params)(poolA)
    encodeB = K.layers.Conv2D(name="encodeBb", filters=fms*2, **params)(encodeB)
    poolB = K.layers.MaxPooling2D(name="poolB", pool_size=(2, 2))(encodeB)

    encodeC = K.layers.Conv2D(name="encodeCa", filters=fms*4, **params)(poolB)
    if args.use_dropout:
        encodeC = K.layers.SpatialDropout2D(dropout,
                                          data_format=data_format)(encodeC)
    encodeC = K.layers.Conv2D(name="encodeCb", filters=fms*4, **params)(encodeC)

    poolC = K.layers.MaxPooling2D(name="poolC", pool_size=(2, 2))(encodeC)

    encodeD = K.layers.Conv2D(name="encodeDa", filters=fms*8, **params)(poolC)
    if args.use_dropout:
        encodeD = K.layers.SpatialDropout2D(dropout,
                                          data_format=data_format)(encodeD)
    encodeD = K.layers.Conv2D(name="encodeDb", filters=fms*8, **params)(encodeD)

    poolD = K.layers.MaxPooling2D(name="poolD", pool_size=(2, 2))(encodeD)

    encodeE = K.layers.Conv2D(name="encodeEa", filters=fms*16, **params)(poolD)
    encodeE = K.layers.Conv2D(name="encodeEb", filters=fms*16, **params)(encodeE)

    if args.use_upsampling:
        up = K.layers.UpSampling2D(name="upE", size=(2, 2),
                                   interpolation="bilinear")(encodeE)
    else:
        up = K.layers.Conv2DTranspose(name="transconvE", filters=fms*8,
                                      **params_trans)(encodeE)
    concatD = K.layers.concatenate([up, encodeD], axis=concat_axis, name="concatD")

    decodeC = K.layers.Conv2D(name="decodeCa", filters=fms*8, **params)(concatD)
    decodeC = K.layers.Conv2D(name="decodeCb", filters=fms*8, **params)(decodeC)

    if args.use_upsampling:
        up = K.layers.UpSampling2D(name="upC", size=(2, 2),
                                   interpolation="bilinear")(decodeC)
    else:
        up = K.layers.Conv2DTranspose(name="transconvC", filters=fms*4,
                                      **params_trans)(decodeC)
    concatC = K.layers.concatenate([up, encodeC], axis=concat_axis, name="concatC")

    decodeB = K.layers.Conv2D(name="decodeBa", filters=fms*4, **params)(concatC)
    decodeB = K.layers.Conv2D(name="decodeBb", filters=fms*4, **params)(decodeB)

    if args.use_upsampling:
        up = K.layers.UpSampling2D(name="upB", size=(2, 2),
                                   interpolation="bilinear")(decodeB)
    else:
        up = K.layers.Conv2DTranspose(name="transconvB", filters=fms*2,
                                      **params_trans)(decodeB)
    concatB = K.layers.concatenate([up, encodeB], axis=concat_axis, name="concatB")

    decodeA = K.layers.Conv2D(name="decodeAa", filters=fms*2, **params)(concatB)
    decodeA = K.layers.Conv2D(name="decodeAb", filters=fms*2, **params)(decodeA)

    if args.use_upsampling:
        up = K.layers.UpSampling2D(name="upA", size=(2, 2),
                                   interpolation="bilinear")(decodeA)
    else:
        up = K.layers.Conv2DTranspose(name="transconvA", filters=fms,
                                      **params_trans)(decodeA)
    concatA = K.layers.concatenate([up, encodeA], axis=concat_axis, name="concatA")

    convOut = K.layers.Conv2D(name="convOuta", filters=fms, **params)(concatA)
    convOut = K.layers.Conv2D(name="convOutb", filters=fms, **params)(convOut)

    prediction = K.layers.Conv2D(name="PredictionMask",
                                 filters=num_chan_out, kernel_size=(1, 1),
                                 data_format=data_format,
                                 activation="sigmoid")(convOut)

    model = K.models.Model(inputs=[inputs], outputs=[prediction])

    optimizer = K.optimizers.Adam(lr=args.learningrate)

    if final:
        model.trainable = False
    else:
        metrics = ["accuracy", dice_coef]
        # loss = dice_coef_loss
        loss = combined_dice_ce_loss

        model.compile(optimizer=optimizer,
                      loss=loss,
                      metrics=metrics)

        if args.print_model:
            model.summary()

    return model


def get_callbacks():
    """
    Define any callbacks for the training
    """

    model_fn = os.path.join(args.output_path, args.inference_filename)

    print("Writing model to '{}'".format(model_fn))

    # Save model whenever we get better validation loss
    model_checkpoint = K.callbacks.ModelCheckpoint(model_fn,
                                                   verbose=1,
                                                   monitor="val_loss",
                                                   save_best_only=True)

    directoryName = "unet_block{}_inter{}_intra{}".format(args.blocktime,
                                                          args.num_threads,
                                                          args.num_inter_threads)

    # Tensorboard callbacks
    if (args.use_upsampling):
        tensorboard_filename = os.path.join(args.output_path,
                                            "keras_tensorboard_upsampling"
                                            "_batch{}/{}".format(
                                                args.batch_size, directoryName))
    else:
        tensorboard_filename = os.path.join(args.output_path,
                                            "keras_tensorboard_transposed"
                                            "_batch{}/{}".format(
                                                args.batch_size, directoryName))

    tensorboard_checkpoint = K.callbacks.TensorBoard(
        log_dir=tensorboard_filename,
        write_graph=True, write_images=True)

    return model_fn, [model_checkpoint, tensorboard_checkpoint]


def evaluate_model(model_fn, imgs_validation, msks_validation):
    """
    Evaluate the best model on the validation dataset
    """

    model = K.models.load_model(
        model_fn, custom_objects={
            "combined_dice_ce_loss": combined_dice_ce_loss,
            "dice_coef_loss": dice_coef_loss,
            "dice_coef": dice_coef})

    K.backend.set_learning_phase(0)
    start_inference = time.time()
    print("Evaluating model. Please wait...")
    loss, accuracy, metric = model.evaluate(
        imgs_validation,
        msks_validation,
        batch_size=args.batch_size,
        verbose=1)
    elapsed_time = time.time() - start_inference
    print("{} images in {:.2f} seconds => {:.3f} images per "
          "second inference".format(
              imgs_validation.shape[0], elapsed_time,
              imgs_validation.shape[0] / elapsed_time))
    print("Mean Dice score for predictions = {:.4f}".format(metric))

    return model


def load_model(imgs_shape, msks_shape,
               dropout=0.2,
               final=False):
    """
    If you have other models, you can try them here
    """
    return unet_model(imgs_shape, msks_shape,
                      dropout=dropout,
                      final=final)
