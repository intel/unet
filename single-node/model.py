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

    inputs = K.layers.Input(imgs_shape[1:], name="mrimages")

    # Convolution parameters
    params = dict(kernel_size=(3, 3), activation="relu",
                  padding="same", data_format=data_format,
                  kernel_initializer="he_uniform")

    # Transposed convolution parameters
    params_trans = dict(data_format=data_format,
                        kernel_size=(2, 2), strides=(2, 2),
                        padding="same")

    fms = 16

    conv1 = K.layers.Conv2D(name="conv1a", filters=fms, **params)(inputs)
    conv1 = K.layers.Conv2D(name="conv1b", filters=fms, **params)(conv1)
    pool1 = K.layers.MaxPooling2D(name="pool1", pool_size=(2, 2))(conv1)

    conv2 = K.layers.Conv2D(name="conv2a", filters=fms*2, **params)(pool1)
    conv2 = K.layers.Conv2D(name="conv2b", filters=fms*2, **params)(conv2)
    pool2 = K.layers.MaxPooling2D(name="pool2", pool_size=(2, 2))(conv2)

    conv3 = K.layers.Conv2D(name="conv3a", filters=fms*4, **params)(pool2)
    conv3 = K.layers.SpatialDropout2D(dropout, data_format=data_format)(conv3)
    conv3 = K.layers.Conv2D(name="conv3b", filters=fms*4, **params)(conv3)

    pool3 = K.layers.MaxPooling2D(name="pool3", pool_size=(2, 2))(conv3)

    conv4 = K.layers.Conv2D(name="conv4a", filters=fms*8, **params)(pool3)
    conv4 = K.layers.SpatialDropout2D(dropout, data_format=data_format)(conv4)
    conv4 = K.layers.Conv2D(name="conv4b", filters=fms*8, **params)(conv4)

    pool4 = K.layers.MaxPooling2D(name="pool4", pool_size=(2, 2))(conv4)

    conv5 = K.layers.Conv2D(name="conv5a", filters=fms*16, **params)(pool4)
    conv5 = K.layers.Conv2D(name="conv5b", filters=fms*16, **params)(conv5)

    if args.use_upsampling:
        up = K.layers.UpSampling2D(name="up6", size=(2, 2))(conv5)
    else:
        up = K.layers.Conv2DTranspose(name="transconv6", filters=fms*8,
                                      **params_trans)(conv5)
    concat1 = K.layers.concatenate([up, conv4], axis=concat_axis, name="concat1")

    conv6 = K.layers.Conv2D(name="conv6a", filters=fms*8, **params)(concat1)
    conv6 = K.layers.Conv2D(name="conv6b", filters=fms*8, **params)(conv6)

    if args.use_upsampling:
        up = K.layers.UpSampling2D(name="up7", size=(2, 2))(conv6)
    else:
        up = K.layers.Conv2DTranspose(name="transconv7", filters=fms*4,
                                      **params_trans)(conv6)
    concat2 = K.layers.concatenate([up, conv3], axis=concat_axis, name="concat2")

    conv7 = K.layers.Conv2D(name="conv7a", filters=fms*4, **params)(concat2)
    conv7 = K.layers.Conv2D(name="conv7b", filters=fms*4, **params)(conv7)

    if args.use_upsampling:
        up = K.layers.UpSampling2D(name="up8", size=(2, 2))(conv7)
    else:
        up = K.layers.Conv2DTranspose(name="transconv8", filters=fms*2,
                                      **params_trans)(conv7)
    concat3 = K.layers.concatenate([up, conv2], axis=concat_axis, name="concat3")

    conv8 = K.layers.Conv2D(name="conv8a", filters=fms*2, **params)(concat3)
    conv8 = K.layers.Conv2D(name="conv8b", filters=fms*2, **params)(conv8)

    if args.use_upsampling:
        up = K.layers.UpSampling2D(name="up9", size=(2, 2))(conv8)
    else:
        up = K.layers.Conv2DTranspose(name="transconv9", filters=fms,
                                      **params_trans)(conv8)
    concat4 = K.layers.concatenate([up, conv1], axis=concat_axis, name="concat4")

    conv9 = K.layers.Conv2D(name="conv9a", filters=fms, **params)(concat4)
    conv9 = K.layers.Conv2D(name="conv9b", filters=fms, **params)(conv9)

    prediction = K.layers.Conv2D(name="predictionmask",
                                 filters=num_chan_out, kernel_size=(1, 1),
                                 data_format=data_format,
                                 activation="sigmoid")(conv9)

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

    if (args.use_upsampling):
        model_fn = os.path.join(args.output_path, "unet_model_upsampling.hdf5")
    else:
        model_fn = os.path.join(args.output_path, "unet_model_transposed.hdf5")

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


def save_inference_model(model, imgs_shape, msks_shape):
    """
    Save the model without the training nodes and custom loss
    functions so that it can be used for inference only.
    """

    # Save final model without custom loss and metrics
    # This way we can easily re-load it into Keras for inference
    model.save_weights(os.path.join(args.output_path, "weights.hdf5"))

    # Model without Dice and custom metrics
    model = load_model(imgs_shape, msks_shape, final=True)
    model.load_weights(os.path.join(args.output_path, "weights.hdf5"))

    model_json = model.to_json()
    with open(os.path.join(args.output_path, "model.json"), "w") as json_file:
        json_file.write(model_json)

    model.save_weights(os.path.join(args.output_path, "weights.hdf5"))

    model_fn = os.path.join(args.output_path, args.inference_filename)

    print("Writing final model (without custom Dice metrics) "
          "for inference to {}".format(model_fn))
    print("Please use that version for inference.")
    K.backend.set_learning_phase(0)
    model.save(model_fn, include_optimizer=False)

    # See if experimental TensorFlow module works
    # For TF >= 1.12, we're supposed to be able to directly
    # save Keras models to TensorFlow serving. This will be
    # great if it works.
    #saved_model_path = tf.contrib.saved_model.save_keras_model(model, "./saved_models")
    #print("Wrote TF serving model to ", saved_model_path)


def load_model(imgs_shape, msks_shape,
               dropout=0.2,
               final=False):
    """
    If you have other models, you can try them here
    """
    return unet_model(imgs_shape, msks_shape,
                      dropout=dropout,
                      final=final)
