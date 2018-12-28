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


# """
# BEGIN - Limit Tensoflow to only use specific GPU
# """
# import os

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"   # 0,1,2,3
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" # Supress Tensforflow debug messages

# # """
# # END - Limit Tensoflow to only use specific GPU
# # """

# numactl -p 1 python train.py --num_threads=50 --num_inter_threads=2
# --batch_size=128 --blocktime=0
#import ngraph_bridge

import numpy as np
import tensorflow as tf
import time
import os
import settings    # Use the custom settings.py file for default parameters
import argparse

parser = argparse.ArgumentParser(
    description="Trains 2D U-Net model (Keras/TF) on BraTS dataset.",
    add_help=True, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--data_path", default=settings.DATA_PATH,
                    help="the path to the data")
parser.add_argument("--data_filename", default=settings.DATA_FILENAME,
                    help="the HDF5 data filename")
parser.add_argument("--output_path", default=settings.OUT_PATH,
                    help="the folder to save the model and checkpoints")
parser.add_argument("--inference_filename", default=settings.INFERENCE_FILENAME,
                    help="the Keras inference model filename")
parser.add_argument("--use_upsampling",
                    help="use upsampling instead of transposed convolution",
                    action="store_true", default=settings.USE_UPSAMPLING)
parser.add_argument("--num_threads", type=int,
                    default=settings.NUM_INTRA_THREADS,
                    help="the number of threads")
parser.add_argument(
    "--num_inter_threads",
    type=int,
    default=settings.NUM_INTER_THREADS,
    help="the number of intraop threads")
parser.add_argument("--batch_size", type=int, default=settings.BATCH_SIZE,
                    help="the batch size for training")
parser.add_argument(
    "--blocktime",
    type=int,
    default=settings.BLOCKTIME,
    help="blocktime")
parser.add_argument("--epochs", type=int, default=settings.EPOCHS,
                    help="number of epochs to train")
parser.add_argument(
    "--learningrate",
    type=float,
    default=settings.LEARNING_RATE,
    help="learningrate")
parser.add_argument(
    "--keras_api",
    help="use keras instead of tf.keras",
    action="store_true",
    default=settings.USE_KERAS_API)
parser.add_argument("--channels_first", help="use channels first data format",
                    action="store_true", default=settings.CHANNELS_FIRST)
parser.add_argument("--print_model", help="print the model",
                    action="store_true", default=settings.PRINT_MODEL)
parser.add_argument(
    "--trace",
    help="create a tensorflow timeline trace",
    action="store_true",
    default=settings.CREATE_TRACE_TIMELINE)

args = parser.parse_args()

num_threads = args.num_threads
num_inter_op_threads = args.num_inter_threads

if (args.blocktime > 1000):
    blocktime = "infinite"
else:
    blocktime = str(args.blocktime)

# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Get rid of the AVX, SSE warnings

os.environ["KMP_BLOCKTIME"] = blocktime
os.environ["KMP_AFFINITY"] = "compact,1,0,granularity=fine"

os.environ["OMP_NUM_THREADS"] = str(num_threads)
os.environ["INTRA_THREADS"] = str(num_threads)
os.environ["INTER_THREADS"] = str(num_inter_op_threads)
os.environ["KMP_SETTINGS"] = "0"  # Show the settings at runtime

# The timeline trace for TF is saved to this file.
# To view it, run this python script, then load the json file by
# starting Google Chrome browser and pointing the URI to chrome://trace
# There should be a button at the top left of the graph where
# you can load in this json file.
timeline_filename = "timeline_ge_unet_{}_{}_{}.json".format(
    blocktime, num_threads, num_inter_op_threads)

config = tf.ConfigProto(intra_op_parallelism_threads=num_threads,
                        inter_op_parallelism_threads=num_inter_op_threads)

sess = tf.Session(config=config)

run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
run_metadata = tf.RunMetadata()  # For Tensorflow trace

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
if args.keras_api:
    import keras as K
else:
    from tensorflow import keras as K

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

def dice_coef_loss(target, prediction, axis=(1, 2, 3), smooth=1.):
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

def unet_model(img_height=128,
               img_width=128,
               num_chan_in=1,
               num_chan_out=1,
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

    if args.use_upsampling:
        print("Using UpSampling2D")
    else:
        print("Using Transposed Deconvolution")

    if args.channels_first:
        inputs = K.layers.Input((num_chan_in, img_height, img_width),
                                name="MRImages")
    else:
        inputs = K.layers.Input((img_height, img_width, num_chan_in),
                                name="MRImages")

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
        up = K.layers.Conv2DTranspose(name="transConv6", filters=fms*8,
                                      **params_trans)(conv5)
    up6 = K.layers.concatenate([up, conv4], axis=concat_axis)

    conv6 = K.layers.Conv2D(name="conv6a", filters=fms*8, **params)(up6)
    conv6 = K.layers.Conv2D(name="conv6b", filters=fms*8, **params)(conv6)

    if args.use_upsampling:
        up = K.layers.UpSampling2D(name="up7", size=(2, 2))(conv6)
    else:
        up = K.layers.Conv2DTranspose(name="transConv7", filters=fms*4,
                                      **params_trans)(conv6)
    up7 = K.layers.concatenate([up, conv3], axis=concat_axis)

    conv7 = K.layers.Conv2D(name="conv7a", filters=fms*4, **params)(up7)
    conv7 = K.layers.Conv2D(name="conv7b", filters=fms*4, **params)(conv7)

    if args.use_upsampling:
        up = K.layers.UpSampling2D(name="up8", size=(2, 2))(conv7)
    else:
        up = K.layers.Conv2DTranspose(name="transConv8", filters=fms*2,
                                      **params_trans)(conv7)
    up8 = K.layers.concatenate([up, conv2], axis=concat_axis)

    conv8 = K.layers.Conv2D(name="conv8a", filters=fms*2, **params)(up8)
    conv8 = K.layers.Conv2D(name="conv8b", filters=fms*2, **params)(conv8)

    if args.use_upsampling:
        up = K.layers.UpSampling2D(name="up9", size=(2, 2))(conv8)
    else:
        up = K.layers.Conv2DTranspose(name="transConv9", filters=fms,
                                      **params_trans)(conv8)
    up9 = K.layers.concatenate([up, conv1], axis=concat_axis)

    conv9 = K.layers.Conv2D(name="conv9a", filters=fms, **params)(up9)
    conv9 = K.layers.Conv2D(name="conv9b", filters=fms, **params)(conv9)

    prediction = K.layers.Conv2D(name="PredictionMask",
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

        if args.trace:
            model.compile(optimizer=optimizer,
                          loss=loss,
                          metrics=metrics,
                          options=run_options, run_metadata=run_metadata)
        else:
            model.compile(optimizer=optimizer,
                          loss=loss,
                          metrics=metrics)

        if args.print_model:
            model.summary()

    return model

def train_and_predict(data_path, n_epoch, mode=1):
    """
    Create a model, load the data, and train it.
    """

    """
    Load data from HDF5 file
    """

    def process_data(array):
        # Data was saved as NHWC (channels last)
        if args.channels_first:  # NCHW
            return np.swapaxes(array, 1, 3)
        else:
            return array

    imgs_train = K.utils.HDF5Matrix(os.path.join(data_path, args.data_filename),
                                    "imgs_train", normalizer=process_data)
    imgs_validation = K.utils.HDF5Matrix(os.path.join(data_path, args.data_filename),
                                         "imgs_validation",
                                         normalizer=process_data)
    msks_train = K.utils.HDF5Matrix(os.path.join(data_path, args.data_filename),
                                    "msks_train", normalizer=process_data)
    msks_validation = K.utils.HDF5Matrix(os.path.join(data_path, args.data_filename),
                                         "msks_validation",
                                         normalizer=process_data)

    print("-" * 30)
    print("Creating and compiling model...")
    print("-" * 30)

    if args.channels_first:
        img_height = imgs_train.shape[3]
        img_width = imgs_train.shape[2]
        input_no = imgs_train.shape[1]
        output_no = msks_train.shape[1]
    else:
        img_height = imgs_train.shape[1]
        img_width = imgs_train.shape[2]
        input_no = imgs_train.shape[3]
        output_no = msks_train.shape[3]

    """
    Define the model
    """
    model = unet_model(img_height, img_width, input_no, output_no)

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

    directoryName = "unet_block{}_inter{}_intra{}".format(blocktime,
                                                          num_threads,
                                                          num_inter_op_threads)

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

    print("-" * 30)
    print("Fitting model...")
    print("-" * 30)

    history = K.callbacks.History()

    print("Batch size = {}".format(args.batch_size))

    print("Training image dimensions:   {}".format(imgs_train.shape))
    print("Training mask dimensions:    {}".format(msks_train.shape))
    print("Validation image dimensions: {}".format(imgs_validation.shape))
    print("Validation mask dimensions:  {}".format(msks_validation.shape))

    history = model.fit(imgs_train, msks_train,
                        batch_size=args.batch_size,
                        epochs=n_epoch,
                        validation_data=(imgs_validation, msks_validation),
                        verbose=1, shuffle="batch",
                        callbacks=[model_checkpoint,
                                   tensorboard_checkpoint])

    if args.trace:
        """
        Save the training timeline
        """
        from tensorflow.python.client import timeline

        fetched_timeline = timeline.Timeline(run_metadata.step_stats)
        chrome_trace = fetched_timeline.generate_chrome_trace_format()
        with open(timeline_filename, "w") as f:
            print("Saved Tensorflow trace to: {}".format(timeline_filename))
            f.write(chrome_trace)

    print("-" * 30)
    print("Loading the best trained model ...")
    print("-" * 30)
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

    # Save final model without custom loss and metrics
    # This way we can easily re-load it into Keras for inference
    model.save_weights(os.path.join(args.output_path, "weights.hdf5"))
    # Model without Dice and custom metrics
    model = unet_model(img_height, img_width, input_no, output_no, final=True)
    model.load_weights(os.path.join(args.output_path, "weights.hdf5"))

    model_json = model.to_json()
    with open(os.path.join(args.output_path, "model.json"), "w") as json_file:
        json_file.write(model_json)

    model.save_weights(os.path.join(args.output_path, "weights.hdf5"))

    model_fn = os.path.join(args.output_path, args.inference_filename)

    print("Writing final model (without custom Dice metrics) "
          "for inference to {}".format(model_fn))
    print("Please use that version for inference.")
    model.save(model_fn, include_optimizer=False)


if __name__ == "__main__":

    os.system("lscpu")

    import datetime
    print("Started script on {}".format(datetime.datetime.now()))

    print("args = {}".format(args))
    os.system("uname -a")
    print("TensorFlow version: {}".format(tf.__version__))
    start_time = time.time()

    train_and_predict(args.data_path,
                      args.epochs,
                      settings.MODE)

    print(
        "Total time elapsed for program = {} seconds".format(
            time.time() -
            start_time))
    print("Stopped script on {}".format(datetime.datetime.now()))
