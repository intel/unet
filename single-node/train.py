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

import psutil
import settings    # Use the custom settings.py file for default parameters
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", default=settings.DATA_PATH,
                    help="the path to the data")
parser.add_argument("--output_path", default=settings.OUT_PATH,
                    help="the folder to save the model and checkpoints")
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

batch_size = args.batch_size

import os

num_threads = args.num_threads
num_inter_op_threads = args.num_inter_threads

if (args.blocktime > 1000):
    blocktime = "infinite"
else:
    blocktime = str(args.blocktime)

#os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Get rid of the AVX, SSE warnings

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

import time

import tensorflow as tf

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

import numpy as np
import os

from preprocess import *
import settings


def dice_coef(y_true, y_pred, smooth=1.):
   intersection = tf.reduce_sum(y_true * y_pred, axis=(1, 2, 3))
   union = tf.reduce_sum(y_true + y_pred, axis=(1, 2, 3))
   numerator = tf.constant(2.) * intersection + smooth
   denominator = union + smooth
   coef = numerator / denominator
   return tf.reduce_mean(coef)


def dice_coef_loss(y_true, y_pred, smooth=1.):

    y_true_f = K.backend.flatten(y_true)
    y_pred_f = K.backend.flatten(y_pred)
    intersection = K.backend.sum(y_true_f * y_pred_f)
    loss = -K.backend.log(2. * intersection + smooth) + \
        K.backend.log((K.backend.sum(y_true_f) +
                       K.backend.sum(y_pred_f) + smooth))

    return loss


def unet_model(img_height=224,
               img_width=224,
               num_chan_in=3,
               num_chan_out=3,
               dropout=0.2, final=False):

    if args.use_upsampling:
        print("Using UpSampling2D")
    else:
        print("Using Transposed Deconvolution")

    if final:
        if args.channels_first:
            inputs = K.layers.Input(batch_shape=(1, num_chan_in, 128, 128),
                                    name="Images")
        else:
            inputs = K.layers.Input(batch_shape=(1, 128, 128, num_chan_in),
                                    name="Images")
    else:
        if args.channels_first:
            inputs = K.layers.Input((num_chan_in, 128, 128),
                                    name="Images")
        else:
            inputs = K.layers.Input((128, 128, num_chan_in),
                                    name="Images")


    # Convolution parameters
    params = dict(kernel_size=(3, 3), activation="relu",
                  padding="same", data_format=data_format,
                  kernel_initializer="he_uniform")

    # Transposed convolution parameters
    params_trans = dict(data_format=data_format,
                        kernel_size=(2, 2), strides=(2, 2),
                        padding="same")

    fms = 64

    conv1 = K.layers.Conv2D(name="conv1a", filters=fms, **params)(inputs)
    conv1 = K.layers.Conv2D(name="conv1b", filters=fms, **params)(conv1)
    pool1 = K.layers.MaxPooling2D(name="pool1", pool_size=(2, 2))(conv1)

    conv2 = K.layers.Conv2D(name="conv2a", filters=fms*2, **params)(pool1)
    conv2 = K.layers.Conv2D(name="conv2b", filters=fms*2, **params)(conv2)
    pool2 = K.layers.MaxPooling2D(name="pool2", pool_size=(2, 2))(conv2)

    conv3 = K.layers.Conv2D(name="conv3a", filters=fms*4, **params)(pool2)
    #conv3 = K.layers.Dropout(dropout)(conv3)
    conv3 = K.layers.Conv2D(name="conv3b", filters=fms*4, **params)(conv3)

    pool3 = K.layers.MaxPooling2D(name="pool3", pool_size=(2, 2))(conv3)

    conv4 = K.layers.Conv2D(name="conv4a", filters=fms*8, **params)(pool3)
    #conv4 = K.layers.Dropout(dropout)(conv4)
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
        loss = dice_coef_loss

        if args.trace:
            model.compile(optimizer=optimizer,
                          loss=loss,
                          metrics=metrics,
                          options=run_options, run_metadata=run_metadata)
        else:
            model.compile(optimizer=optimizer,
                          loss=loss,
                          metrics=metrics)

    return model


def train_and_predict(data_path, img_height, img_width, n_epoch,
                      input_no=3, output_no=3, mode=1):

    # print("-" * 40)
    # print("Loading and preprocessing train data...")
    # print("-" * 40)
    #
    # imgs_train, msks_train = load_data(data_path, "_train") #_norm")
    # # imgs_train, msks_train = update_channels(imgs_train, msks_train,
    # #                                          input_no, output_no, mode)
    #
    # print("-" * 40)
    # print("Loading and preprocessing test data...")
    # print("-" * 40)
    # imgs_test, msks_test = load_data(data_path, "_test") #_norm")
    # imgs_test, msks_test = update_channels(imgs_test, msks_test,
    #                                        input_no, output_no, mode)

    import h5py
    df = h5py.File(os.path.join(data_path, "decathlon_brats.h5"))

    imgs_train = df["imgs_train"]
    imgs_test = df["imgs_test"]
    msks_train = df["msks_train"]
    msks_test = df["msks_test"]

    print(imgs_train.shape)
    print(imgs_test.shape)
    print(msks_train.shape)
    print(msks_test.shape)
    
    print("-" * 30)
    print("Creating and compiling model...")
    print("-" * 30)

    model = unet_model(img_height, img_width, input_no, output_no)

    if (args.use_upsampling):
        model_fn = os.path.join(args.output_path, "unet_model_upsampling.hdf5")
    else:
        model_fn = os.path.join(args.output_path, "unet_model_transposed.hdf5")

    print("Writing model to '{}'".format(model_fn))

    model_checkpoint = K.callbacks.ModelCheckpoint(model_fn,
                                                   monitor="loss",
                                                   save_best_only=True)

    plateau_callback = K.callbacks.ReduceLROnPlateau(monitor="val_loss",
                factor=0.9, patience=3, verbose=1, min_lr=0.00001)

    directoryName = "unet_block{}_inter{}_intra{}".format(blocktime,
                                                          num_threads,
                                                          num_inter_op_threads)

    if (args.use_upsampling):
        tensorboard_checkpoint = K.callbacks.TensorBoard(
            log_dir=os.path.join(args.output_path,
                "keras_tensorboard_upsampling_batch{}/{}".format(
                batch_size, directoryName)),
            write_graph=True, write_images=True)
    else:
        tensorboard_checkpoint = K.callbacks.TensorBoard(
            log_dir=os.path.join(args.output_path,
                "keras_tensorboard_transposed_batch{}/{}".format(
                batch_size, directoryName)),
            write_graph=True, write_images=True)

    print("-" * 30)
    print("Fitting model...")
    print("-" * 30)

    history = K.callbacks.History()

    print("Batch size = {}".format(batch_size))
    if args.channels_first:  # Swap first and last axes on data
        imgs_train = np.swapaxes(imgs_train, 1, -1)
        msks_train = np.swapaxes(msks_train, 1, -1)
        imgs_test = np.swapaxes(imgs_test, 1, -1)
        msks_test = np.swapaxes(msks_test, 1, -1)

    history = model.fit(imgs_train, msks_train,
                        batch_size=batch_size,
                        epochs=n_epoch,
                        validation_data=(imgs_test, msks_test),
                        verbose=1,
                        callbacks=[model_checkpoint,
                                   tensorboard_checkpoint,
                                   plateau_callback])

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
            "dice_coef_loss": dice_coef_loss,
            "dice_coef": dice_coef})

    print("-" * 30)
    print("Predicting masks on test data...")
    print("-" * 30)
    msks_pred = model.predict(imgs_test, verbose=1)

    print("Saving predictions to file")
    if (args.use_upsampling):
        np.save("msks_pred_upsampling.npy", msks_pred)
    else:
        np.save("msks_pred_transposed.npy", msks_pred)

    start_inference = time.time()
    print("Evaluating model")
    scores = model.evaluate(
        imgs_test,
        msks_test,
        batch_size=batch_size,
        verbose=2)

    elapsed_time = time.time() - start_inference
    print("{} images in {:.2f} seconds = {:.3f} images per second inference".format(
        imgs_test.shape[0], elapsed_time, imgs_test.shape[0] / elapsed_time))
    print("Evaluation Scores", scores)

    # Save final model without custom loss and metrics
    # This way we can easily re-load it into Keras for inference
    model.save_weights(os.path.join(args.output_path,"weights.hdf5"))
    model = unet_model(img_height, img_width, input_no, output_no, final=True)  # Model without Dice and custom metrics
    model.load_weights(os.path.join(args.output_path,"weights.hdf5"))

    model_json = model.to_json()
    with open(os.path.join(args.output_path,"model.json"), "w") as json_file:
        json_file.write(model_json)

    model.save_weights(os.path.join(args.output_path,"weights.hdf5"))

    if (args.use_upsampling):
        model_fn = os.path.join(args.output_path, "unet_model_upsampling_for_inference.hdf5")
    else:
        model_fn = os.path.join(args.output_path, "unet_model_transposed_for_inference.hdf5")

    print("Writing final model (without custom Dice metrics) for inference to {}".format(model_fn))
    print("Please use that version for inference.")
    model.save(model_fn, include_optimizer=False)

    sess = K.backend.get_session()

    model = K.models.load_model(model_fn)

    K.backend.set_learning_phase(0)

    saver = tf.train.Saver()

    directory = os.path.join(args.output_path, "tf_checkpoint")
    try:
        os.stat(directory)
    except:
        os.mkdir(directory)

    save_path = saver.save(sess, os.path.join(directory, "unet_model"))
    print("Checkpoint saved in path: {}".format(save_path))


if __name__ == "__main__":

    os.system("lscpu")

    import datetime
    print("Started script on {}".format(datetime.datetime.now()))

    print("args = {}".format(args))
    os.system("uname -a")
    print("TensorFlow version: {}".format(tf.__version__))
    start_time = time.time()

    train_and_predict(args.data_path, settings.IMG_HEIGHT,
                      settings.IMG_WIDTH,
                      args.epochs, settings.NUM_IN_CHANNELS,
                      settings.NUM_OUT_CHANNELS,
                      settings.MODE)

    print(
        "Total time elapsed for program = {} seconds".format(
            time.time() -
            start_time))
    print("Stopped script on {}".format(datetime.datetime.now()))
