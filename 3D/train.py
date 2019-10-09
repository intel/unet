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

from dataloader import DataGenerator
from model import unet
import datetime
import os
import tensorflow as tf
from argparser import args
if args.keras_api:
    import keras as K
else:
    from tensorflow import keras as K

print("Args = {}".format(args))

CHANNELS_LAST = True

if CHANNELS_LAST:
   print("Data format = channels_last")
else:
   print("Data format = channels_first")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Get rid of the AVX, SSE warnings
os.environ["OMP_NUM_THREADS"] = str(args.intraop_threads)
os.environ["KMP_BLOCKTIME"] = str(args.blocktime)
os.environ["KMP_AFFINITY"] = "granularity=thread,compact"

# os.system("lscpu")
start_time = datetime.datetime.now()
print("Started script on {}".format(start_time))

#os.system("uname -a")
print("TensorFlow version: {}".format(tf.__version__))
print("Intel MKL-DNN is enabled = {}".format(tf.pywrap_tensorflow.IsMklEnabled()))

print("Keras API version: {}".format(K.__version__))

# Optimize CPU threads for TensorFlow
CONFIG = tf.ConfigProto(
    inter_op_parallelism_threads=args.interop_threads,
    intra_op_parallelism_threads=args.intraop_threads)

SESS = tf.Session(config=CONFIG)

K.backend.set_session(SESS)

unet_model = unet(use_upsampling=args.use_upsampling,
                  learning_rate=args.lr,
                  n_cl_in=args.number_input_channels,
                  n_cl_out=1,  # single channel (greyscale)
                  feature_maps = args.featuremaps,
                  dropout=0.2,
                  print_summary=args.print_model,
                  channels_last = CHANNELS_LAST)  # channels first or last

unet_model.model.compile(optimizer=unet_model.optimizer,
              loss=unet_model.loss,
              metrics=unet_model.metrics)

# Save best model to hdf5 file
saved_model_directory = os.path.dirname(args.saved_model)
try:
    os.stat(saved_model_directory)
except:
    os.mkdir(saved_model_directory)

# If there is a current saved file, then load weights and start from
# there.
if os.path.isfile(args.saved_model):
    unet_model.model.load_weights(args.saved_model)

checkpoint = K.callbacks.ModelCheckpoint(args.saved_model,
                                         verbose=1,
                                         save_best_only=True)

# TensorBoard
currentDT = datetime.datetime.now()
tb_logs = K.callbacks.TensorBoard(log_dir=os.path.join(
    saved_model_directory, "tensorboard_logs", currentDT.strftime("%Y/%m/%d-%H:%M:%S")), update_freq="batch")

# Keep reducing learning rate if we get to plateau
reduce_lr = K.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2,
                                          patience=5, min_lr=0.0001)

callbacks = [checkpoint, tb_logs, reduce_lr]

training_data_params = {"dim": (args.patch_height, args.patch_width, args.patch_depth),
                        "batch_size": args.bz,
                        "n_in_channels": args.number_input_channels,
                        "n_out_channels": 1,
                        "train_test_split": args.train_test_split,
                        "validate_test_split": args.validate_test_split,
                        "augment": True,
                        "shuffle": True,
                        "seed": args.random_seed}

training_generator = DataGenerator("train", args.data_path,
                                   **training_data_params)
training_generator.print_info()

validation_data_params = {"dim": (args.patch_height, args.patch_width, args.patch_depth),
                          "batch_size": 1,
                          "n_in_channels": args.number_input_channels,
                          "n_out_channels": 1,
                          "train_test_split": args.train_test_split,
                          "validate_test_split": args.validate_test_split,
                          "augment": False,
                          "shuffle": False,
                          "seed": args.random_seed}
validation_generator = DataGenerator("validate", args.data_path,
                                     **validation_data_params)
validation_generator.print_info()

# Fit the model
"""
Keras Data Pipeline using Sequence generator
https://www.tensorflow.org/api_docs/python/tf/keras/utils/Sequence

The sequence generator allows for Keras to load batches at runtime.
It's very useful in the case when your entire dataset won't fit into
memory. The Keras sequence will load one batch at a time to
feed to the model. You can specify pre-fetching of batches to
make sure that an additional batch is in memory when the previous
batch finishes processing.

max_queue_size : Specifies how many batches will be prepared (pre-fetched)
in the queue. Does not indicate multiple generator instances.

workers, use_multiprocessing: Generates multiple generator instances.

num_data_loaders is defined in argparser.py
"""

unet_model.model.fit_generator(training_generator,
                    epochs=args.epochs, verbose=1,
                    validation_data=validation_generator,
                    callbacks=callbacks,
                    max_queue_size=args.num_prefetched_batches,
                    workers=args.num_data_loaders,
                    use_multiprocessing=True)  #False)  # True seems to cause fork issue

# Evaluate final model on test holdout set
testing_generator = DataGenerator("test", args.data_path,
                                     **validation_data_params)
testing_generator.print_info()

scores = unet_model.model.evaluate_generator(testing_generator, verbose=1)
print("Final model metrics on test dataset:")
for idx, name in enumerate(unet_model.model.metrics_names):
    print("{} \t= {}".format(name, scores[idx]))

stop_time = datetime.datetime.now()
print("Started script on {}".format(start_time))
print("Stopped script on {}".format(stop_time))
print("\nTotal time for training model = {}".format(stop_time - start_time))
