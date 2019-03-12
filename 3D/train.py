#!/usr/bin/python

# ----------------------------------------------------------------------------
# Copyright 2019 Intel
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

from imports import *  # All of the common imports

import os
import datetime

from model import *
from dataloader import DataGenerator
from argparser import args

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Get rid of the AVX, SSE warnings
os.environ["OMP_NUM_THREADS"] = str(args.intraop_threads)
os.environ["KMP_BLOCKTIME"] = str(args.blocktime)
os.environ["KMP_AFFINITY"] = "granularity=thread,compact,1,0"

os.system("lscpu")
start_time = datetime.datetime.now()
print("Started script on {}".format(start_time))

os.system("uname -a")
print("TensorFlow version: {}".format(tf.__version__))
print("Intel MKL-DNN is enabled = {}".format(tf.pywrap_tensorflow.IsMklEnabled()))

print("Keras API version: {}".format(K.__version__))

# Optimize CPU threads for TensorFlow
config = tf.ConfigProto(
    inter_op_parallelism_threads=args.interop_threads,
    intra_op_parallelism_threads=args.intraop_threads)

sess = tf.Session(config=config)

K.backend.set_session(sess)

print_summary = args.print_model

model, opt = unet_3d(use_upsampling=args.use_upsampling,
                learning_rate=args.lr,
                n_cl_in=args.number_input_channels,
                n_cl_out=1,  # single channel (greyscale)
                dropout=0.2,
                print_summary=print_summary)

model.compile(optimizer=opt,
              #loss=[combined_dice_ce_loss],
              loss=[dice_coef_loss],
              metrics=[dice_coef, "accuracy",
                       sensitivity, specificity])

# Save best model to hdf5 file
saved_model_directory = os.path.dirname(args.saved_model)
try:
    os.stat(saved_model_directory)
except:
    os.mkdir(saved_model_directory)

# If there is a current saved file, then load weights and start from
# there.
if os.path.isfile(args.saved_model):
    model.load_weights(args.saved_model)

checkpoint = K.callbacks.ModelCheckpoint(args.saved_model,
                                         verbose=1,
                                         save_best_only=True)

# TensorBoard
tb_logs = K.callbacks.TensorBoard(log_dir=os.path.join(
    saved_model_directory, "tensorboard_logs"), update_freq="batch")

# Keep reducing learning rate if we get to plateau
reduce_lr = K.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2,
                              patience=5, min_lr=0.0001)

callbacks = [checkpoint, tb_logs, reduce_lr]

training_data_params = {"dim": (args.patch_height, args.patch_width, args.patch_depth),
                        "batch_size": args.bz,
                        "n_in_channels": args.number_input_channels,
                        "n_out_channels": 1,
                        "train_test_split": args.train_test_split,
                        "augment": True,
                        "shuffle": True,
                        "seed": args.random_seed}

training_generator = DataGenerator(True, args.data_path,
                                   **training_data_params)

validation_data_params = {"dim": (args.patch_height, args.patch_width, args.patch_depth),
                          "batch_size": 1,
                          "n_in_channels": args.number_input_channels,
                          "n_out_channels": 1,
                          "train_test_split": args.train_test_split,
                          "augment": False,
                          "shuffle": False,
                          "seed": args.random_seed}
validation_generator = DataGenerator(False, args.data_path,
                                     **validation_data_params)

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

model.fit_generator(training_generator,
                    epochs=args.epochs, verbose=1,
                    validation_data=validation_generator,
                    callbacks=callbacks,
                    max_queue_size=args.num_prefetched_batches,
                    workers=args.num_data_loaders,
                    use_multiprocessing=False)  # True seems to cause fork issue

stop_time = datetime.datetime.now()
print("Started script on {}".format(start_time))
print("Stopped script on {}".format(stop_time))
print("\nTotal time for training model = {}".format(stop_time - start_time))
