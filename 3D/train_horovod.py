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

import horovod.keras as hvd
hvd.init()

from argparser import args

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Get rid of the AVX, SSE warnings
os.environ["OMP_NUM_THREADS"] = str(args.intraop_threads)
os.environ["KMP_BLOCKTIME"] = str(args.blocktime)
os.environ["KMP_AFFINITY"] = "granularity=thread,compact,1,0"

if (hvd.rank() == 0): # Only print on worker 0
    print_summary = args.print_model
    verbose = 1
    os.system("lscpu")
    os.system("uname -a")
    print("TensorFlow version: {}".format(tf.__version__))
    print("Intel MKL-DNN is enabled = {}".format(tf.pywrap_tensorflow.IsMklEnabled()))
    print("Keras API version: {}".format(K.__version__))

else:  # Don't print on workers > 0
    print_summary = 0
    verbose = 0
    # Horovod needs to have every worker do the same amount of work.
    # Otherwise it will complain at the end of the epoch when
    # worker 0 takes more time than the others to do validation,
    # logging, and model checkpointing.
    # We'll save the worker logs and models separately but only
    # use the logs/saved model from worker 0.
    args.saved_model = "./worker{}/3d_unet_decathlon.hdf5".format(hvd.rank())

# Optimize CPU threads for TensorFlow
config = tf.ConfigProto(
    inter_op_parallelism_threads=args.interop_threads,
    intra_op_parallelism_threads=args.intraop_threads)

sess = tf.Session(config=config)

K.backend.set_session(sess)


model, opt = unet_3d(use_upsampling=args.use_upsampling,
                n_cl_in=args.number_input_channels,
                learning_rate=args.lr*hvd.size(),
                n_cl_out=1,  # single channel (greyscale)
                dropout=0.2,
                print_summary=print_summary)

opt = hvd.DistributedOptimizer(opt)

model.compile(optimizer=opt,
              #loss=[combined_dice_ce_loss],
              loss=[dice_coef_loss],
              metrics=[dice_coef, "accuracy",
                       sensitivity, specificity])

if hvd.rank() == 0:
    start_time = datetime.datetime.now()
    print("Started script on {}".format(start_time))

# Save best model to hdf5 file
saved_model_directory = os.path.dirname(args.saved_model)
try:
    os.stat(saved_model_directory)
except:
    os.mkdir(saved_model_directory)

# if os.path.isfile(args.saved_model):
#     model.load_weights(args.saved_model)

checkpoint = K.callbacks.ModelCheckpoint(args.saved_model,
                                         verbose=verbose,
                                         save_best_only=True)

# TensorBoard
if (hvd.rank() == 0):
    tb_logs = K.callbacks.TensorBoard(log_dir=os.path.join(
        saved_model_directory, "tensorboard_logs"), update_freq="batch")
else:
    tb_logs = K.callbacks.TensorBoard(log_dir=os.path.join(
        saved_model_directory, "tensorboard_logs_worker{}".format(hvd.rank())),
        update_freq="batch")

# NOTE:
# Horovod talks about having callbacks for rank 0 and callbacks
# for other ranks. For example, they recommend only doing checkpoints
# and tensorboard on rank 0. However, if there is a signficant time
# to execute tensorboard update or checkpoint update, then
# this might cause an issue with rank 0 not returning in time.
# My thought is that all ranks need to have essentially the same
# time taken for each rank.
callbacks = [
    # Horovod: broadcast initial variable states from
    # rank 0 to all other processes.
    # This is necessary to ensure consistent initialization
    # of all workers when
    # training is started with random weights or
    # restored from a checkpoint.
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),

    # Horovod: average metrics among workers at the end of every epoch.
    #
    # Note: This callback must be in the list before the ReduceLROnPlateau,
    # TensorBoard or other metrics-based callbacks.
    hvd.callbacks.MetricAverageCallback(),

    # Horovod: using `lr = 1.0 * hvd.size()` from the very
    # beginning leads to worse final
    # accuracy. Scale the learning rate
    # `lr = 1.0` ---> `lr = 1.0 * hvd.size()` during
    # the first five epochs. See https://arxiv.org/abs/1706.02677
    # for details.
    hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=3, verbose=verbose),

    # Reduce the learning rate if training plateaus.
    K.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.6,
                                  verbose=verbose,
                                  patience=5, min_lr=0.0001),
    tb_logs,  # we need this here otherwise tensorboard delays rank 0
    checkpoint
]

# Run the script  "load_brats_images.py" to generate these Numpy data files
#imgs_test = np.load(os.path.join(sys.path[0],"imgs_test_3d.npy"))
#msks_test = np.load(os.path.join(sys.path[0],"msks_test_3d.npy"))

seed = hvd.rank()  # Make sure each worker gets different random seed
training_data_params = {"dim": (args.patch_height, args.patch_width, args.patch_depth),
                        "batch_size": args.bz,
                        "n_in_channels": args.number_input_channels,
                        "n_out_channels": 1,
                        "train_test_split": args.train_test_split,
                        "augment": True,
                        "shuffle": True,
                        "seed": seed}

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
# Do at least 3 steps for training and validation
steps_per_epoch = max(3, training_generator.get_length()//(args.bz*hvd.size()))
validation_steps = max(3,3*training_generator.get_length()//(args.bz*hvd.size()))

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
                    steps_per_epoch=steps_per_epoch,
                    epochs=args.epochs, verbose=verbose,
                    validation_data=validation_generator,
                    #validation_steps=validation_steps,
                    callbacks=callbacks,
                    max_queue_size=args.num_prefetched_batches,
                    workers=args.num_data_loaders,
                    use_multiprocessing=False) #True)

if hvd.rank() == 0:
    stop_time = datetime.datetime.now()
    print("Started script on {}".format(start_time))
    print("Stopped script on {}".format(stop_time))
    print("\nTotal time = {:,.3f} seconds".format(
        stop_time - start_time))
