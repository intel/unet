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

from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io
import shutil

print("Args = {}".format(args))

CHANNELS_LAST = True

if CHANNELS_LAST:
   print("Data format = channels_last")
else:
   print("Data format = channels_first")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Get rid of the AVX, SSE warnings
os.environ["OMP_NUM_THREADS"] = str(args.intraop_threads)
os.environ["KMP_BLOCKTIME"] = str(args.blocktime)

# If hyperthreading is enabled, then use
os.environ["KMP_AFFINITY"] = "granularity=thread,compact,1,0"

# If hyperthreading is NOT enabled, then use
#os.environ["KMP_AFFINITY"] = "granularity=thread,compact"

# os.system("lscpu")
start_time = datetime.datetime.now()
print("Started script on {}".format(start_time))

#os.system("uname -a")
print("TensorFlow version: {}".format(tf.__version__))
from tensorflow.python import pywrap_tensorflow
print("Intel MKL-DNN is enabled = {}".format(pywrap_tensorflow.IsMklEnabled()))

print("Keras API version: {}".format(K.__version__))

def save_frozen_model(model_filename, input_shape):
    """
    Save frozen TensorFlow formatted model protobuf
    """
    model = K.models.load_model(model_filename, compile=None)

    # Change filename to protobuf extension
    base = os.path.basename(model_filename)
    output_model = os.path.splitext(base)[0] + ".pb"

    # Set Keras to inference
    K.backend._LEARNING_PHASE = tf.constant(0)
    K.backend.set_learning_phase(False)
    K.backend.set_learning_phase(0)
    K.backend.set_image_data_format("channels_last")

    num_output = len(model.outputs)
    predictions = [None] * num_output
    prediction_node_names = [None] * num_output

    for i in range(num_output):
        prediction_node_names[i] = "output_node" + str(i)
        predictions[i] = tf.identity(model.outputs[i],
                name=prediction_node_names[i])

    sess = K.backend.get_session()

    constant_graph = graph_util.convert_variables_to_constants(sess,
                     sess.graph.as_graph_def(), prediction_node_names)
    infer_graph = graph_util.remove_training_nodes(constant_graph)

    # Write protobuf of frozen model
    frozen_dir = "./tf_protobuf/"
    shutil.rmtree(frozen_dir, ignore_errors=True) # Remove existing directory
    graph_io.write_graph(infer_graph, frozen_dir, output_model, as_text=False)

    pb_filename = os.path.join(frozen_dir, output_model)
    print("\n\nFrozen TensorFlow model written to: {}".format(pb_filename))
    print("Convert this to OpenVINO by running:\n")
    print("source /opt/intel/openvino/bin/setupvars.sh")
    print("python $INTEL_OPENVINO_DIR/deployment_tools/model_optimizer/mo_tf.py \\")
    print("       --input_model {} \\".format(pb_filename))

    shape_string = "[1"
    for idx in range(len(input_shape[1:])):
        shape_string += ",{}".format(input_shape[idx+1])
    shape_string += "]"

    print("       --input_shape {} \\".format(shape_string))
    print("       --output_dir openvino_models/FP32/ \\")
    print("       --data_type FP32\n\n")



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
                    use_multiprocessing=False)  #False)  # True seems to cause fork issue


# Evaluate final model on test holdout set
testing_generator = DataGenerator("test", args.data_path,
                                     **validation_data_params)
testing_generator.print_info()

# Load the best model
print("Loading the best model: {}".format(args.saved_model))
unet_model.model.load_weights(args.saved_model)
scores = unet_model.model.evaluate_generator(testing_generator, verbose=1)

print("Final model metrics on test dataset:")
for idx, name in enumerate(unet_model.model.metrics_names):
    print("{} \t= {}".format(name, scores[idx]))

# Save a frozen version of the model for use in OpenVINO
save_frozen_model(args.saved_model,
                 [1, args.patch_height, args.patch_width, args.patch_depth,
                 args.number_input_channels])

stop_time = datetime.datetime.now()
print("Started script on {}".format(start_time))
print("Stopped script on {}".format(stop_time))
print("\nTotal time for training model = {}".format(stop_time - start_time))
