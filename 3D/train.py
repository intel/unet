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
import tensorflow as tf   # TensorFlow 2
from tensorflow import keras as K

import os
import datetime

from argparser import args
from dataloader import DatasetGenerator
from model import dice_coef, soft_dice_coef, dice_loss, unet_3d


def test_intel_tensorflow():
    """
    Check if Intel version of TensorFlow is installed
    """
    import tensorflow as tf

    print("We are using Tensorflow version {}".format(tf.__version__))

    major_version = int(tf.__version__.split(".")[0])
    if major_version >= 2:
        from tensorflow.python import _pywrap_util_port
        print("Intel-optimizations (DNNL) enabled:",
              _pywrap_util_port.IsMklEnabled())
    else:
        print("Intel-optimizations (DNNL) enabled:",
              tf.pywrap_tensorflow.IsMklEnabled())

print(args)
test_intel_tensorflow()  # Prints if Intel-optimized TensorFlow is used.

"""
crop_dim = Dimensions to crop the input tensor
"""
crop_dim = (args.tile_height, args.tile_width,
            args.tile_depth, args.number_input_channels)

"""
1. Load the dataset
"""
brats_data = DatasetGenerator(crop_dim,
                              data_path=args.data_path,
                              batch_size=args.batch_size,
                              train_test_split=args.train_test_split,
                              validate_test_split=args.validate_test_split,
                              number_output_classes=args.number_output_classes,
                              random_seed=args.random_seed)

brats_data.print_info()  # Print dataset information

"""
2. Create the TensorFlow model
"""
model = unet_3d(input_dim=crop_dim, filters=args.filters,
                number_output_classes=args.number_output_classes,
                use_upsampling=args.use_upsampling,
                concat_axis=-1, model_name=args.saved_model_name)

local_opt = K.optimizers.Adam()
model.compile(loss=dice_loss,
              metrics=[dice_coef, soft_dice_coef],
              optimizer=local_opt)

checkpoint = K.callbacks.ModelCheckpoint(args.saved_model_name,
                                         verbose=1,
                                         save_best_only=True)

# TensorBoard
logs_dir = os.path.join(
    "logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tb_logs = K.callbacks.TensorBoard(log_dir=logs_dir)

callbacks = [checkpoint, tb_logs]

"""
3. Train the model
"""
steps_per_epoch = brats_data.num_train // args.batch_size
model.fit(brats_data.get_train(), epochs=args.epochs,
          steps_per_epoch=steps_per_epoch,
          validation_data=brats_data.get_validate(),
          callbacks=callbacks,
          verbose=1)

"""
4. Load best model on validation dataset and run on the test
dataset to show generalizability
"""
best_model = K.models.load_model(args.saved_model_name,
                                 custom_objects={"dice_loss": dice_loss,
                                                 "dice_coef": dice_coef,
                                                 "soft_dice_coef": soft_dice_coef})

print("\n\nEvaluating best model on the testing dataset.")
print("=============================================")
loss, dice_coef, soft_dice_coef = best_model.evaluate(brats_data.get_test())

print("Average Dice Coefficient on testing dataset = {:.4f}".format(dice_coef))

"""
5. Save the best model without the custom objects (dice, etc.)
   NOTE: You should be able to do .load_model(compile=False), but this
   appears to currently be broken in TF2. To compensate, we're
   just going to re-compile the model without the custom objects and
   save as a new model (with suffix "_final")
"""
final_model_name = args.saved_model_name + "_final"
best_model.compile(loss="binary_crossentropy", metrics=["accuracy"],
                   optimizer="adam")
K.models.save_model(best_model, final_model_name,
                    include_optimizer=False)

"""
Converting the model to OpenVINO
"""
print("\n\nConvert the TensorFlow model to OpenVINO by running:\n")
print("source /opt/intel/openvino/bin/setupvars.sh")
print("python $INTEL_OPENVINO_DIR/deployment_tools/model_optimizer/mo_tf.py \\")
print("       --saved_model_dir {} \\".format(final_model_name))
print("       --model_name {} \\".format(args.saved_model_name))
print("       --batch 1  \\")
print("       --output_dir {} \\".format(os.path.join("openvino_models", "FP32")))
print("       --data_type FP32\n\n")

