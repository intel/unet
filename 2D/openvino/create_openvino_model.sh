#!/bin/bash
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

FROZEN_MODEL="../frozen_model/saved_model_frozen.pb"

if [ ! -f $FROZEN_MODEL ]; then
   echo "File $FROZEN_MODEL doesn't exist."
   echo "Please make sure you have a trained model and then run the script: "
   echo "'python helper_scripts/convert_keras_to_tensorflow_serving_model.py --input_filename output/unet_model_for_decathlon.hdf5'"
   echo "The directions at the end of the script will show you the commands to"
   echo "create a frozen model."
   exit 1
fi

# For CPU
python ${INTEL_OPENVINO_DIR}/deployment_tools/model_optimizer/mo_tf.py \
      --input_model $FROZEN_MODEL \
      --input_shape=[1,144,144,4] \
      --data_type FP32  \
      --output_dir models/FP32  \
      --model_name saved_model

# For NCS
python ${INTEL_OPENVINO_DIR}/deployment_tools/model_optimizer/mo_tf.py \
      --input_model ../frozen_model/saved_model_frozen.pb \
      --input_shape=[1,144,144,4] \
      --data_type FP16  \
      --output_dir models/FP16  \
      --model_name saved_model
