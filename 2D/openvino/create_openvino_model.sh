#!/bin/bash
#
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
