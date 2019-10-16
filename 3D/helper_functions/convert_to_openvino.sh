#!/usr/bin/env bash
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

# Setup OpenVINO environment
source /opt/intel/openvino/bin/setupvars.sh

# Convert model to TensorFlow Serving protobug format
python convert_keras_to_tensorflow_serving_model.py

echo "Model converted to TensorFlow Serving"

# Freeze TensorFlow Serving model
mkdir -p frozen_model
python ${CONDA_PREFIX}/lib/python3.6/site-packages/tensorflow/python/tools/freeze_graph.py --input_saved_model_dir saved_3dunet_model_protobuf --output_node_names PredictionMask/Sigmoid --output_graph frozen_model/saved_model_frozen.pb

echo "Model frozen"

# Convert to Intel OpenVINO intermediate representation
python ${INTEL_CVSDK_DIR}/deployment_tools/model_optimizer/mo_tf.py --input_model frozen_model/saved_model_frozen.pb --input_shape=[1,144,144,144,1] --data_type FP32  --output_dir openvino_models/FP32  --model_name 3d_unet_decathlon

echo "OpenVINO model saved to openvino_models/FP32 directory"
