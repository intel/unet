#!/usr/bin/env bash
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

# Setup OpenVINO environment
source /opt/intel/computer_vision_sdk/bin/setupvars.sh

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
