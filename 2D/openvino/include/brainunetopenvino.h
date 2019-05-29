/* 
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
*/

#ifndef OPENVINO_H
#define OPENVINO_H

#include <chrono>
#include <fstream>
#include <functional>
#include <gflags/gflags.h>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <time.h>
#include <vector>

#include <inference_engine.hpp>
#include <opencv2/opencv.hpp>

using namespace InferenceEngine;

#endif // OPENVINO_H

#ifndef BRAINUNETOPENVINO_H
#define BRAINUNETOPENVINO_H

#define MODEL_DIR std::string("../models/")
#define MODEL_FILENAME std::string("saved_model")
#define DATA_FILENAME std::string("../data/validation_data.npz")

#define M_IE_PLUGIN "/opt/intel/computer_vision_sdk/" \
        "inference_engine/lib/ubuntu_16.04/intel64/"

#include "../src/cnpy/cnpy.h"

class BrainUnetOpenVino {
// Class to keep the OpenVINO model data and functions
private:

        struct array_dims {
                size_t NH;
                size_t NW;
                size_t NC;
                size_t NN;
        };

public:
        array_dims input_shape; // Input matrix shape
        array_dims output_shape;  // Output matrix shape
        std::vector<double> img_data;  // Input data
        std::vector<double> msk_data;  // Ground truth data
        Blob::Ptr prediction_blob;     // Model prediction data
        size_t img_id; // Image id

        std::string M_IE_PLUGIN_PATH;  // Plugin for OpenVINO
        std::string PRECISION;    // Precision for model
        std::string CHANNEL_FORMAT = std::string("NHWC"); // NCHW or NHWC

        void loadData(int img_index);
        void doInference(InferenceEngine::TargetDevice targetDevice);
        void plotResults();
};

#endif // BRAINUNETOPENVINO_H
