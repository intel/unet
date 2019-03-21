/* 
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
