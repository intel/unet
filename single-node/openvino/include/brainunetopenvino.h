
/* Copyright (c) 2019 Intel Corporation
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
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

#define M_IE_PLUGIN_PATH                                                       \
  "/opt/intel/computer_vision_sdk/deployment_tools/inference_engine/lib/"      \
  "ubuntu_16.04/intel64"


#endif // OPENVINO_H

#ifndef BRAINUNETOPENVINO_H
#define BRAINUNETOPENVINO_H

#define MODEL_DIR std::string("../models/")
#define MODEL_FILENAME std::string("saved_model")
#define DATA_FILENAME std::string("../data/validation_data.npz")

#include "../src/cnpy/cnpy.h"

class BrainUnetOpenVino {
public:
  double *loaded_data_masks;
  double *loaded_data;
  void loadNumpyData(cnpy::NpyArray &arr, cnpy::NpyArray &arr_msks);
  void makeInference(int img_index, InferenceEngine::TargetDevice targetDevice,
                     cnpy::NpyArray &arr, cnpy::NpyArray &arr_msks);
};

#endif // BRAINUNETOPENVINO_H
