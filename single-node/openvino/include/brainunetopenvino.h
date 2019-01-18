
#include <gflags/gflags.h>
#include <iostream>
#include <fstream>
#include <memory>
#include <functional>
#include <map>
#include <random>
#include <string>
#include <vector>
#include <time.h>
#include <chrono>
#include <limits>
#include <iomanip>
#include "../src/cnpy/cnpy.h"


#include <inference_engine.hpp>
#include <opencv2/opencv.hpp>




#ifndef BRAINUNETOPENVINO_H
#define BRAINUNETOPENVINO_H
using namespace InferenceEngine;


class BrainUnetOpenVino
{
public:
   double* loaded_data_masks;
   double* loaded_data;
   void loadNumpyData(cnpy::NpyArray &arr, cnpy::NpyArray &arr_msks);
   void makeInference(int img_index, InferenceEngine::TargetDevice targetDevice, cnpy::NpyArray &arr, cnpy::NpyArray &arr_msks);

};

#endif // BRAINUNETOPENVINO_H
