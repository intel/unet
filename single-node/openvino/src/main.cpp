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

#include "../include/brainunetopenvino.h"

int IMAGE_FILE_INDEX = 1;
InferenceEngine::TargetDevice CPU_PLUGIN = TargetDevice::eCPU;
int image_file_index;
InferenceEngine::TargetDevice plugin_name;

void printUsage(const std::string &app_name = std::string()) {
  std::cout << std::endl
            << "To run application in default configuration use:" << std::endl;
  if (!app_name.empty())
    std::cout << app_name << " ";
  std::cout << "-p" << std::endl << std::endl;
  std::cout << "Usage:" << std::endl;
  if (!app_name.empty())
    std::cout << app_name << std::endl;
  std::cout << "     [-h] [--h] [-help] [--help] : display this usage page. No "
               "other commands accepted."
            << std::endl;
  std::cout
      << "     [-p MYRIAD | myriad | CPU | cpu] : select plugin (defaults is "
         "CPU)"
      << std::endl;
  std::cout
      << "     [-f filename] : specify the image number to be detected and "
         "recognized"
      << std::endl
      << std::endl;
}

// Parses the command line to find the plugin to use.
void parseArgs(int argc, char *argv[]) {

  try {

    // Specify the default file name in case users don't pass it
    image_file_index = IMAGE_FILE_INDEX;
    plugin_name = CPU_PLUGIN;

    // Start parsing out parameters
    for (int i = 1; i < argc; i++) {
      std::string arg = argv[i];

      if (arg == "-h") { // users want to see help
        printUsage(argv[0]);
        return;
      }

      if (arg == "-i") {
        if (i + 1 < argc) {
          i++;
          image_file_index = std::stoi(argv[i]);
        } else
          throw i; // invalid parameter
      }            // end if (-i)

      if (arg == "-p") { // if users pass this option, mean they want test this
                         // particular interface

        if (i + 1 < argc) {
          i++;
          arg = argv[i];
          if (arg == "myriad" || arg == "MYRIAD")
            plugin_name = TargetDevice::eMYRIAD;
          if (arg == "cpu" || arg == "CPU")
            plugin_name = TargetDevice::eCPU;
        }
      }
    }
  } catch (const std::exception& e) {
    std::cout << e.what() << std::endl;
    std::cout << "Use parameter -h for a list of valid parameters."
              << std::endl;

    throw "ERROR parsing command line.";
  }
}

int main(int argc, char *argv[]) {
  std::cout << "Starting program" << std::endl;
  BrainUnetOpenVino brainunetobj;

  try {
    parseArgs(argc, argv); // request users for default
    brainunetobj.makeInference(image_file_index, plugin_name);

  } catch (const std::exception& e) {
    std::cout << e.what() << std::endl;
  }

  return 0;
}
