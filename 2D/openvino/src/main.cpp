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
#
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
  std::cout << "     [-h] : display this usage page. No "
               "other commands accepted."
            << std::endl;
  std::cout
      << "     [-p MYRIAD | myriad | CPU | cpu] : select plugin (default is "
         "CPU)"
      << std::endl;
  std::cout << "     [-d directory_for_openvino_plugins] "
            << " (default is the environment variable OPENVINO_PLUGIN_PATH)"
            << std::endl;
  std::cout
      << "     [-i image_number] : specify the image number to be detected and "
         "recognized (test data has 13 times (0-12))"
      << std::endl
      << std::endl;
}

// Parses the command line to find the plugin to use.
int parseArgs(int argc, char *argv[], BrainUnetOpenVino &brainunetobj) {

  try {

    // Specify the default file name in case users don't pass it
    image_file_index = IMAGE_FILE_INDEX;
    plugin_name = CPU_PLUGIN;

    // Check to see if environment variable is set.
    // If so, then use it instead of the default
    if (std::getenv("OPENVINO_PLUGIN_PATH")) {
      brainunetobj.M_IE_PLUGIN_PATH = std::getenv("OPENVINO_PLUGIN_PATH");
    } else {
      brainunetobj.M_IE_PLUGIN_PATH = std::string(M_IE_PLUGIN);
    };

    // Start parsing out parameters
    for (int i = 1; i < argc; i++) {
      std::string arg = argv[i];

      if (arg == "-h") { // users want to see help
        printUsage(argv[0]);
        return -1;
      }

      if (arg == "-d") {
        if (i + 1 < argc) {
          i++;
          brainunetobj.M_IE_PLUGIN_PATH = argv[i];
        } else
          throw i; // invalid parameter
      }            // end if (-d)

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
  } catch (const std::exception &e) {
    std::cout << e.what() << std::endl;
    std::cout << "Use parameter -h for a list of valid parameters."
              << std::endl;

    throw "ERROR parsing command line.";
  }

  return 0;

}

int main(int argc, char *argv[]) {

  BrainUnetOpenVino brainunetobj;

  try {
    if (parseArgs(argc, argv, brainunetobj) != 0) {
      return -1;
    }; // request users for default

    brainunetobj.loadData(image_file_index);
    brainunetobj.doInference(plugin_name);
    brainunetobj.plotResults();

  } catch (const std::exception &e) {
    std::cout << e.what() << std::endl;
  }

  return 0;
}
