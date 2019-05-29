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

source /opt/intel/openvino/bin/setupvars.sh

# Clone CNPY project to read numpy files in C++
cd src
git clone https://github.com/rogersce/cnpy.git
cd cnpy
mkdir -p build
cd build
cmake ..
make
sudo make install

# Build C++ project
cd ../../../project
qmake    # Using Qt make system to generate the Makefile
make clean  # Clean existing file
make -j8 # Make C++ executable
