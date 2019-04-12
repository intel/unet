#!/bin/bash
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
