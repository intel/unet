#!/bin/sh
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

if [ -d "openvino" ]; then
  cd openvino
fi

if [ -f /etc/redhat-release ]; then
  # CentOS
  OPENVINO_LIB=${INTEL_CVSDK_DIR}/inference_engine/lib/centos_7.4/intel64/
fi

if [ -f /etc/lsb-release ]; then
  # Ubuntu
  OPENVINO_LIB=${INTEL_CVSDK_DIR}/inference_engine/lib/ubuntu_16.04/intel64/
fi

python inference_openvino.py -l ${OPENVINO_LIB}/libcpu_extension_avx2.so \
       --plot --stats -d MYRIAD
