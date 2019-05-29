#!/bin/bash
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

if [ -d "openvino" ]; then
  cd openvino
fi

OPENVINO_LIB=${INTEL_OPENVINO_DIR}/inference_engine/lib/intel64/

python inference_openvino.py -l ${OPENVINO_LIB}/libcpu_extension_avx512.so \
       --plot
