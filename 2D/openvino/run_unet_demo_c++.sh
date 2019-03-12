#!/bin/sh
# ----------------------------------------------------------------------------
# Copyright 2019 Intel
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

if [ -f /etc/redhat-release ]; then
  # CentOS
  PLUGIN_LIB=${INTEL_CVSDK_DIR}/inference_engine/lib/centos_7.4/intel64/
fi

if [ -f /etc/lsb-release ]; then
  # Ubuntu
  PLUGIN_LIB=${INTEL_CVSDK_DIR}/inference_engine/lib/ubuntu_16.04/intel64/
fi

cd build
./UnetOpenVino -d ${PLUGIN_LIB}
