#!/bin/sh
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

# Please see the full instructions at https://hub.docker.com/r/openvino/ubuntu18_dev_no_samples
# If you need Myriad* (NCS2) accelerator only, run image with the following command:
# docker run -it --device-cgroup-rule='c 189:* rmw' -v /dev/bus/usb:/dev/bus/usb --rm openvino/ubuntu18_dev_no_samples:latest 

# For CPU only, run this command:
docker run -it --rm openvino/ubuntu18_dev_no_samples:latest 
