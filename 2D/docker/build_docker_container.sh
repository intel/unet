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

# Copy the model and scripts from the OpenVINO directory here
echo "Copying the contents of openvino_saved_model directory here."
rsync -av ../openvino/* . --exclude=README.md
rsync -av ../output/*.hdf5 ./models/keras/

echo "Building Docker container"
docker build -t unet_openvino \
       --build-arg HTTP_PROXY=${HTTP_PROXY} \
       --build-arg HTTPS_PROXY=${HTTPS_PROXY} \
       --build-arg NO_PROXY=${NO_PROXY} \
       --build-arg http_proxy=${http_proxy} \
       --build-arg https_proxy=${https_proxy} \
       --build-arg no_proxy=${no_proxy} \
       .

if [ $? -eq 0 ]; then
    echo "Docker built successfully."
    echo "TO RUN BUILT DOCKER CONTAINER:"
    echo "1. For Neural Compute Stick 2 - 'docker run --net=host -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp.X11-unix --privileged -v /dev:/dev -it unet_openvino'"
    echo "2. For CPU - 'docker run -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp.X11-unix  -it unet_openvino'"
else
    echo "DOCKER BUILD FAILED."
fi

