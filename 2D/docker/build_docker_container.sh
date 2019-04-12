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

