#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2018 Intel Corporation
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

import numpy as np
"""
This module loads the training and validation datasets.
If you have custom datasets, you can load and preprocess them here.
"""

from argparser import args

if args.keras_api:
    import keras as K
else:
    from tensorflow import keras as K


"""
Load data from HDF5 file
"""


def process_data(array):
    """
    You can process (e.g. augment) the data as it loads
    by using this function.
    """

    # Data was saved as NHWC (channels last)
    if args.channels_first:  # NCHW
        return np.swapaxes(array, 1, 3)
    else:
        return array


def load_data(hdf5_data_filename):
    """
    Load the data from the HDF5 file using the Keras HDF5 wrapper.
    """

    # Training dataset
    imgs_train = K.utils.HDF5Matrix(hdf5_data_filename,
                                    "imgs_train",
                                    normalizer=process_data)
    msks_train = K.utils.HDF5Matrix(hdf5_data_filename,
                                    "msks_train",
                                    normalizer=process_data)

    # Validation dataset
    imgs_validation = K.utils.HDF5Matrix(hdf5_data_filename,
                                         "imgs_validation",
                                         normalizer=process_data)
    msks_validation = K.utils.HDF5Matrix(hdf5_data_filename,
                                         "msks_validation",
                                         normalizer=process_data)

    print("Batch size = {}".format(args.batch_size))

    print("Training image dimensions:   {}".format(imgs_train.shape))
    print("Training mask dimensions:    {}".format(msks_train.shape))
    print("Validation image dimensions: {}".format(imgs_validation.shape))
    print("Validation mask dimensions:  {}".format(msks_validation.shape))

    return imgs_train, msks_train, imgs_validation, msks_validation
