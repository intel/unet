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

class AugumentedHDF5Matrix(K.utils.HDF5Matrix):
    """
    Wraps HDF5Matrix with some image augumentation.
    """

    def __init__(self, image_datagen, seed, *args, **kwargs):
        self.image_datagen = image_datagen
        self.seed = seed
        self.idx = 0
        super().__init__(*args, **kwargs)

    def __getitem__(self, key):
        data = super().__getitem__(key)
        self.idx += 1
        if len(data.shape) == 3:
            img = self.image_datagen.random_transform(
                data, seed=self.seed + self.idx)
            if args.channels_first:  # NCHW
                return np.swapaxes(img, 1, 3)
            else:
                return img
        else:  # Need to test the code below. Usually only 3D tensors expected
            img = np.array([
                self.image_datagen.random_transform(
                    x, seed=self.seed + self.idx) for x in data
            ])
            if args.channels_first:  # NCHW
                return np.swapaxes(img, 1, 3)
            else:
                return img

def process_data(array):
    """
    Standard data processing. No augmentation.
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
    # Make sure both input and label start with the same random seed
    # Otherwise they won't get the same random transformation
    if args.use_augmentation:

        # Keras image preprocessing performs randomized rotations/flips
        image_datagen = K.preprocessing.image.ImageDataGenerator(
            zca_whitening=True, # Do ZCA pre-whitening to consider richer features
            shear_range=2, # Up to 2 degree random shear
            horizontal_flip=True,
            vertical_flip=True)

        imgs_train = AugumentedHDF5Matrix(image_datagen, 816,
                                          hdf5_data_filename,
                                          "imgs_train")
        msks_train = AugumentedHDF5Matrix(image_datagen, 816,
                                          hdf5_data_filename,
                                          "msks_train")
    else:
        imgs_train = K.utils.HDF5Matrix(hdf5_data_filename,
                                             "imgs_train",
                                             normalizer=process_data)
        msks_train = K.utils.HDF5Matrix(hdf5_data_filename,
                                             "msks_train",
                                             normalizer=process_data)

    # Validation dataset
    # No data augmentation
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
