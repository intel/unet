#!/usr/bin/env python
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

from argparser import args
"""
For BraTS (Task 1):

INPUT CHANNELS:  "modality": {
     "0": "FLAIR",
     "1": "T1w",
     "2": "t1gd",
     "3": "T2w"
 },
LABEL_CHANNELS: "labels": {
     "0": "background",
     "1": "edema",
     "2": "non-enhancing tumor",
     "3": "enhancing tumour"
 }

"""

import numpy as np
"""
This module loads the training and validation datasets.
If you have custom datasets, you can load and preprocess them here.
"""


if args.keras_api:
    import keras as K
else:
    from tensorflow import keras as K

"""
Load data from HDF5 file
"""


class PreprocessHDF5Matrix(K.utils.HDF5Matrix):
    """
    Wraps HDF5Matrix in preprocessing code.
    Performs image augmentation if needed.
    """

    def __init__(self, image_datagen, use_augmentation, seed, *args, **kwargs):
        self.image_datagen = image_datagen
        self.use_augmentation = use_augmentation
        self.seed = seed
        self.idx = 0
        super().__init__(*args, **kwargs)

    def __getitem__(self, key):
        data = super().__getitem__(key)
        self.idx += 1
        if len(data.shape) == 3:
            if self.use_augmentation:
                img = self.image_datagen.random_transform(
                    data, seed=self.seed + self.idx)
            else:
                img = data

            if args.channels_first:  # NCHW
                return np.swapaxes(img, 1, 3)
            else:
                return img
        else:  # Need to test the code below. Usually only 3D tensors expected
            if self.use_augmentation:
                img = np.array([
                    self.image_datagen.random_transform(
                        x, seed=self.seed + self.idx) for x in data
                ])
            else:
                img = np.array([x for x in data])

            if args.channels_first:  # NCHW
                return np.swapaxes(img, 1, 3)
            else:
                return img

def load_data(hdf5_data_filename):
    """
    Load the data from the HDF5 file using the Keras HDF5 wrapper.
    """

    # Training dataset
    # Make sure both input and label start with the same random seed
    # Otherwise they won't get the same random transformation

    image_datagen = K.preprocessing.image.ImageDataGenerator(
        zca_whitening=True, # Do ZCA pre-whitening to consider richer features
        shear_range=2, # Up to 2 degree random shear
        horizontal_flip=True,
        vertical_flip=True)

    msk_datagen = K.preprocessing.image.ImageDataGenerator(
        shear_range=2, # Up to 2 degree random shear
        horizontal_flip=True,
        vertical_flip=True)

    random_seed = 816
    imgs_train = PreprocessHDF5Matrix(image_datagen,
                                      args.use_augmentation,
                                      random_seed,
                                      hdf5_data_filename,
                                      "imgs_train")
    msks_train = PreprocessHDF5Matrix(msk_datagen,
                                      args.use_augmentation,
                                      random_seed,
                                      hdf5_data_filename,
                                      "msks_train")

    # Validation dataset
    # No data augmentation
    imgs_validation = PreprocessHDF5Matrix(image_datagen,
                                      False, # Don't augment
                                      816,
                                      hdf5_data_filename,
                                      "imgs_validation")
    msks_validation = PreprocessHDF5Matrix(image_datagen,
                                      False, # Don't augment
                                      816,
                                      hdf5_data_filename,
                                      "msks_validation")

    print("Batch size = {}".format(args.batch_size))

    print("Training image dimensions:   {}".format(imgs_train.shape))
    print("Training mask dimensions:    {}".format(msks_train.shape))
    print("Validation image dimensions: {}".format(imgs_validation.shape))
    print("Validation mask dimensions:  {}".format(msks_validation.shape))

    return imgs_train, msks_train, imgs_validation, msks_validation
