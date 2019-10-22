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

import h5py

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

    def __init__(self, datapath, dataset, datagen, start=0, end=None,
                 normalizer=None, crop_dim=128,
                 use_augmentation=False, seed=args.seed,
                 channels_first=args.channels_first):
        """
        This will need to keep up with the HDF5Matrix code
        base.  It allows us to do random image cropping and
        use Keras online data augmentation.
        """
        self.image_datagen = datagen
        self.use_augmentation = use_augmentation
        self.seed = seed
        self.crop_dim = crop_dim
        self.idx = 0
        self.channels_first = channels_first

        if h5py is None:
            raise ImportError("The use of HDF5Matrix requires "
                              "HDF5 and h5py installed.")

        if datapath not in list(self.refs.keys()):
            f = h5py.File(datapath, "r")
            self.refs[datapath] = f
        else:
            f = self.refs[datapath]
        self.data = f[dataset]
        self.start = start
        if end is None:
            self.end = self.data.shape[0]
        else:
            self.end = end
        self.normalizer = normalizer
        if self.normalizer is not None:
            first_val = self.normalizer(self.data[0:1])
        else:
            first_val = self.data[0:1]
        self._base_dtype = first_val.dtype

        # Handle the shape if we want to crop
        bshape = list(first_val.shape[1:])

        self.original_height = bshape[0]
        self.original_width = bshape[1]

        if ((self.crop_dim[0] > 0) and
            (self.crop_dim[1] > 0) and
            (self.crop_dim[0] < self.original_height) and
                (self.crop_dim[1] < self.original_width)):
            bshape[0] = self.crop_dim[0]
            bshape[1] = self.crop_dim[1]
            base_shape = tuple(bshape)
            self.crop = True
        else:
            # Don't crop
            base_shape = first_val.shape[1:]
            self.crop = False

        if self.channels_first:
            self._base_shape = tuple([base_shape[2], base_shape[0],
                                      base_shape[1]])
        else:
            self._base_shape = base_shape

    def random_crop_img(self, img):
        """
        Random crop image - Assumes 2D images. NHWC format
        """

        if self.crop:

            # Assuming NHWC format
            height, width = img.shape[1], img.shape[2]
            dx, dy = self.crop_dim[0], self.crop_dim[1]

            img_temp = np.zeros((img.shape[0], dx, dy, img.shape[3]))
            for idx in range(img.shape[0]):

                if self.use_augmentation:
                    x = np.random.randint(0, height - dx + 1)
                    y = np.random.randint(0, width - dy + 1)

                else:  # If no augmentation, then just do center crop
                    x = (height - dx) // 2
                    y = (width - dy) // 2

                img_temp[idx] = img[idx, x:(x + dx), y:(y + dy), :]

            return img_temp
        else:  # Don't crop
            return img

    def __getitem__(self, key):
        """
        Grab a batch of images and do online data augmentation and cropping
        """
        data = super().__getitem__(key)
        self.idx += 1
        if len(data.shape) == 3:
            if self.use_augmentation:
                img = self.image_datagen.random_transform(
                    data, seed=self.seed + self.idx)
            else:
                img = data

            img = self.random_crop_img(img)

            if self.channels_first:  # NCHW
                outData = np.swapaxes(img, 1, 3)
            else:
                outData = img

        else:  # Need to test the code below. Usually only 3D tensors expected
            if self.use_augmentation:
                img = np.array([
                    self.image_datagen.random_transform(
                        x, seed=self.seed + self.idx) for x in data
                ])
            else:
                img = np.array([x for x in data])

            img = self.random_crop_img(img)

            if self.channels_first:  # NCHW
                outData = np.swapaxes(img, 1, 3)
            else:
                outData = img

        return outData


def load_data(hdf5_data_filename, batch_size=128, crop_dim=[-1, -1],
              channels_first=args.channels_first, seed=args.seed):
    """
    Load the data from the HDF5 file using the Keras HDF5 wrapper.
    """

    # Training dataset
    # Make sure both input and label start with the same random seed
    # Otherwise they won't get the same random transformation

    params = dict(horizontal_flip=True,
                  vertical_flip=True,
                  rotation_range=90,  # degrees
                  shear_range=5  # degrees
                  )
    image_datagen = K.preprocessing.image.ImageDataGenerator(**params)
    msk_datagen = K.preprocessing.image.ImageDataGenerator(**params)

    imgs_train = PreprocessHDF5Matrix(hdf5_data_filename,
                                      "imgs_train",
                                      image_datagen,
                                      crop_dim=crop_dim,
                                      use_augmentation=args.use_augmentation,
                                      seed=seed,
                                      channels_first=channels_first)
    msks_train = PreprocessHDF5Matrix(hdf5_data_filename,
                                      "msks_train",
                                      msk_datagen,
                                      crop_dim=crop_dim,
                                      use_augmentation=args.use_augmentation,
                                      seed=seed,
                                      channels_first=channels_first)

    # Validation dataset
    # No data augmentation
    imgs_validation = PreprocessHDF5Matrix(hdf5_data_filename,
                                           "imgs_validation",
                                           image_datagen,
                                           crop_dim=crop_dim,
                                           use_augmentation=False,  # Don't augment
                                           seed=seed,
                                           channels_first=channels_first)
    msks_validation = PreprocessHDF5Matrix(hdf5_data_filename,
                                           "msks_validation",
                                           msk_datagen,
                                           crop_dim=crop_dim,
                                           use_augmentation=False,  # Don't augment
                                           seed=seed,
                                           channels_first=channels_first)

    # Testing dataset
    # No data augmentation
    imgs_testing = PreprocessHDF5Matrix(hdf5_data_filename,
                                        "imgs_testing",
                                        image_datagen,
                                        crop_dim=crop_dim,
                                        use_augmentation=False,  # Don't augment
                                        seed=seed,
                                        channels_first=channels_first)
    msks_testing = PreprocessHDF5Matrix(hdf5_data_filename,
                                        "msks_testing",
                                        msk_datagen,
                                        crop_dim=crop_dim,
                                        use_augmentation=False,  # Don't augment
                                        seed=seed,
                                        channels_first=channels_first)

    print("Batch size = {}".format(batch_size))

    print("Training image dimensions:   {}".format(imgs_train.shape))
    print("Training mask dimensions:    {}".format(msks_train.shape))
    print("Validation image dimensions: {}".format(imgs_validation.shape))
    print("Validation mask dimensions:  {}".format(msks_validation.shape))
    print("Testing image dimensions: {}".format(imgs_testing.shape))
    print("Testing mask dimensions:  {}".format(msks_testing.shape))

    return imgs_train, msks_train, imgs_validation, msks_validation, imgs_testing, msks_testing
