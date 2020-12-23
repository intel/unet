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
import tensorflow as tf
import numpy as np


class DatasetGenerator:
    """
    TensorFlow Dataset from Python/NumPy Iterator

    Loads the 2D slices NumPy files that were created via convert_raw_to_numpy.py
    """

    def __init__(self, dirName, batch_size=8, crop_dim=240, augment=False, seed=816):

        self.dirName = dirName
        self.batch_size = batch_size
        self.crop_dim = [crop_dim, crop_dim]
        self.augment = augment
        self.seed = seed

        self.file_list = tf.io.gfile.glob(self.dirName)
        self.__len__ = len(self.file_list)

        # Read one file to calculate the input and output shapes
        img, msk = self.read_file(self.file_list[0])
        self.input_shape = img.shape
        self.output_shape = msk.shape

    def __shape__(self):
        """
        Gets the shape of the data loader return
        """

        return self.get_input_shape(), self.get_output_shape()

    def get_input_shape(self):
        """
        Gets the shape of the input
        """
        return self.input_shape

    def get_output_shape(self):
        """
        Gets the shape of the output
        """
        return self.output_shape

    def z_normalize_img(self, img):
        """
        Normalize the image so that the mean value for each image
        is 0 and the standard deviation is 1.
        """
        for channel in range(img.shape[-1]):

            img_temp = img[..., channel]

            if np.std(img_temp) > 0:
                img_temp = (img_temp - np.mean(img_temp)) / np.std(img_temp)

            img[..., channel] = img_temp

        return img

    def crop_input(self, img, msk):
        """
        Randomly crop the image and mask
        """

        slices = []

        # Do we randomize?
        is_random = self.augment and np.random.rand() > 0.5

        for idx in range(2):  # Go through each dimension

            cropLen = self.crop_dim[idx]
            imgLen = img.shape[idx]

            start = (imgLen-cropLen)//2

            ratio_crop = 0.20  # Crop up this this % of pixels for offset
            # Number of pixels to offset crop in this dimension
            offset = int(np.floor(start*ratio_crop))

            if offset > 0:
                if is_random:
                    start += np.random.choice(range(-offset, offset))
                    if ((start + cropLen) > imgLen):  # Don't fall off the image
                        start = (imgLen-cropLen)//2
            else:
                start = 0

            slices.append(slice(start, start+cropLen))

        return img[tuple(slices)], msk[tuple(slices)]

    def __length__(self):
        """
        Number of items in the dataset
        """

        return self.__len__

    def combine_mask(self, msk):
        """
        Combine the masks into one mask
        """
        msk[msk > 0] = 1.0

        return msk

    def augment_data(self, img, msk):
        """
        Data augmentation
        Flip image and mask. Rotate image and mask.
        """

        if np.random.rand() > 0.5:
            ax = np.random.choice([0, 1])
            img = np.flip(img, ax)
            msk = np.flip(msk, ax)

        if np.random.rand() > 0.5:
            rot = np.random.choice([1, 2, 3])  # 90, 180, or 270 degrees

            img = np.rot90(img, rot, axes=[0, 1])  # Rotate axes 0 and 1
            msk = np.rot90(msk, rot, axes=[0, 1])  # Rotate axes 0 and 1

        return img, msk

    def read_file(self, fileIdx):
        """
        Read from the NumPy file
        """

        with np.load(fileIdx) as data:
            img = data["img"]
            msk = data["msk"]

        msk = self.combine_mask(msk)

        if self.crop_dim[0] != -1:  # Determine if we need to crop
        	img, msk = self.crop_input(img, msk)

        if self.augment:
            img, msk = self.augment_data(img, msk)

        img = self.z_normalize_img(img)

        return img, msk

    def read_file_tf(self, fileIdx):
        """
        Read file map to tf dataset
        """

        img, msk = self.read_file(fileIdx.numpy())

        return img, msk

    def get_dataset(self):
        """
        Return a dataset
        """
        ds = tf.data.Dataset.from_tensor_slices(
            self.file_list).shuffle(self.__len__, seed=self.seed)
        ds = ds.map(lambda x: tf.py_function(self.read_file_tf,
                                             [x], [tf.float32, tf.float32]),
                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

        return ds

    def plot_samples(self, num_samples=8):
        """
        Plot some dataset samples
        """
        import matplotlib.pyplot as plt

        dt = self.get_dataset().take(1).as_numpy_iterator()
        plt.figure(figsize=(20, 20))
        for img, msk in dt:
            if num_samples > img.shape[0]:
                num_samples = img.shape[0]
                
            for idx in range(num_samples):
                plt.subplot(num_samples, 2, 1+2*idx)
                plt.imshow(img[idx, :, :, 0], cmap="bone", origin="lower")
                plt.subplot(num_samples, 2, 2+2*idx)
                plt.imshow(msk[idx, :, :], cmap="bone", origin="lower")
