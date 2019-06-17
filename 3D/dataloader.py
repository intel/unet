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

import nibabel as nib
import json
import ntpath
import os
import numpy as np
import tensorflow as tf
import keras as K
#from tensorflow import keras as K

CHANNEL_LAST = True
if CHANNEL_LAST:
    concat_axis = -1
    data_format = "channels_last"

else:
    concat_axis = 1
    data_format = "channels_first"

import os
import sys
import cv2
import numpy as np
import vdms

from multiprocessing.pool import ThreadPool

class vdms_loader(object):

    """
    A dataloader to get entire 3D MRI and label mask from
    VDMS. Input is a connection (url) and a dictionary
    specifying the scan type as the key and the
    shape of the tensor as the value (channels, height, width, depth/slices)

    Usage:
    loader = vdms_loader(connection = "sky4.jf.intel.com",
                         scan_types = { "mri" : [240, 240, 155, 4],
                                        "label" : [240, 240, 155, 4]}
                         )

    """

    def __init__(self, connection, scan_types):
        """
        Scan Types is a dictionary with the key being
        the name of the scan in the VDMS database and
        the value being an integer array of the
        number of channels, height, width, and depth (# slices) of the images
        """
        self.scan_types = scan_types
        self.scan_type_names = list(scan_types.keys())
        self.num_scan_types = len(self.scan_type_names)   # MRI and Label
        self.connection = connection

        self.num_channels = []
        for idx, name in enumerate(scan_types):
            if CHANNEL_LAST:
                self.num_channels.append(scan_types[name][-1])
            else:
                self.num_channels.append(scan_types[name][0])

    def get_scan(self, scan_id, img_type, channel):

        db = vdms.vdms()
        db.connect(self.connection)

        query = [{
            "FindEntity": {
                "class": "Scan",
                "_ref": 33,
                "constraints": {
                    "id": ["==", scan_id]
                }
            }
        }, {
            "FindImage": {
                "link": {"ref": 33},
                "constraints": {
                    "channel":   ["==", channel],
                    "type": ["==", img_type]
                },
                "results": {
                    # "list": ["id", "type", "channel", "slice_number"],
                    "sort": "slice_number"  # Important to sort by slice #
                }
            }
        }]

        response, arr = db.query(query)

        # print(db.get_last_response_str())
        # print(len(arr))

        return arr

    def get_batch(self, patient_list):

        num_patients = len(patient_list)

        # Create a thread pool for the VDMS calls
        pool = ThreadPool(processes=1) #np.prod(self.num_channels) * num_patients)

        combined_result = {}
        vdms_result = {}
        for name_idx, name in enumerate(self.scan_type_names):
            for patient_num, patient in enumerate(patient_list):
                for channel in range(self.num_channels[name_idx]):
                    """
                    Make the VDMS calls as asyncronous process pools
                    so that they can execute in parallel.
                    """
                    vdms_result[name_idx, patient_num, channel] = pool.apply_async(
                        self.get_scan, (patient, name, channel))

        for name_idx, name in enumerate(self.scan_type_names):

            combined_result[name] = np.ndarray(
                shape=([num_patients] + self.scan_types[name]))

            for patient_idx, patient in enumerate(patient_list):
                for channel in range(self.num_channels[name_idx]):

                    """
                    TODO: Not sure if this code should be changed. Currently,
                    it is going through sequentially; however, since the
                    processes are run in parallel asynchronously, there
                    is no reason why they should come back in a particular
                    order.
                    """
                    for slice_num, pngs in enumerate(vdms_result[name_idx, patient_idx, channel].get()):  # Get the slices

                        # The imdecode will create a RGB image
                        # Just take single channel
                        image_input = cv2.imdecode(np.frombuffer(
                            pngs, dtype=np.uint8), 1)  # for vdms

                        if CHANNEL_LAST:
                            combined_result[name][patient_idx, :, :, slice_num,
                                                    channel] = image_input[:, :, 2]
                        else:
                            # Change data layout from HWC to CHW (dont need when using .npz )
                            image = image_input.transpose((2, 0, 1))
                            combined_result[name][patient_idx, channel, :, :,
                                                    slice_num] = image[2, :, :]

        return combined_result


class DataGenerator(K.utils.Sequence):
    """
    Generates data for Keras/TensorFlow

    This uses the Keras Sequence which is a better data pipeline.
    It will allow for multiple data generator processes and
    batch pre-fetching.

    If you have a different type of dataset, you'll just need to
    change the loading code in self.__data_generation to return
    the correct image and label.

    """

    def __init__(self,
                 setType,     # ["train", "validate", "test"]
                 data_path,    # File path for data
                 train_test_split=0.85,  # Train test split
                 validate_test_split=0.5,  # Validation/test split
                 batch_size=8,  # batch size
                 dim=(128, 128, 128),  # Dimension of images/masks
                 n_in_channels=1,  # Number of channels in image
                 n_out_channels=1,  # Number of channels in mask
                 shuffle=True,  # Shuffle list after each epoch
                 augment=False,   # Augment images
                 seed=816):     # Seed for random number generator
        """
        Initialization
        """
        self.data_path = data_path
        self.setType = setType
        self.dim = dim
        self.batch_size = batch_size
        self.train_test_split = train_test_split
        self.validate_test_split = validate_test_split

        self.n_in_channels = n_in_channels
        self.n_out_channels = n_out_channels
        self.shuffle = shuffle
        self.augment = augment

        self.seed = seed
        self.list_IDs = self.get_file_list()
        self.num_images = self.get_length()

        self.on_epoch_end()   # Generate the sequence

        self.num_batches = self.__len__()

        if CHANNEL_LAST:
            self.loader = vdms_loader(connection="hsw3.jf.intel.com",
                                      scan_types={"mri": [240, 240, 155, 4],
                                      "label": [240, 240, 155, 4]} )
        else:
            self.loader = vdms_loader(connection="hsw3.jf.intel.com",
                                      scan_types={"mri": [240, 240, 155, 4],
                                      "label": [240, 240, 155, 4]} )

        # Determine if axes are equal and can be rotated
        # If the axes aren't equal then we can't rotate them.
        equal_dim_axis = []
        for idx in range(0, len(dim)):
            for jdx in range(idx+1, len(dim)):
                if dim[idx] == dim[jdx]:
                    equal_dim_axis.append([idx, jdx])  # Valid rotation axes
        self.dim_to_rotate = equal_dim_axis

    def get_length(self):
        return len(self.list_IDs)

    def print_info(self):
        """
        Print the dataset information
        """

        print("*"*30)
        print("="*30)
        print("Number of {} images = {}".format(self.setType, self.num_images))
        print("Dataset name:        ", self.name)
        print("Dataset description: ", self.description)
        print("Tensor image size:   ", self.tensorImageSize)
        print("Dataset release:     ", self.release)
        print("Dataset reference:   ", self.reference)
        print("Input channels:      ", self.input_channels)
        print("Output labels:       ", self.output_channels)
        print("Dataset license:     ", self.license)
        print("="*30)
        print("*"*30)

    def get_file_list(self):
        """
        Get list of the files from the BraTS raw data
        Split into training and testing sets.
        """
        json_filename = os.path.join(self.data_path, "dataset.json")

        try:
            with open(json_filename, "r") as fp:
                experiment_data = json.load(fp)
        except IOError as e:
            print("File {} doesn't exist. It should be part of the "
                  "Decathlon directory".format(json_filename))

        self.output_channels = experiment_data["labels"]
        self.input_channels = experiment_data["modality"]
        self.description = experiment_data["description"]
        self.name = experiment_data["name"]
        self.release = experiment_data["release"]
        self.license = experiment_data["licence"]
        self.reference = experiment_data["reference"]
        self.tensorImageSize = experiment_data["tensorImageSize"]

        """
        Randomize the file list. Then separate into training and
        validation lists. We won't use the testing set since we
        don't have ground truth masks for this.
        """
        numFiles = experiment_data["numTraining"]
        idxList = np.arange(numFiles)  # List of file indices

        self.imgFiles = {}
        self.mskFiles = {}

        for idx in idxList:
            self.imgFiles[idx] = os.path.join(self.data_path,
                                              experiment_data["training"][idx]["image"])
            self.mskFiles[idx] = os.path.join(self.data_path,
                                              experiment_data["training"][idx]["label"])

        np.random.seed(self.seed)
        randomIdx = np.random.random(numFiles)  # List of random numbers
        # Random number go from 0 to 1. So anything above
        # self.train_split is in the validation list.
        trainIdx = idxList[randomIdx < self.train_test_split]

        listIdx = idxList[randomIdx >= self.train_test_split]
        randomIdx = np.random.random(len(listIdx))  # List of random numbers
        validateIdx = listIdx[randomIdx >= self.validate_test_split]
        testIdx = listIdx[randomIdx < self.validate_test_split]

        if self.setType == "train":
            return trainIdx
        elif self.setType == "validate":
            return validateIdx
        elif self.setType == "test":
            return testIdx
        else:
            print("error with type of data: {}".format(self.setType))
            return []

    def __len__(self):
        """
        The number of batches per epoch
        """
        return len(self.list_IDs) // self.batch_size

    def __getitem__(self, index):
        """
        Generate one batch of data
        """
        # Generate indicies of the batch
        indexes = np.sort(
            self.indexes[index*self.batch_size:(index+1)*self.batch_size])

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def get_batch(self, index):
        """
        Public method to get one batch of data
        """
        return self.__getitem__(index)

    def get_batch_fileIDs(self, index):
        """
        Get the original filenames for the batch at this index
        """
        indexes = np.sort(
            self.indexes[index*self.batch_size:(index+1)*self.batch_size])
        fileIDs = {}

        for idx, fileIdx in enumerate(indexes):
            name = self.imgFiles[fileIdx]
            filename = ntpath.basename(name)  # Strip all but filename
            filename = os.path.splitext(filename)[0]
            fileIDs[idx] = os.path.splitext(filename)[0]

        return fileIDs

    def on_epoch_end(self):
        """
        Updates indices after each epoch
        If shuffle is true, then it will shuffle the training set
        after every epoch.
        """
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def crop_img(self, img, msk, randomize=True):
        """
        Crop the image and mask
        """

        slices = []

        # Only randomize half when asked
        randomize = randomize and (np.random.rand() > 0.5)

        for idx in range(len(self.dim)):  # Go through each dimension

            cropLen = self.dim[idx]
            imgLen = img.shape[idx]

            start = (imgLen-cropLen)//2

            ratio_crop = 0.20  # Crop up this this % of pixels for offset
            # Number of pixels to offset crop in this dimension
            offset = int(np.floor(start*ratio_crop))

            if offset > 0:
                if randomize:
                    start += np.random.choice(range(-offset, offset))
                    if ((start + cropLen) > imgLen):  # Don't fall off the image
                        start = (imgLen-cropLen)//2
            else:
                start = 0

            slices.append(slice(start, start+cropLen))

        # No slicing along channels
        slices.append(slice(0, self.n_in_channels))

        return img[tuple(slices)], msk[tuple(slices)]

    def augment_data(self, img, msk):
        """
        Data augmentation
        Flip image and mask. Rotate image and mask.
        """

        if np.random.rand() > 0.5:
            # Random 0,1 (axes to flip)
            ax = np.random.choice(np.arange(len(self.dim)-1))
            img = np.flip(img, ax)
            msk = np.flip(msk, ax)

        elif (len(self.dim_to_rotate) > 0) and (np.random.rand() > 0.5):
            rot = np.random.choice([1, 2, 3])  # 90, 180, or 270 degrees

            # This will choose the axes to rotate
            # Axes must be equal in size
            random_axis = self.dim_to_rotate[np.random.choice(
                len(self.dim_to_rotate))]
            img = np.rot90(img, rot, axes=random_axis)  # Rotate axes 0 and 1
            msk = np.rot90(msk, rot, axes=random_axis)  # Rotate axes 0 and 1

        # elif np.random.rand() > 0.5:
        #     rot = np.random.choice([1, 2, 3])  # 90, 180, or 270 degrees
        #     axis = np.random.choice([0, 1]) # Axis to rotate through
        #     img = np.rot90(img, rot, axes=(axis,2))
        #     msk = np.rot90(msk, rot, axes=(axis,2))

        return img, msk

    def z_normalize_img(self, img):
        """
        Normalize the image so that the mean value for each image
        is 0 and the standard deviation is 1.
        """
        for channel in range(img.shape[-1]):

            img_temp = img[..., channel]
            if np.std(img_temp)!=0:
              img_temp = (img_temp - np.mean(img_temp)) / np.std(img_temp)

            img[..., channel] = img_temp

        return img

    def __data_generation(self, list_IDs_temp):
        """
        Generates data containing batch_size samples

        This just reads the list of filename to load.
        Change this to suit your dataset.
        """

        # Make empty arrays for the images and mask batches
        imgs = np.zeros((self.batch_size, *self.dim, self.n_in_channels))
        msks = np.zeros((self.batch_size, *self.dim, self.n_out_channels))

        fileList = ["BRATS_{0:03d}".format(x) for x in list_IDs_temp]
        #fileList = ["BRATS_301" for x in list_IDs_temp]
        results = self.loader.get_batch(fileList)

        # results = self.loader.get_batch(
        #     ["BRATS_301", "BRATS_020", "BRATS_047", "BRATS_024", "BRATS_397"])
        img_temp, msk_temp = results["mri"], results["label"]

        for idx, fileIdx in enumerate(list_IDs_temp):

            """
            "modality": {
                 "0": "FLAIR",
                 "1": "T1w",
                 "2": "t1gd",
                 "3": "T2w"
            """

            if self.n_in_channels == 1:
                img = img_temp[idx, :, :, :, 0]  # FLAIR channel
                img = np.expand_dims(img, -1)
            else:
                img = img_temp[idx]

            """
            "labels": {
                 "0": "background",
                 "1": "edema",
                 "2": "non-enhancing tumor",
                 "3": "enhancing tumour"}
             """
            # Combine all masks but background
            msk = msk_temp[idx]
            msk = msk[:, :, :, [0]] + msk[:, :, :, [1]] \
                + msk[:, :, :, [2]] + msk[:, :, :, [3]]
            msk[msk > 1] = 1.0

            # Take a crop of the patch_dim size
            img, msk = self.crop_img(img, msk, self.augment)

            img = self.z_normalize_img(img)  # Normalize the image

            # Data augmentation
            if self.augment and (np.random.rand() > 0.5):
                img, msk = self.augment_data(img, msk)

            imgs[idx, ] = img
            msks[idx, ] = msk

        return imgs, msks
