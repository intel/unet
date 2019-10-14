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
from argparser import args

TRAIN_TESTVAL_SEED = 816

if args.keras_api:
    import keras as K
else:
    from tensorflow import keras as K

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

        if setType not in ["train", "test", "validate"]:
            print("Dataloader error.  You forgot to specify train, test, or validate.")

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

        np.random.seed(TRAIN_TESTVAL_SEED)  # Seed has to be same for all workers so that train/test/val lists are the same
        self.list_IDs = self.create_file_list()
        self.num_images = self.get_length()

        np.random.seed(self.seed)  # Now seed workers differently so that the sequence is different for each worker
        self.on_epoch_end()   # Generate the sequence

        self.num_batches = self.__len__()

        # Determine if axes are equal and can be rotated
        # If the axes aren't equal then we can't rotate them.
        equal_dim_axis = []
        for idx in range(0, len(dim)):
            for jdx in range(idx+1, len(dim)):
                if dim[idx] == dim[jdx]:
                    equal_dim_axis.append([idx, jdx])  # Valid rotation axes
        self.dim_to_rotate = equal_dim_axis

    def get_length(self):
        """
        Get the length of the list of file IDs associated with this data loader
        """
        return len(self.list_IDs)

    def get_file_list(self):
        """
        Get the list of file IDs associated with this data loader
        """
        return self.list_IDs

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

    def create_file_list(self):
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

        idxList = np.random.permutation(idxList)  # Randomize list

        train_len = int(np.floor(numFiles * self.train_test_split)) # Number of training files
        test_val_len = numFiles - train_len
        val_len = int(np.floor(test_val_len * self.validate_test_split))  # Number of validation files
        test_len = test_val_len - val_len  # Number of testing files

        trainIdx = idxList[0:train_len]  # List of training indices
        validateIdx = idxList[train_len:(train_len+val_len)]  # List of validation indices
        testIdx = idxList[-test_len:]  # List of testing indices (last testIdx elements)

        if self.setType == "train":

            with open("train.csv", "w") as writeFile:
                fileList = {}
                for idx in trainIdx:
                    fileList[self.imgFiles[idx]] = self.mskFiles[idx]

                for img in sorted(fileList):
                    writeFile.write("{},{}\n".format(img, fileList[img]))

            writeFile.close()

            return trainIdx
        elif self.setType == "validate":

            with open("validate.csv", "w") as writeFile:
                fileList = {}
                for idx in validateIdx:
                    fileList[self.imgFiles[idx]] = self.mskFiles[idx]

                for img in sorted(fileList):
                    writeFile.write("{},{}\n".format(img, fileList[img]))

            writeFile.close()

            return validateIdx
        elif self.setType == "test":

            with open("test.csv", "w") as writeFile:
                fileList = {}
                for idx in testIdx:
                    fileList[self.imgFiles[idx]] = self.mskFiles[idx]

                for img in sorted(fileList):
                    writeFile.write("{},{}\n".format(img, fileList[img]))

            writeFile.close()

            return testIdx
        else:
            print("Error. You forgot to specify train, test, or validate. Instead received {}".format(self.setType))
            return []

    def __len__(self):
        """
        The number of batches per epoch
        """
        return self.num_images // self.batch_size

    def __getitem__(self, index):
        """
        Generate one batch of data
        """
        # Generate indicies of the batch
        indexes = np.sort(
            self.indexes[(index*self.batch_size):((index+1)*self.batch_size)])

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return (X, y)

    def get_batch(self, index):
        """
        Public method to get one batch of data
        """
        return self.__getitem__(index)

    def on_epoch_end(self):
        """
        Updates indices after each epoch
        If shuffle is true, then it will shuffle the training set
        after every epoch.
        """
        self.indexes = np.arange(self.num_images)
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

        return img, msk

    def z_normalize_img(self, img):
        """
        Normalize the image so that the mean value for each image
        is 0 and the standard deviation is 1.
        """
        for channel in range(img.shape[-1]):

            img_temp = img[..., channel]
            img_temp = (img_temp - np.mean(img_temp)) / np.std(img_temp)

            img[..., channel] = img_temp

        return img

    def get_batch_fileIDs(self):
        """
        Get the file IDs of the last batch that was loaded.
        """
        return self.fileIDs

    def __data_generation(self, list_IDs_temp):
        """
        Generates data containing batch_size samples

        This just reads the list of filename to load.
        Change this to suit your dataset.
        """

        # Make empty arrays for the images and mask batches
        imgs = np.zeros((self.batch_size, *self.dim, self.n_in_channels))
        msks = np.zeros((self.batch_size, *self.dim, self.n_out_channels))

        self.fileIDs = {}

        for idx, fileIdx in enumerate(list_IDs_temp):

            img_temp = np.array(nib.load(self.imgFiles[fileIdx]).dataobj)

            filename = ntpath.basename(self.imgFiles[fileIdx])  # Strip all but filename
            filename = os.path.splitext(filename)[0]
            self.fileIDs[idx] = os.path.splitext(filename)[0]

            """
            "modality": {
                 "0": "FLAIR",
                 "1": "T1w",
                 "2": "t1gd",
                 "3": "T2w"
            """
            if self.n_in_channels == 1:
                img = img_temp[:, :, :, [0]]  # FLAIR channel
            else:
                img = img_temp

            # Get mask data
            msk = np.array(nib.load(self.mskFiles[fileIdx]).dataobj)

            """
            "labels": {
                 "0": "background",
                 "1": "edema",
                 "2": "non-enhancing tumor",
                 "3": "enhancing tumour"}
             """
            # Combine all masks but background
            msk[msk > 0] = 1.0
            msk = np.expand_dims(msk, -1)

            # Take a crop of the patch_dim size
            img, msk = self.crop_img(img, msk, self.augment)

            img = self.z_normalize_img(img)  # Normalize the image

            # Data augmentation
            if self.augment and (np.random.rand() > 0.5):
                img, msk = self.augment_data(img, msk)

            imgs[idx, ] = img
            msks[idx, ] = msk

        return imgs, msks
