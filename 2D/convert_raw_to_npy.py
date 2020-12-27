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
"""
Converts the Medical Decathlon raw 3D Nifti files into
2D NumPy files for easier use in TensorFlow/Keras 2D models.

You'll need to download the raw dataset from
the Medical Decathlon website (http://medicaldecathlon.com),
extract the data (untar), and run this script.

The raw dataset has the CC-BY-SA 4.0 license.
https://creativecommons.org/licenses/by-sa/4.0/

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

import os
import nibabel as nib  # pip install nibabel
import numpy as np
from tqdm import tqdm  # pip install tqdm
import json
import settings


def crop_center(img, cropx, cropy, cropz):
    """
    Take a center crop of the images.
    If we are using a 2D model, then we'll just stack the
    z dimension.
    """

    x, y, z, c = img.shape

    # Make sure starting index is >= 0
    startx = max(x // 2 - (cropx // 2), 0)
    starty = max(y // 2 - (cropy // 2), 0)
    startz = max(z // 2 - (cropz // 2), 0)

    # Make sure ending index is <= size
    endx = min(startx + cropx, x)
    endy = min(starty + cropy, y)
    endz = min(startz + cropz, z)

    return img[startx:endx, starty:endy, startz:endz, :]


def normalize_img(img):
    """
    Normalize the pixel values.
    This is one of the most important preprocessing steps.
    We need to make sure that the pixel values have a mean of 0
    and a standard deviation of 1 to help the model to train
    faster and more accurately.
    """

    for channel in range(img.shape[3]):
        img[:, :, :, channel] = (
            img[:, :, :, channel] - np.mean(img[:, :, :, channel])) \
            / np.std(img[:, :, :, channel])

    return img


def preprocess_inputs(img, resize):
    """
    Process the input images

    For BraTS subset:
    INPUT CHANNELS:  "modality": {
             "0": "FLAIR", T2-weighted-Fluid-Attenuated Inversion Recovery MRI
             "1": "T1w",  T1-weighted MRI
             "2": "t1gd", T1-gadolinium contrast MRI
             "3": "T2w"   T2-weighted MRI
     }
    """
    while len(img.shape) < 4:  # Make sure 4D
        img = np.expand_dims(img, -1)

    if resize != -1:
       img = crop_center(img, resize, resize, resize)
    img = normalize_img(img)

    img = np.swapaxes(np.array(img), 0, -2)

    return img


def preprocess_labels(msk, num_labels, resize):
    """
    Process the ground truth labels

    For BraTS subset:
    LABEL_CHANNELS: "labels": {
             "0": "background",  No tumor
             "1": "edema",       Swelling around tumor
             "2": "non-enhancing tumor",  Tumor that isn't enhanced by Gadolinium contrast
             "3": "enhancing tumour"  Gadolinium contrast enhanced regions
     }

    """

    while len(msk.shape) < 4:  # Make sure 4D
        msk = np.expand_dims(msk, -1)

    if resize != -1:
       msk = crop_center(msk, resize, resize, resize)

    msk = np.swapaxes(np.array(msk), 0, -2)

    return msk


def save_img_msk(idx_array, name, fileIdx, save_dir, dataDir, num_labels, resize):
    """
    Save the image and mask in a numpy file
    """

    for idx in tqdm(idx_array):

        image_file = fileIdx[idx]["image"]
        label_file = fileIdx[idx]["label"]

        bratsname = os.path.splitext(os.path.basename(image_file))[0]
        bratsname = os.path.splitext(bratsname)[0]

        data_filename = os.path.join(dataDir, image_file)
        img = np.array(nib.load(data_filename).dataobj)

        img = preprocess_inputs(img, resize)
        num_rows = img.shape[0]

        data_filename = os.path.join(dataDir, label_file)
        msk = np.array(nib.load(data_filename).dataobj)
        msk = preprocess_labels(msk, num_labels, resize)

        for idy in range(num_rows):
            np.savez(os.path.join(save_dir, name,
                                  "{}_{:03d}".format(bratsname, idy)),
                     img=img[idy],
                     msk=msk[idy])


def convert_raw_data_to_numpy(trainIdx, validateIdx, testIdx,
                              dataDir, json_data, save_dir, resize=-1):
    """
    Go through the Decathlon dataset.json file.
    We've already split into training and validation subsets.
    Read in Nifti format files. Crop images and masks.
    This code is will convert the 3D images and masks
    into a stack of 2D slices.
    """

    num_labels = len(json_data["labels"])
    fileIdx = json_data["training"]

    # Create directory
    try:
        os.makedirs(save_dir)
    except OSError:
        if not os.path.isdir(save_dir):
            raise

    # Save training set images
    print("Step 1 of 3. Save training set 3D scans to 2D slices.")
    dirname = "train"
    try:
        os.makedirs(os.path.join(save_dir, dirname))
    except OSError:
        if not os.path.isdir(save_dir):
            raise
    save_img_msk(trainIdx, dirname, fileIdx, save_dir, dataDir, num_labels, resize)

    # Save testing set images
    print("Step 2 of 3. Save testing set 3D scans to 2D slices.")
    dirname = "testing"
    try:
        os.makedirs(os.path.join(save_dir, dirname))
    except OSError:
        if not os.path.isdir(save_dir):
            raise
    save_img_msk(testIdx, dirname, fileIdx, save_dir, dataDir, num_labels, resize)

    # Save validation set images
    print("Step 3 of 3. Save validation set 3D scans to 2D slices.")
    dirname = "validation"
    try:
        os.makedirs(os.path.join(save_dir, dirname))
    except OSError:
        if not os.path.isdir(save_dir):
            raise
    save_img_msk(validateIdx, dirname, fileIdx, save_dir, dataDir, num_labels, resize)

    print("Finished processing.")


if __name__ == "__main__":

    from argparser import args
    print(args)

    """
	Get the training file names from the data directory.
	Decathlon should always have a dataset.json file in the
	subdirectory which lists the experiment information including
	the input and label filenames.
	"""

    json_filename = os.path.join(args.original_data_path, "dataset.json")

    try:
        with open(json_filename, "r") as fp:
            experiment_data = json.load(fp)
    except IOError as e:
        print("File {} doesn't exist. It should be part of the "
              "Decathlon directory".format(json_filename))

    # Print information about the Decathlon experiment data
    print("*" * 30)
    print("=" * 30)
    print("Dataset name:        ", experiment_data["name"])
    print("Dataset description: ", experiment_data["description"])
    print("Tensor image size:   ", experiment_data["tensorImageSize"])
    print("Dataset release:     ", experiment_data["release"])
    print("Dataset reference:   ", experiment_data["reference"])
    print("Dataset license:     ", experiment_data["licence"])  # sic
    print("=" * 30)
    print("*" * 30)

    """
	Randomize the file list. Then separate into training and
	validation lists. We won't use the testing set since we
	don't have ground truth masks for this; instead we'll
	split the validation set into separate test and validation
	sets.
	"""
    # Set the random seed so that always get same random mix
    np.random.seed(args.seed)
    numFiles = experiment_data["numTraining"]
    idxList = np.arange(numFiles)  # List of file indices
    randomList = np.random.random(numFiles)  # List of random numbers
    # Random number go from 0 to 1. So anything above
    # args.train_split is in the validation list.
    trainList = idxList[randomList < args.split]

    otherList = idxList[randomList >= args.split]
    randomList = np.random.random(len(otherList))  # List of random numbers
    validateList = otherList[randomList >= 0.5]
    testList = otherList[randomList < 0.5]

    convert_raw_data_to_numpy(trainList, validateList, testList,
                              args.original_data_path,
                              experiment_data,
                              args.data_path,
                              args.resize)
