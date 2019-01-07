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

import os
import nibabel as nib
import numpy as np
from tqdm import tqdm
import glob

import argparse

parser = argparse.ArgumentParser(
    description="Convert Decathlon raw Nifti data "
    "(http://medicaldecathlon.com/) "
    "files to Numpy data files",
    add_help=True, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--data_path",
                    default="../../data/decathlon/Task01_BrainTumour/",
                    help="Path to the raw BraTS datafiles")
parser.add_argument("--save_path",
                    default="../../data/decathlon/",
                    help="Folder to save Numpy data files")
parser.add_argument("--resize", type=int, default=128,
                    help="Resize height and width to this size. "
                    "Original size = 240")
parser.add_argument("--split", type=float, default=0.85,
                    help="Train/test split ratio")

args = parser.parse_args()


def crop_center(img, cropx, cropy, cropz):
    """
    Take a center crop of the images.
    If we are using a 2D model, then we'll just stack the
    z dimension so we can take all slices in that case
    rather than just the crop.
    """
    if len(img.shape) == 4:
        x, y, z, c = img.shape
    else:
        x, y, z = img.shape

    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    startz = z//2-(cropz//2)

    if len(img.shape) == 4:
        # return img[startx:startx+cropx,starty:starty+cropy,startz:startz+cropz,:]
        return img[startx:startx+cropx, starty:starty+cropy, :, :]
    else:
        # return img[startx:startx+cropx,starty:starty+cropy,startz:startz+cropz]
        return img[startx:startx+cropx, starty:starty+cropy, :]


def normalize_img(img):
    """
    Normalize the pixel values.
    This is one of the most important preprocessing steps.
    We need to make sure that the pixel values have a mean of 0
    and a standard deviation of 1 t0 help the model to train
    faster and more accurately.
    """

    for channel in range(img.shape[3]):
        img[:, :, :, channel] = (
            img[:, :, :, channel] - np.mean(img[:, :, :, channel])) \
            / np.std(img[:, :, :, channel])

    return img


def convert_raw_data_to_numpy(trainList, testList, imgList, save_dir):

    # Save training set images
    print("Step 1 of 4. Save training images.")
    first = True
    for idx in tqdm(trainList):

        # nibabel loads images as X,Y,Z,C  (HWDC)
        img = np.array(nib.load(imgList[idx]).dataobj)
        img = crop_center(img, args.resize, args.resize, args.resize)
        img = normalize_img(img)

        if first:
            imgsArray = img
            first = False
        else:
            imgsArray = np.concatenate([imgsArray, img], axis=2)

    np.save(os.path.join(save_dir, "imgs_train.npy"),
            np.swapaxes(imgsArray, 0, -2))

    del imgsArray

    # Save testing set images

    print("Step 2 of 4. Save testing images.")
    first = True
    for idx in tqdm(testList):

        img = np.array(nib.load(imgList[idx]).dataobj)
        img = crop_center(img, args.resize, args.resize, args.resize)
        img = normalize_img(img)

        if first:
            imgsArray = img
            first = False
        else:
            imgsArray = np.concatenate([imgsArray, img], axis=2)

    np.save(os.path.join(save_dir, "imgs_test.npy"),
            np.swapaxes(imgsArray, 0, -2))
    del imgsArray

    # Save training set masks
    print("Step 3 of 4. Save training masks.")
    first = True
    for idx in tqdm(trainList):

        msk = np.array(nib.load(mskList[idx]).dataobj)
        msk = crop_center(msk, args.resize, args.resize, args.resize)

        msk[msk > 1] = 1  # Combine all masks
        msk = np.expand_dims(msk, -1)

        if first:
            msksArray = msk
            first = False
        else:
            msksArray = np.concatenate([msksArray, msk], axis=2)

    np.save(os.path.join(save_dir, "msks_train.npy"),
            np.swapaxes(msksArray, 0, -2))

    del msksArray

    # Save testing set masks
    print("Step 4 of 4. Save testing masks.")
    first = True
    for idx in tqdm(testList):

        msk = np.array(nib.load(mskList[idx]).dataobj)
        msk = crop_center(msk, args.resize, args.resize, args.resize)

        msk[msk > 1] = 1  # Combine all masks
        msk = np.expand_dims(msk, -1)

        if first:
            msksArray = msk
            first = False
        else:
            msksArray = np.concatenate([msksArray, msk], axis=2)

    np.save(os.path.join(save_dir, "msks_test.npy"),
            np.swapaxes(msksArray, 0, -2))

    del msksArray

    print("Finished processing.")
    print("Numpy arrays saved to {}".format(save_dir))


if __name__ == "__main__":

    print("Converting Decathlon raw Nifti data files to training and testing"
          " Numpy data files.")
    print(args)

    save_dir = os.path.join(
        args.save_path, "{}x{}/".format(args.resize, args.resize))

    # Create directory
    try:
        os.makedirs(save_dir)
    except OSError:
        if not os.path.isdir(save_dir):
            raise

    # Check for existing numpy train/test files
    check_dir = os.listdir(save_dir)
    for item in check_dir:
        if item.endswith(".npy"):
            os.remove(os.path.join(save_dir, item))
            print("Removed old version of {}".format(item))

    """
    Get the training file names from the data directory.
    Anything ending in .nii.gz in the imagesTr subdirectory
    is a training file.
    """
    imgList = glob.glob(os.path.join(args.data_path, "imagesTr", "*.nii.gz"))
    mskList = [w.replace("imagesTr", "labelsTr") for w in imgList]

    """
    Randomize the file list. Then separate into training and
    validation (testing) lists.
    """
    numFiles = len(imgList)
    # Set the random seed so that always get same random mix
    np.random.seed(816)
    idxList = np.arange(numFiles)  # List of file indices
    np.random.shuffle(idxList)  # Randomize the file list
    trainList = idxList[:np.int(numFiles*args.split)]
    testList = idxList[np.int(numFiles*args.split):]

    convert_raw_data_to_numpy(trainList, testList, imgList, save_dir)
