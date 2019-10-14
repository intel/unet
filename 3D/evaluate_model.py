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

import tensorflow as tf
import numpy as np
import datetime
from tqdm import tqdm
from argparser import args
import nibabel as nib
import os
from dataloader import DataGenerator
from model import unet
if args.keras_api:
    import keras as K
else:
    from tensorflow import keras as K


def dice_score(pred, truth):
    """
    Sorensen Dice score
    Measure of the overlap between the prediction and ground truth masks
    """
    numerator = np.sum(np.round(pred) * truth) * 2.0
    denominator = np.sum(np.round(pred)) + np.sum(truth)

    return numerator / denominator


print("Started script on {}".format(datetime.datetime.now()))

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Get rid of the AVX, SSE warnings
os.environ["OMP_NUM_THREADS"] = str(args.intraop_threads)
os.environ["KMP_BLOCKTIME"] = str(args.blocktime)
os.environ["KMP_AFFINITY"] = "granularity=thread,compact,1,0"

# Optimize CPU threads for TensorFlow
CONFIG = tf.ConfigProto(
    inter_op_parallelism_threads=args.interop_threads,
    intra_op_parallelism_threads=args.intraop_threads)

SESS = tf.Session(config=CONFIG)

K.backend.set_session(SESS)

unet_model = unet(channels_last = True)  # channels first or last

model = K.models.load_model(args.saved_model, custom_objects=unet_model.custom_objects)

print("Loading images and masks from test set")

validation_data_params = {"dim": (args.patch_height, args.patch_width, args.patch_depth),
                          "batch_size": 1,
                          "n_in_channels": args.number_input_channels,
                          "n_out_channels": 1,
                          "train_test_split": args.train_test_split,
                          "augment": False,
                          "shuffle": False, "seed": args.random_seed}
testing_generator = DataGenerator("test", args.data_path,
                                     **validation_data_params)
testing_generator.print_info()

m = model.evaluate_generator(testing_generator, verbose=1,
                             max_queue_size=args.num_prefetched_batches,
                             workers=args.num_data_loaders,
                             use_multiprocessing=False)

print("\n\nTest metrics")
print("============")
for idx, name in enumerate(model.metrics_names):
    print("{} = {:.4f}".format(name, m[idx]))


save_directory = "predictions_directory"
try:
    os.stat(save_directory)
except:
    os.mkdir(save_directory)

print("Predicting masks")

for batch_idx in tqdm(range(testing_generator.num_batches),
                      desc="Predicting on batch"):

    imgs, msks = testing_generator.get_batch(batch_idx)
    fileIDs = testing_generator.get_batch_fileIDs()

    preds = model.predict_on_batch(imgs)

    # Save the predictions as Nifti files so that we can
    # display them on a 3D viewer.
    for idx in tqdm(range(preds.shape[0]), desc="Saving to Nifti file"):

        img = nib.Nifti1Image(imgs[idx, :, :, :, 0], np.eye(4))
        img.to_filename(os.path.join(save_directory,
                                     "{}_img.nii.gz".format(fileIDs[idx])))

        msk = nib.Nifti1Image(msks[idx, :, :, :, 0], np.eye(4))
        msk.to_filename(os.path.join(save_directory,
                                     "{}_msk.nii.gz".format(fileIDs[idx])))

        pred = nib.Nifti1Image(preds[idx, :, :, :, 0], np.eye(4))
        pred.to_filename(os.path.join(save_directory,
                                      "{}_pred.nii.gz".format(fileIDs[idx])))

        print("\n{}, Dice = {:f}".format(fileIDs[idx], dice_score(preds[idx, :, :, :, 0],msks[idx, :, :, :, 0])))

print("\n\n\nModel predictions saved to directory: {}".format(save_directory))
print("Stopped script on {}".format(datetime.datetime.now()))
