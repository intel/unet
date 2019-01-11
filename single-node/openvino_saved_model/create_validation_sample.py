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

"""
Gets a sub-sample of the validation set for plotting predictions
"""
import os

import numpy as np
import h5py
import argparse

parser = argparse.ArgumentParser(
    description="Generates NumPy data file with subsample of validation set",
    add_help=True, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--hdf5_datafile", required=True,
                    help="the name and path of the HDF5 dataset")

args = parser.parse_args()

with h5py.File(args.hdf5_datafile, "r") as df:

    indicies_validation = [40,61,400,1100,4385,5566,5673,6433,7864,8899,9003,9722,10591]
    imgs_validation = df["imgs_validation"][indicies_validation,]
    msks_validation = df["msks_validation"][indicies_validation,]

    np.savez("validation_data.npz",
             imgs_validation=imgs_validation,
             msks_validation=msks_validation,
             indicies_validation=indicies_validation)

    print("Created validation_data.npz for sample inference.")
    print("You may now run `python inference_keras.py`")
