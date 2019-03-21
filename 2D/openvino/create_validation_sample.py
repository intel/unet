#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Intel Corporation
#
# This program is free software: you can redistribute it and/or modify  
# it under the terms of the GNU General Public License as published by  
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but 
# WITHOUT ANY WARRANTY; without even the implied warranty of 
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU 
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License 
# along with this program. If not, see <http://www.gnu.org/licenses/>.
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

    indicies_validation = [40, 61, 400, 1100, 4385,
                           5566, 5673, 6433, 7864, 8899, 9003, 9722, 10591]
    imgs_validation = df["imgs_validation"][indicies_validation, ]
    msks_validation = df["msks_validation"][indicies_validation, ]

    np.savez("data/validation_data.npz",
             imgs_validation=imgs_validation,
             msks_validation=msks_validation,
             indicies_validation=indicies_validation)

    print("Created validation_data.npz for sample inference.")
    print("You may now run `python inference_keras.py`")
