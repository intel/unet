# UNet

UNet architecture for Multimodal Brain Tumor Segmentation, built with TensorFlow 1.4.0 and optimized for single and multi-node execution on Intel KNL and Skylake servers.

## Overview

This repo contains code for single and multi-node execution:

	train.py: Single node operation on Intel KNL or Skylake servers.

	train_dist.py: Multi-node implementation for synchronous weight updates, optimized for use on Intel KNL servers.

Note: the following instructions must be completed identically on all nodes in the network. Or, more accurately, the final state of each 'unet' directory on all workers/parameter servers must be identical before running the train script.

## Required Packages

Intel optimized TensorFlow 1.4.0 for Python 2.7. Install instructions can be found at https://software.intel.com/en-us/articles/intel-optimized-tensorflow-wheel-now-available

Additional packages required:

```
SimpleITK
opencv-python
h5py
shutil
tqdm
numactl
```

## Required Data

Data files are not included in this public repo but can be provided upon request.

Data is stored in the following numpy files: 

```
imgs_test.npy
imgs_train.npy
msks_test.npy
msks_train.npy
```

Put these files in `/home/bduser/ge_tensorflow/data/slices/Results/` on all worker nodes. The parameter server does not need a copy of the data.

## Modifications - settings.py

Once an environment is constructed which meets the above requirements, clone this repo on each node (both workers and parameter servers). 

Within the cloned directory 'unet', open `settings.py` and replace the current addresses:ports in the `ps_hosts` and `worker_hosts` lists with the appropriate addresses:ports for your cluster.

Depending on your hardware, you may need to modify the NUM_INTRA_THREADS value. This code was developed on Intel KNL servers which have 68 cores each, so an intra-op threads value of 57 was most ideal. Please note that maxing out the NUM_INTRA_THREADS value may result in segmentation faults or other memory issues.

## Training Execution

We use numactl to execute the python scripts. A version of the following command must be run on each machine to initiate distributed training:

```
numactl -p 1 python train_dist.py --job_name="worker" --task_index=1
```

`numactl -p 1` is used to control how our script will utilize the onboard MCDRAM. The `-p` flag specifies that we prefer using the MCDRAM but, if necessary, are OK expanding into DRAM as needed. Replacing the `-p` with `-m` will force the script to use only MCDRAM. If using the `-m` option, be careful to keep the batch size low enough that all training data and network activations will fit in the MCDRAM. If the storage required exceeds that available in MCDRAM the script will be killed.

## Important hyperparameters







