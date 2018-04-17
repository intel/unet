# UNet
test
UNet architecture for Multimodal Brain Tumor Segmentation, built with TensorFlow 1.4.0 and optimized for single and multi-node execution on Intel KNL and Skylake servers.

## Overview

This repo contains code for single-node execution:

	train.py: Single node operation on Intel KNL or Skylake servers.

## Setup

Use conda to setup a virtual environment called 'tf' with the following command:

```
conda create -n tf -c intel python=2 pip numpy
```

This will default the conda environment to use the Intel Python distribution. Use `source activate tf` to enter the virtual environment, then install the following packages:

```
Tensorflow 1.4.0
SimpleITK
opencv-python
h5py
shutil
tqdm
numactl
ansible
```

We use Intel optimized TensorFlow 1.4.0 for Python 2.7. Install instructions can be found at https://software.intel.com/en-us/articles/intel-optimized-tensorflow-wheel-now-available.

Once an environment is constructed which meets the above requirements, clone this repo anywhere on the host machine.

## Required Data

Data files are not included in this public repo but can accessed by registering (using your institutional email address) at the following link: https://www.smir.ch/BRATS/Start2016. Once access has been granted, you may download the raw data. To convert those datasets into numpy arrays having shape [num_images, x_dimension (128), y_dimension (128), num_channels] run `python converter.py` after changing its `root_dir` variable to point to the location your MICCAI_BraTS... folder (processing will take a few minutes). Once complete, the following four files will be saved to /home/unet/data/slices/Results/. 

```
imgs_test.npy
imgs_train.npy
msks_test.npy
msks_train.npy
```

## Single-Node Execution

We use numactl to execute the python script on KNL machines. Note that numa is not available on all Intel servers. To run on a non-KNL server, simply remove the `numactl -p 1` from the below run statement. 

In the 'unet' directory, enter the 'tf' virtual environment and execute the following command:

```
numactl -p 1 python train.py
```

Updates on training progress will be printed to stdout. This script saves the model to a checkpoint every 60 seconds. The saved model will also be located in the local 'unet' directory.

Default settings can be overridden by appending the above command with the following flags:

```
--use_upsampling    # Boolean, Use the UpSampling2D method in place of Conv2DTranspose (default: False)
--num_threads       # Int, Number of intra-op threads (default: 50)
--num_inter_threads # Int, Number of inter-op threads (default: 2)
--batch_size        # Int, Images per batch (default: 128)
--blocktime         # Int, Set KMP_BLOCKTIME environment variable (default: 0)
--epochs            # Int, Number of epochs to train (default: 10)
--learningrate      # Float, Learning rate (default: 0.0001)
```

`numactl -p 1` is used to control how our script will utilize the onboard MCDRAM. The `-p` flag specifies that we prefer using the MCDRAM but, if necessary, are OK expanding into DRAM as needed. Replacing the `-p` with `-m` will force the script to use only MCDRAM. If using the `-m` option, take care to keep the batch size low enough that all training data and network activations will fit in the MCDRAM. If the storage required exceeds that available in MCDRAM, the script will be killed.

## Citations

Whenever using and/or refering to the BraTS datasets in your publications, please make sure to cite the following papers.

1. https://www.ncbi.nlm.nih.gov/pubmed/25494501
2. https://www.ncbi.nlm.nih.gov/pubmed/28872634


