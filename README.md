# UNet

UNet architecture for Multimodal Brain Tumor Segmentation, built with TensorFlow 1.4.0 and optimized for single and multi-node execution on Intel KNL and Skylake servers.

## Overview

This repo contains code for single and multi-node execution:

	train.py: Single node operation on Intel KNL or Skylake servers.

	train_dist.py: Multi-node implementation for synchronous weight updates, optimized for use on Intel KNL servers.

Note: If running multi-node training, the following instructions must be completed identically on all nodes in the network. Or, more accurately, the final state of each machine's 'unet' directory must be identical and the virtual environments must all contain the dependencies listed below.

## Required Packages

Intel optimized TensorFlow 1.4.0 for Python 2.7. Install instructions can be found at https://software.intel.com/en-us/articles/intel-optimized-tensorflow-wheel-now-available.

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

Data files are not included in this public repo but can be provided upon request. We use the 2017 BRaTS dataset.

Data is stored in the following numpy files: 

```
imgs_test.npy
imgs_train.npy
msks_test.npy
msks_train.npy
```

Put these files in `/home/bduser/ge_tensorflow/data/slices/Results/`. For distributed execution, data must be in that same location on all worker nodes. The parameter server does not need a copy of the data.

## Modifications - settings.py

Once an environment is constructed which meets the above requirements, clone this repo anywhere on the host machine. For distributed execution, all nodes must have a copy of this repo (both workers and parameter servers). 

For single-node execution, no changes are needed to `settings.py`.

For multi-node execution, within the cloned directory 'unet', open `settings_dist.py` and replace the current addresses:ports in the `ps_hosts` and `worker_hosts` lists with the appropriate addresses:ports for your cluster.

Depending on your hardware, you may need to modify the NUM_INTRA_THREADS value. This code was developed on Intel KNL and SKL servers having 68 and 56 cores each, so an intra-op thread values of 57 and 50 was most ideal. Please note that maxing out the NUM_INTRA_THREADS value may result in segmentation faults or other memory issues.

## Single-Node Execution

We use numactl to execute the python script. In the 'unet' directory, execute the following command:

```
numactl -p 1 python train.py
```

Updates on training progress will be printed to stdout. This script saves the model to a checkpoint every 60 seconds. The saved model will also be located in the local 'unet' directory.

Default settings can be overridden by appending the above command with:

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

## Multi-Node Execution

Similarly to the Single-Node case, we use numactl to execute the distributed python script. A version of the following example command must be run on each machine to initiate distributed training:

```
numactl -p 1 python train_dist.py --job_name="worker" --task_index=0
```

For parameter servers, `--jobname` must be set to `"ps"`. If there are multiple workers or multiple parameter servers, the `--task_index` argument must be set to the corresponding worker or parameter server number. For example, on a system with 1 parameter server and 4 worker nodes, the following commands would be executed on each machine:

```
Parameter Server: numactl -p 1 python train_dist.py --job_name="ps" --task_index=0
Worker 0:         numactl -p 1 python train_dist.py --job_name="worker" --task_index=0
Worker 1:         numactl -p 1 python train_dist.py --job_name="worker" --task_index=1
Worker 2:         numactl -p 1 python train_dist.py --job_name="worker" --task_index=2
Worker 3:         numactl -p 1 python train_dist.py --job_name="worker" --task_index=3
```

A natural consequence of synchronizing updates across several workers is a proportional decrease in the number of weight updates per epoch. To decrease overall training time, we default to a larger initial learning rate and decay it as the model trains. 

In addition to the manually overridable settings in Single-Node execution, we provide the following variables for switching on/off and modulating learning rate decay in Multi-Node execution: 

```
--const_learningrate # Bool, Pass this flag alone if a constant learningrate is desired (default: False)
--decay_steps # Int, Steps taken to decay learningrate by lr_fraction% (default: 150)
--lr_fraction # Float, learningrate's fraction of its original value after decay_steps global steps (default: 0.25)
```

## Important hyperparameters







