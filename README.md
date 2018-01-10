# UNet

UNet architecture for Multimodal Brain Tumor Segmentation, built with TensorFlow 1.4.0 and optimized for single and multi-node execution on Intel KNL and Skylake servers.

## Overview

This repo contains code for single and multi-node execution:

	train.py: Single node operation on Intel KNL or Skylake servers.

	train_dist.py: Multi-node implementation for synchronous weight updates, optimized for use on Intel KNL servers.

## Setup

Note: if running distributed training, the following virtual environment must be present on all workers and PS nodes.

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

## Required Data

Data files are not included in this public repo but can be provided upon request. We use the 2017 BRaTS dataset.

Data is stored in the following numpy files: 

```
imgs_test.npy
imgs_train.npy
msks_test.npy
msks_train.npy
```

In single-node execution, put these files in `/home/unet/data/slices/Results/`. 

For distributed execution, put these files in `/home/unet/data/slices/Results/` on the parameter server. The parameter server is where we will run the distributed training script from.

## Modifications to settings files

Once an environment is constructed which meets the above requirements, clone this repo anywhere on the host machine. For distributed execution, this must be cloned only to the parameter server.

For single-node execution, no changes are needed to `settings.py`.

For multi-node execution, within the cloned directory 'unet', open `settings_dist.py` and replace the current addresses:ports in the `ps_hosts` and `worker_hosts` lists with the appropriate addresses:ports for your cluster.

Depending on your hardware, you may need to modify the NUM_INTRA_THREADS value. This code was developed on Intel KNL and SKL servers having 68 and 56 cores each, so intra-op thread values of 57 and 50 were most ideal. Please note that maxing out the NUM_INTRA_THREADS value may result in segmentation faults or other memory issues.

Note that a natural consequence of synchronizing updates across several workers is a proportional decrease in the number of weight updates per epoch and slower convergence. To combat this slowdown and reduce the training time in multi-node execution, we default to a large initial learning rate which decays as the model trains. This learning rate is contained in `settings_dist.py`.

In addition to the manually overridable settings in Single-Node execution, we provide the following variables for switching on/off and modulating learning rate decay in Multi-Node execution: 

```
--const_learningrate # Bool, Pass this flag alone if a constant learningrate is desired (default: False)
--decay_steps # Int, Steps taken to decay learningrate by lr_fraction% (default: 150)
--lr_fraction # Float, learningrate's fraction of its original value after decay_steps global steps (default: 0.25)
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

## Multi-Node Execution

We use an Ansible's playbook function to automate Multi-Node execution. This playbook will be run from the parameter server.
To initiate training, enter the command `./run_distributed_training.sh` in this cloned repo.

This command will run the `distributed_train.yml` playbook and initiate the following:

1. Create the `inv.yml` file from the addresses listed in `settings_dist.py`.
2. Synchronize all files from the `unet` directory on the parameter server to the `unet` directories on the workers.
3. Start the parameter server with the following command:

```
Parameter Server:	numactl -p 1 python train_dist.py --job_name="ps" --task_index=0
```

4. Run the `Distributed.sh` bash script on all the workers, which executes a run command on each worker:

```
Worker 0:	numactl -p 1 python train_dist.py --job_name="worker" --task_index=0
Worker 1:	numactl -p 1 python train_dist.py --job_name="worker" --task_index=1
Worker 2:	numactl -p 1 python train_dist.py --job_name="worker" --task_index=2
Worker 3:	numactl -p 1 python train_dist.py --job_name="worker" --task_index=3
```

5. While these commands are running, ansible registers their outputs (global step, training loss, dice score, etc.) and saves that to `training.log`. 

To view training progress, as well as sets of images, predictions, and ground truth masks, direct your chrome browser to `http://your_parameter_servers_address:6006/`. After a few moments, the webpage will populate and a series of training visualizations will become available. Explore the Scalars, Images, Graphs, Distributions, and Histograms tabs for detailed visualizations of training progress.

If you have not yet created an ssh tunnel between your local machine and PS, you may not be able to connect to the PS's tensorboard. Run the following command on your local machine, replacing `lancelot` with your cluster's name:

```
ssh -f lancelot -L 6006:localhost:6006 -N
```









