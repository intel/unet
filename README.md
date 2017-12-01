# UNet

UNet architecture for Multimodal Brain Tumor Segmentation, built with TensorFlow 1.4.0 and optimized for single and multi-node execution on Intel KNL and Skylake servers.

## Getting Started

This repo contains code for single and multi-node execution:

	train.py: Single node operation on Intel KNL or Skylake servers.

	train_dist.py: Multi-node implementation for synchronous weight updates, optimized for use on Intel KNL servers.

## Requirements

Intel optimized TensorFlow 1.4.0, install instructions can be found at https://software.intel.com/en-us/articles/intel-optimized-tensorflow-wheel-now-available

Additional python packages required:

```
SimpleITK
opencv-python
h5py
shutil
tqdm
```

## TODO

Add notes about running with numactl, keeping intra_op_threads low enough, communicating over OPA, etc.