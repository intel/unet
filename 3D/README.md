# 3D U-Net for Medical Decathlon Dataset

![pred152_3D](https://github.com/IntelAI/unet/blob/master/3D/images/BRATS_152_img3D.gif
"BRATS image #152:  Purple voxels indicate a perfect prediction by the model. Red are false positives. Blue are false negatives").

## Trains a 3D U-Net on the brain tumor segmentation ([BraTS](https://www.med.upenn.edu/sbia/brats2017.html)) subset of the [Medical Segmentation Decathlon dataset](http://medicaldecathlon.com/) dataset. 

This model can achieve a [Dice coefficient](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1415224/) of > 0.85 on the whole tumor using just the [FLAIR](https://en.wikipedia.org/wiki/Fluid-attenuated_inversion_recovery) channel.

### Steps to train a new model:

1. Go to the [Medical Segmentation Decathlon](http://medicaldecathlon.com) website and download the [BraTS subset](https://drive.google.com/file/d/1A2IU8Sgea1h3fYLpYtFb2v7NYdMjvEhU/view?usp=sharing). The dataset has the [Creative Commons Attribution-ShareAlike 4.0 International license](https://creativecommons.org/licenses/by-sa/4.0/).

2. Untar the "Task01_BrainTumour.tar" file:   
```
tar -xvf Task01_BrainTumour.tar
```
3. We use [conda virtual environments](https://www.anaconda.com/distribution/#download-section) to run Python scripts. Once you download and install conda, create a new conda environment with [TensorFlow* with Intel&reg; MKL-DNN](https://software.intel.com/en-us/articles/intel-optimization-for-tensorflow-installation-guide?page=1). Run the command: 
```
conda create -c anaconda -n decathlon pip python=3.6 tensorflow=1.15 keras tqdm h5py psutil
```

This will create a new conda virtual environment called "decathlon" and install [TensorFlow* with Intel&reg; MKL-DNN](https://software.intel.com/en-us/articles/intel-optimization-for-tensorflow-installation-guide) for CPU training and inference.

4. Enable the new environment. Run the command: 
```
conda activate decathlon
```
5. Install the package [nibabel](http://nipy.org/nibabel/). Run the command: 
```
pip install nibabel
```
6. Run the command 
```
python train.py --data_path $DECATHLON_ROOT_DIRECTORY
```
where `$DECATHLON_ROOT_DIRECTORY` is the root directory where you un-tarred the Decathlon dataset.

![commandline](https://github.com/IntelAI/unet/blob/master/3D/images/3d_commandline.png)

NOTE: The default settings take a [Height, Width, Depth] = [144, 144, 144] crop of the original image and mask using 8 images/masks per training batch. This requires over [40 gigabytes](https://github.com/NervanaSystems/topologies/blob/master/3D_UNet/keras_training_only_version/images/training_memory_3d_unet.png) of memory to train the model. We trained our model on an Intel&reg; Xeon&reg; 8180 server with 384 GB of RAM. If you don't have enough memory or are getting out of memory (OOM) errors, you can pass `--patch_height=64 --patch_width=64 --patch_depth=64` to the `train.py` which will use a smaller ([64,64,64]) crop. You can also consider smaller batch sizes (e.g. `--bz=4` for a batch size of 4).

### Steps to evaluate a pre-trained 3D U-Net model.

1. Download the [Medical Decathlon dataset](http://medicaldecathlon.com/). Specifically, this model was trained on the brain tumor segmentation (BraTS 2016 & 2017) portion of the dataset ([Task 1](https://drive.google.com/open?id=1A2IU8Sgea1h3fYLpYtFb2v7NYdMjvEhU)).

2. Untar the Task01_BrainTumour.tar to a folder.

3. Start the model training. Run the inference script:
```
python evaluate_model.py --data_path $DECATHLON_ROOT_DIRECTORY --saved_model $SAVED_HDF5_FILE
```
where `$DECATHLON_ROOT_DIRECTORY` is the root directory where you un-tarred the Decathlon dataset and $SAVE_HDF5_FILE is the name of the pre-trained Keras model.

For example,
```
python evaluate_model.py --data_path ../../data/decathlon/Task01_BrainTumour/ --saved_model 3d_unet_decathlon_dice8621.hdf5
``` 
replacing `--data_path` and `--saved_model` with your local paths/files.

4. The inference script will print the average [Dice coefficient](https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient) of the validation set. It will also save the predictions, MRI images, and ground truth masks for each validation sample into the `predictions_directory` sub-folder. 

### Displaying the Results

There are many programs that will display [Nifti](https://nifti.nimh.nih.gov/) 3D files.  For the images above and below, the red overlay is the prediction from the model and the blue overlay is the ground truth mask. Any purple voxels are true positives.

![pred195](https://github.com/IntelAI/unet/blob/master/3D/images/BRATS_195_img.gif "BRATS image #195:  Purple voxels indicate a perfect prediction by the model. Red are false positives. Blue are false negatives")

 ![pred152](https://github.com/IntelAI/unet/blob/master/3D/images/BRATS_152.png "BRATS image #152:  Purple voxels indicate a perfect prediction by the model. Red are false positives. Blue are false negatives")

 ![pred426](https://github.com/IntelAI/unet/blob/master/3D/images/BRATS_426.png "BRATS image #426:  Purple voxels indicate a perfect prediction by the model. Red are false positives. Blue are false negatives")

## Extra Credit - [Intel&reg; OpenVINO&trade;](https://software.intel.com/en-us/openvino-toolkit) model conversion

To convert the trained model to [Intel&reg; OpenVINO&trade;](https://software.intel.com/en-us/openvino-toolkit):

1. Download and install [Intel&reg; OpenVINO&trade;](https://software.intel.com/en-us/openvino-toolkit) (2019 R2). Start the Intel&reg; OpenVINO&trade; environment: 
```
source /opt/intel/openvino/bin/setupvars.sh
```

2. After training finishes, the `train.py` will save a frozen TensorFlow protobuf for you and give you the command to run the Intel&reg; OpenVINO&trade; model optimizer command to convert the model:

```
python ${INTEL_OPENVINO_DIR}/deployment_tools/model_optimizer/mo_tf.py --input_model tf_protobuf/3d_unet_decathlon.pb --input_shape=[1,144,144,144,1] --data_type FP32  --output_dir openvino_models/FP32  --model_name 3d_unet_decathlon
```

5. The saved model should be located in the `openvino_models/FP32` subfolder.

6. A sample inference script for the OpenVINO version of the model can be found `inference_openvino.py`. It compares the prediction of the Intel&reg; OpenVINO&trade; model with the Keras/TensorFlow* model. There is also a Jupyter Notebook which does the same comparison.

REFERENCES:
1. Menze BH, Jakab A, Bauer S, Kalpathy-Cramer J, Farahani K, Kirby J, Burren Y, Porz N, Slotboom J, Wiest R, Lanczi L, Gerstner E, Weber MA, Arbel T, Avants BB, Ayache N, Buendia P, Collins DL, Cordier N, Corso JJ, Criminisi A, Das T, Delingette H, Demiralp Γ, Durst CR, Dojat M, Doyle S, Festa J, Forbes F, Geremia E, Glocker B, Golland P, Guo X, Hamamci A, Iftekharuddin KM, Jena R, John NM, Konukoglu E, Lashkari D, Mariz JA, Meier R, Pereira S, Precup D, Price SJ, Raviv TR, Reza SM, Ryan M, Sarikaya D, Schwartz L, Shin HC, Shotton J, Silva CA, Sousa N, Subbanna NK, Szekely G, Taylor TJ, Thomas OM, Tustison NJ, Unal G, Vasseur F, Wintermark M, Ye DH, Zhao L, Zhao B, Zikic D, Prastawa M, Reyes M, Van Leemput K. "The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS)", IEEE Transactions on Medical Imaging 34(10), 1993-2024 (2015) DOI: 10.1109/TMI.2014.2377694

2. Bakas S, Akbari H, Sotiras A, Bilello M, Rozycki M, Kirby JS, Freymann JB, Farahani K, Davatzikos C. "Advancing The Cancer Genome Atlas glioma MRI collections with expert segmentation labels and radiomic features", Nature Scientific Data, 4:170117 (2017) DOI: 10.1038/sdata.2017.117

3. Simpson AL, Antonelli M, Bakas S, Bilello M, Farahani K, van Ginneken B, Kopp-Schneider A, Landman BA, Litjens G, Menze B, Ronneberger O, Summers RM, Bilic P, Christ PF, Do RKG, Gollub M, Golia-Pernicka J, Heckers SH, Jarnagin WR, McHugo MK, Napel S, Vorontsov E, Maier-Hein L, Cardoso MJ. "A large annotated medical image dataset for the development and evaluation of segmentation algorithms." https://arxiv.org/abs/1902.09063 

### Optimization notice
Please see our [optimization notice](https://software.intel.com/en-us/articles/optimization-notice#opt-en).

### Architecture
|  lscpu  |   |
| -- | -- | 
| Architecture:   |       x86_64 |
| CPU op-mode(s):  |      32-bit, 64-bit |
| Byte Order:       |     Little Endian |
| CPU(s):            |    56 |
| On-line CPU(s) list: |  0-55 |
| Thread(s) per core:  |  1 |
| Core(s) per socket:  |  28 |
| Socket(s):           |  2 |
| NUMA node(s):        |  2 |
| Vendor ID:          |   GenuineIntel |
| CPU family:          |  6 |
| Model:               |  85 |
| Model name:          |  Intel&reg; Xeon&reg; Platinum 8180 CPU @ 2.50GHz |
| Stepping:           |   4 |
| CPU MHz:             |  999.908 |
| CPU max MHz:          | 2500.0000 |
| CPU min MHz:          | 1000.0000 |
| BogoMIPS:             | 5000.00 |
| Virtualization:       | VT-x |
| L1d cache:            | 32K |
| L1i cache:            | 32K |
| L2 cache:             | 1024K |
| L3 cache:             | 39424K |
| NUMA node0 CPU(s):    | 0-27 |
| NUMA node1 CPU(s):    | 28-55 |
| Flags:                | fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc art arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc aperfmperf eagerfpu pni pclmulqdq dtes64 monitor ds_cpl vmx smx est tm2 ssse3 sdbg fma cx16 xtpr pdcm pcid dca sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand lahf_lm abm 3dnowprefetch epb cat_l3 cdp_l3 intel_ppin intel_pt mba tpr_shadow vnmi flexpriority ept vpid fsgsbase tsc_adjust bmi1 hle avx2 smep bmi2 erms invpcid rtm cqm mpx rdt_a avx512f avx512dq rdseed adx smap clflushopt clwb avx512cd avx512bw avx512vl xsaveopt xsavec xgetbv1 cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local ibpb ibrs dtherm arat pln pts hwp hwp_act_window hwp_epp hwp_pkg_req pku ospke |

