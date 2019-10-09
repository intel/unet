# Intel&reg; [OpenVINO&trade;](https://software.intel.com/en-us/openvino-toolkit) integration (2019 R2)

### How to freeze a saved TensorFlow*/Keras model and convert it to Intel&reg; OpenVINO&trade; format

For more details, please see the [OpenVINO documentation](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_TensorFlow.html) 

1. Convert your Keras model to TensorFlow* saved model fomat.

Use the [tf.train.Saver](https://www.tensorflow.org/api_docs/python/tf/train/Saver) to save the model. The script
[convert_keras_to_tensorflow_serving_model.py](https://github.com/IntelAI/unet/blob/master/2D/helper_scripts/convert_keras_to_tensorflow_serving_model.py) will do this for you and should tell you the correct output_node_names. It defaults to saving the model in the directory `saved_2dunet_model_protobuf`.

```
python helper_scripts/convert_keras_to_tensorflow_serving_model.py --input_filename output/unet_model_for_decathlon.hdf5
```

```
Saving the model to directory saved_2dunet_model_protobuf
TensorFlow protobuf version of model is saved.
Model input name =  MRImages
Model input shape =  (?, 144, 144, 4)
Model output name =  PredictionMask/Sigmoid
Model output shape =  (?, 144, 144, 1)
```
2. Freeze the TensorFlow* saved format model.

This strips any remaining training nodes and turns variables into constants.

If you have installed TensorFlow into the conda environment, then the TensorFlow `freeze_graph` program should be installed in the path. 

```
mkdir frozen_model
freeze_graph --input_saved_model_dir saved_2dunet_model_protobuf --output_node_names PredictionMask/Sigmoid --output_graph frozen_model/saved_model_frozen.pb
```

3. Use the Intel&reg; OpenVINO&trade; model optimizer to convert the frozen TensorFlow* model to Intel&reg; OpenVINO&trade; IR format.

Once you have a frozen model, you can use the Intel&reg; OpenVINO&trade; model optimizer to create the Intel&reg; OpenVINO&trade; version.

Change the directory to `openvino`:

```
cd openvino
```

We've written a bash script to do Intel&reg; OpenVINO&trade; model conversion for you:

```./create_openvino_model.sh```

If you'd like to try it manually, the steps are:

First, set the Intel&reg; OpenVINO&trade; environment (this assumes you've already downloaded and installed [Intel&reg; OpenVINO&trade;](https://software.intel.com/en-us/openvino-toolkit/choose-download) on your computer. These instructions assume version R5 2018):

```
source /opt/intel/openvino/bin/setupvars.sh
```

Then to create a FP32 precision model,

```
python ${INTEL_OPENVINO_DIR}/deployment_tools/model_optimizer/mo_tf.py --input_model ../frozen_model/saved_model_frozen.pb --input_shape=[1,144,144,4] --data_type FP32  --output_dir models/FP32  --model_name saved_model
```

For the Intel&reg; Neural Compute Stick&trade; you'll need to create a FP16 model like this:

```
python ${INTEL_OPENVINO_DIR}/deployment_tools/model_optimizer/mo_tf.py --input_model ../frozen_model/saved_model_frozen.pb --input_shape=[1,144,144,4] --data_type FP16  --output_dir models/FP16  --model_name saved_model
```

4. Run the script `python create_validation_sample.py` which will select a few samples from the HDF5 datafile and save them to a separate NumPy datafile called `validation_data.npz`. The inference scripts will use this NumPy file.

For example:

```python create_validation_sample.py --hdf5_datafile ../../../data/decathlon/Task01_BrainTumour.h5```

5. The scripts `inference_keras.py` and `inference_openvino.py` load the `validation_data.npz` data file and run inference. Add the `--plot` argument to the command line and the script will plot figures for each prediction.

NOTE: The U-Net model can be created with either Transposed Convolution layers or UpSampling2D layers. The UpSampling2D (Bilinear Interpolation) is supported on Intel&reg; OpenVINO&trade; via a shared [TensorFlow* with Intel&reg; MKL-DNN](https://software.intel.com/en-us/articles/intel-optimization-for-tensorflow-installation-guide?page=1) library. To build the library run the script:

```
bash ${INTEL_OPENVINO_DIR}/inference_engine/samples/build_samples.sh
```

This should cause all of the Intel&reg; OpenVINO&trade; shared libraries to be built on your system under the directory `${INTEL_CVSDK_DIR}/inference_engine/lib`. For CPU you'll need to link to `libcpu_extension_avx2.so` or `libcpu_extension_avx512.so`. For example,

```
python inference_openvino.py -l ${INTEL_OPENVINO_DIR}/inference_engine/lib/intel64/libcpu_extension_avx2.so
```

