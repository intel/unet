# Intel [OpenVINO](https://software.intel.com/en-us/openvino-toolkit) integration

### How to freeze a saved TensorFlow/Keras model and convert it to OpenVINO format

1. Convert your Keras model to TensorFlow saved model fomat.

Use the tf.train.Saver model to save the model. The script
https://github.com/IntelAI/unet/blob/master/single-node/helper_scripts/convert_keras_to_tensorflow_serving_model.py
will do this for you and  should tell you the correct output_node_names. It defaults to saving the model in the directory `saved_2dunet_model_protobuf`.

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
2. Freeze the TensorFlow saved format model.

This strips any remaining training nodes and turns variables into constants.

The CONDA_PREFIX should be something like /home/bduser/anaconda3/envs/tf112_mkl_p36.
It refers to where your Conda packages are installed for this environment.
It'd be nice if there were an easier way to find freeze_graph.py

```
mkdir frozen_model
python ${CONDA_PREFIX}/lib/python3.6/site-packages/tensorflow/python/tools/freeze_graph.py --input_saved_model_dir saved_2dunet_model_protobuf --output_node_names PredictionMask/Sigmoid --output_graph frozen_model/saved_model_frozen.pb
```

3. Use the OpenVINO model optimizer to convert the frozen TensorFlow model to OpenVINO IR format.

Once you have a frozen model, you can use the OpenVino model optimizer to create the OpenVINO version.

First, set the OpenVINO environment:

```
source /opt/intel/computer_vision_sdk/bin/setupvars.sh
```

Then,

```
python ${INTEL_CVSDK_DIR}/deployment_tools/model_optimizer/mo_tf.py --input_model ../frozen_model/saved_model_frozen.pb --input_shape=[1,144,144,4] --data_type FP32  --output_dir FP32  --model_name saved_model
```

4. Run the script `python create_validation_sample.py` which will select a few samples from the HDF5 datafile and save them to a separate NumPy datafile called `validation_data.npz`. The inference scripts will use this NumPy file.

5. The scripts `inference_keras.py` and `inference_openvino.py` load the `validation_data.npz` data file and run inference. Add the `--plot` argument to the command line and the script will plot figures for each prediction.

NOTE: The baseline model uses UpSampling2D (Bilinear Interpolation). This is supported on OpenVINO via a shared TensorFlow MKL-DNN library. To build the library run the script:

```
bash ${INTEL_CVSDK_DIR}/inference_engine/samples/build_samples.sh
```

This should cause all of the OpenVINO shared libraries to be built on your system under the directory `${INTEL_CVSDK_DIR}/inference_engine/lib`. For CPU you'll need to link to `libcpu_extension_avx2.so`. For example,

```
python inference_openvino.py -l ${INTEL_CVSDK_DIR}/inference_engine/lib/centos_7.4/intel64/libcpu_extension_avx2.so
```

or

```
python inference_openvino.py -l ${INTEL_CVSDK_DIR}/inference_engine/lib/ubuntu_16.04/intel64/libcpu_extension_avx2.so
```

depending on your operating system.
