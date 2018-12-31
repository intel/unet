# Intel [OpenVINO](https://software.intel.com/en-us/openvino-toolkit) integration 

### How to freeze a saved TensorFlow/Keras model and convert it to OpenVINO format

Use the tf.train.Saver model to save the model. The script
https://github.com/IntelAI/unet/blob/master/single-node/helper_scripts/convert_keras_to_tensorflow_checkpoint.py
will do this for you and  should tell you the correct output_node_names. It defaults to saving the model in the directory `saved_2dunet_model_protobuf`.

`python helper_scripts/convert_keras_to_tensorflow_serving_model.py --input_filename output/unet_model_for_inference.hdf5`
```
Saving the model to directory saved_2dunet_model_protobuf
TensorFlow protobuf version of model is saved.
Model input name =  MRImages
Model input shape =  (?, 144, 144, 4)
Model output name =  PredictionMask/Sigmoid
Model output shape =  (?, 144, 144, 1)
```

The CONDA_PREFIX should be something like /home/bduser/anaconda3/envs/tf112_mkl_p36.
It refers to where your Conda packages are installed for this environment.
It'd be nice if there were an easier way to find freeze_graph.py

`python ${CONDA_PREFIX}/lib/python3.6/site-packages/tensorflow/python/tools/freeze_graph.py \
       --input_saved_model_dir saved_2dunet_model_protobuf/ \
       --output_node_names "predictionMask/Sigmoid" \
       --output_graph saved_model.pb \ 
       --output_dir frozen_model
`

Once you have a frozen model, you can use the OpenVino model optimizer
to create the OpenVino version.

First, set the OpenVino environment:

`source /opt/intel/computer_vision_sdk/bin/setupvars.sh`

Then,

`python ${INTEL_CVSDK_DIR}/deployment_tools/model_optimizer/mo_tf.py \
       --input_model frozen_model/saved_model.pb \
       --input_shape=[1,144,144,4] \
       --data_type FP32 \
       --output_dir FP32 \
       --model_name saved_model
`
