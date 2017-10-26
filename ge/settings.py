#!/usr/bin/env python
''' 
----------------------------------------------------------------------------
Copyright 2017 Intel Nervana 
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
----------------------------------------------------------------------------
'''
settings = {}   # Create a settings dictionary. This way we can just pass the dictionary to functions
settings['batch_size'] = 1024
settings['training_epochs'] = 50
settings['display_step'] = 1

settings['BASE'] = '/home/bduser/ge_tensorflow/data/'
settings['OUT_PATH'] = settings['BASE'] + 'slices/Results/'

# The path and file names for the training and testing files
settings['name of training images numpy file'] = settings['OUT_PATH'] + 'imgs_train.npy'
settings['name of training masks numpy file'] = settings['OUT_PATH'] + 'msks_train.npy'
settings['name of testing images numpy file'] = settings['OUT_PATH'] + 'imgs_test.npy'
settings['name of testing masks numpy file'] = settings['OUT_PATH'] + 'msks_test.npy'

settings['IN_CHANNEL_NO'] = 1
settings['OUT_CHANNEL_NO'] = 1


settings['MODEL_FN'] = 'brainWholeTumor' #Name for Mode=1
#MODEL_FN = "brainActiveTumor" #Name for Mode=2
#MODEL_FN = "brainCoreTumor" #Name for Mode=3

#Use flair to identify the entire tumor: test reaches 0.78-0.80: MODE=1
#Use T1 Post to identify the active tumor: test reaches 0.65-0.75: MODE=2
#Use T2 to identify the active core (necrosis, enhancing, non-enh): test reaches 0.5-0.55: MODE=3
settings['MODE'] = 1

# The timeline trace for TF is saved to this file.
# To view it, run this python script, then load the json file by 
# starting Google Chrome browser and pointing the URI to chrome://trace
# There should be a button at the top left of the graph where
# you can load in this json file.
settings['timeline_filename'] = 'tf_timeline_unet.json'

# The model will be saved to this file
settings['savedModelWeightsFileName'] = 'savedModels/unet_model_weights_ge.ckpt'
settings['USE_SAVED_MODEL'] = True  # Start training by loading in a previously trained model

# The predictions for the testing data set will be saved to this file. This way we can compare it to msks_test.npy
settings['test predictions file'] = 'test_predictions.npy'


'''
Settings for KNL
'''
import os

settings['omp_threads'] = 50 # 50 
settings['intra_threads'] = 5 # 5 
os.environ["KMP_BLOCKTIME"] = "0" # 0 - Let KNL figure out the optimal time
os.environ["KMP_AFFINITY"]="granularity=thread,compact,1,0"
os.environ["OMP_NUM_THREADS"]= str(settings['omp_threads'])
#os.environ['MKL_VERBOSE'] = '1'
#os.environ['KMP_SETTINGS'] = '1'  # If true, then it outputs the device settings

#os.environ["TF_ADJUST_HUE_FUSED"] = '1'
#os.environ['TF_ADJUST_SATURATION_FUSED'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'  # Get rid of the AVX, SSE warnings

#os.environ['MKL_DYNAMIC']='1'
