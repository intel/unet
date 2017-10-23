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

'''
  Test program to evaluate if MKL is being used for UNet.
  This should produce messages at runtime about MKL, such as:
  MKL_VERBOSE SGEMM(N,N,10,128,128,0x7f3b0effc148,0x7f3a9d088440,10,0x7f3a10203c40,128,0x7f3b0effc150,0x7f3a9d0ce040,10) 41.32us CNR:OFF Dyn:1 FastMM:1 TID:0  NThr:50 WDiv:HOST:+0.000
  MKL_VERBOSE SGEMM(N,T,10,128,128,0x7f3b0f7fd148,0x7f3a9d0d0840,10,0x7f3a10203c40,128,0x7f3b0f7fd150,0x7f3a101f0040,10) 60.00us CNR:OFF Dyn:1 FastMM:1 TID:0  NThr:50 WDiv:HOST:+0.000

  It should also produce a Tensorflow timeline showing
  the execution trace. 
'''

# The timeline trace for TF is saved to this file.
# To view it, run this python script, then load the json file by 
# starting Google Chrome browser and pointing the URI to chrome://trace
# There should be a button at the top left of the graph where
# you can load in this json file.
timeline_filename = 'kerastimeline_unet.json'

import os

omp_threads = 50
intra_threads = 5
os.environ["KMP_BLOCKTIME"] = "1" 
os.environ["KMP_AFFINITY"]="granularity=thread,compact"
os.environ["OMP_NUM_THREADS"]= str(omp_threads)
os.environ['MKL_VERBOSE'] = '1'
#os.environ['KMP_SETTINGS'] = '1'

os.environ["TF_ADJUST_HUE_FUSED"] = '1'
os.environ['TF_ADJUST_SATURATION_FUSED'] = '1'

os.environ['MKL_DYNAMIC']='1'


import tensorflow as tf
import numpy as np
from tqdm import tqdm

sess = tf.Session(config=tf.ConfigProto(
        intra_op_parallelism_threads=omp_threads, inter_op_parallelism_threads=intra_threads))

from keras import backend as K
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout, concatenate

K.set_session(sess)

batch_size = 128
num_samples = 512

img = tf.placeholder(tf.float32, shape=(batch_size, 128, 128, 1))
labels = tf.placeholder(tf.float32, shape=(batch_size, 128, 128, 1))


# Keras layers can be called on TensorFlow tensors:
conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(img)
conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(conv1)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(pool1)
conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(conv2)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

conv3 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(pool2)
conv3 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(conv3)
pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

conv4 = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(pool3)
conv4 = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(conv4)
pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

conv5 = Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same')(pool4)
conv5 = Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same')(conv5)

up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=3)
conv6 = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(up6)
conv6 = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(conv6)

up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=3)
conv7 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(up7)
conv7 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(conv7)

up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=3)
conv8 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(up8)
conv8 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(conv8)

up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=3)
conv9 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(up9)
conv9 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(conv9)

preds = Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid')(conv9)


from keras.objectives import binary_crossentropy
loss = tf.reduce_mean(binary_crossentropy(labels, preds))


train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

# Initialize all variables
init_op = tf.global_variables_initializer()

options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
run_metadata = tf.RunMetadata()
sess.run(init_op, options=options, run_metadata=run_metadata)

batch ={}


batch[0] = np.random.rand(batch_size,128,128,1)
batch[1] = np.random.rand(batch_size,128,128,1)

# Run training loop
with sess.as_default():

	for i in tqdm(range(num_samples//batch_size)):
	    #batch = mnist_data.train.next_batch(50)
	    
	    train_step.run(feed_dict={img: batch[0],
	                              labels: batch[1]})

	from tensorflow.python.client import timeline

	fetched_timeline = timeline.Timeline(run_metadata.step_stats)
	chrome_trace = fetched_timeline.generate_chrome_trace_format()
	with open(timeline_filename, 'w') as f:
	    f.write(chrome_trace)

