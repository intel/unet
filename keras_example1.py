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
  Test program to evaluate if MKL is being used.
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
timeline_filename = 'kerastimeline_simpleDense.json'

import os

omp_threads = 50
intra_threads = 2
os.environ["KMP_BLOCKTIME"] = "30" 
os.environ["KMP_AFFINITY"]="granularity=thread,compact,1,0"
os.environ["OMP_NUM_THREADS"]= str(omp_threads)
os.environ['OMP_SCHEDULE'] = 'dynamic'
os.environ['MKL_VERBOSE'] = '1'
os.environ['KMP_SETTINGS'] = '1'  # prints the KNL settings at runtime

os.environ["TF_ADJUST_HUE_FUSED"] = '1'
os.environ['TF_ADJUST_SATURATION_FUSED'] = '1'

import tensorflow as tf

sess = tf.Session(config=tf.ConfigProto(
	intra_op_parallelism_threads=omp_threads, 
	inter_op_parallelism_threads=intra_threads))

from keras import backend as K
K.set_session(sess)

from keras.layers import Dense
from keras.objectives import categorical_crossentropy

import numpy as np

batchSize = 128  # Btach size for network training

'''
Use tensorflow placeholders as the input and label variables for our model.
'''
img = tf.placeholder(tf.float32, shape=(batchSize, 784)) # Images are 28x28x1 (HxWxC)
labels = tf.placeholder(tf.float32, shape=(batchSize, 10))

# Keras layers can be called on TensorFlow tensors.
# This way we are using Keras to define the topology BUT tensorflow is actually
# being used directly to perform the training.
x = Dense(128, activation='relu')(img)  # fully-connected layer with 128 units and ReLU activation
x = Dense(128, activation='relu')(x)
preds = Dense(10, activation='softmax')(x)  # output layer with 10 units and a softmax activation

# Cross Entropy loss is the cost function for the model
loss = tf.reduce_mean(categorical_crossentropy(labels, preds))

# Training minimizes loss with SGD (momentum 0.5)
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

# Initialize all variables
init_op = tf.global_variables_initializer()
sess.run(init_op)

# We'll just hard code the input and label batch to be
# two random arrays. In reality we want to have a generator to
# load in batch data iteratively.
batch ={}
batch[0] = np.random.rand(batchSize, 784)
batch[1] = np.random.rand(batchSize,10)

# Run training loop
with sess.as_default():

	run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
	run_metadata = tf.RunMetadata()

	for i in range(100):
	    #batch = mnist_data.train.next_batch(50)
	    
	    # train_step.run(feed_dict={img: batch[0],
	    #                           labels: batch[1]})

		sess.run(train_step,
		             feed_dict={img: batch[0],
		                          labels: batch[1]},
		             options=run_options,
		             run_metadata=run_metadata)

	'''
	Save the training timeline
	'''
	from tensorflow.python.client import timeline

	fetched_timeline = timeline.Timeline(run_metadata.step_stats)
	chrome_trace = fetched_timeline.generate_chrome_trace_format()
	with open(timeline_filename, 'w') as f:
	    f.write(chrome_trace)



