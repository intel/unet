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
  UNET entirely written in Tensorflow

  Test program to evaluate if MKL is being used for UNet.
  
  It should also produce a Tensorflow timeline showing
  the execution trace. 
'''

# The timeline trace for TF is saved to this file.
# To view it, run this python script, then load the json file by 
# starting Google Chrome browser and pointing the URI to chrome://trace
# There should be a button at the top left of the graph where
# you can load in this json file.
timeline_filename = 'tf_timeline_unet.json'

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

#os.environ['MKL_DYNAMIC']='1'

import tensorflow as tf
import numpy as np
from tqdm import tqdm

batch_size = 1024 #128
num_samples = 24800 #24800

img_height = 128  # Needs to be an even number for max pooling to work
img_width = 128   # Needs to be an even number for max pooling to work
n_channels = 1

'''
BEGIN 
TF UNET IMPLEMENTATION
'''
img = tf.placeholder(tf.float32, shape=(None, img_height, img_width, n_channels))
labels = tf.placeholder(tf.float32, shape=(None, img_height, img_width, n_channels))


conv1 = tf.layers.conv2d(inputs=img, filters=32, kernel_size=[3, 3], activation=tf.nn.relu, padding='SAME')
conv1 = tf.layers.conv2d(inputs=conv1, filters=32, kernel_size=[3, 3], activation=tf.nn.relu, padding='SAME')
pool1 = tf.layers.max_pooling2d(conv1, 2, 2) # img = 64 x 64 if original size was 128 x 128

conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[3, 3], activation=tf.nn.relu, padding='SAME')
conv2 = tf.layers.conv2d(inputs=conv2, filters=64, kernel_size=[3, 3], activation=tf.nn.relu, padding='SAME')
pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2) # img = 32 x 32 if original size was 128 x 128

conv3 = tf.layers.conv2d(inputs=pool2, filters=128, kernel_size=[3, 3], activation=tf.nn.relu, padding='SAME')
conv3 = tf.layers.conv2d(inputs=conv3, filters=128, kernel_size=[3, 3], activation=tf.nn.relu, padding='SAME')
pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2) # img = 16 x 16 if original size was 128 x 128

conv4 = tf.layers.conv2d(inputs=pool3, filters=256, kernel_size=[3, 3], activation=tf.nn.relu, padding='SAME')
conv4 = tf.layers.conv2d(inputs=conv4, filters=256, kernel_size=[3, 3], activation=tf.nn.relu, padding='SAME')
pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2) #img = 8 x 8 if original size was 128 x 128

conv5 = tf.layers.conv2d(inputs=pool4, filters=512, kernel_size=[3, 3], activation=tf.nn.relu, padding='SAME')
conv5 = tf.layers.conv2d(inputs=conv5, filters=512, kernel_size=[3, 3], activation=tf.nn.relu, padding='SAME')

up6 = tf.concat([tf.image.resize_nearest_neighbor(conv5, (img_height//8, img_width//8)), conv4], -1)
conv6 = tf.layers.conv2d(up6, 256, 3, activation=tf.nn.relu, padding='SAME')
conv6 = tf.layers.conv2d(conv6, 256, 3, activation=tf.nn.relu, padding='SAME')

up7 = tf.concat([tf.image.resize_nearest_neighbor(conv6, (img_height//4, img_width//4)), conv3], -1)
conv7 = tf.layers.conv2d(inputs=up7, filters=128, kernel_size=[3, 3], activation=tf.nn.relu, padding='SAME')
conv7 = tf.layers.conv2d(conv7, 128, 3, activation=tf.nn.relu, padding='SAME')

up8 = tf.concat([tf.image.resize_nearest_neighbor(conv7, (img_height//2, img_width//2)), conv2], -1)
conv8 = tf.layers.conv2d(inputs=up8, filters=64, kernel_size=[3, 3], activation=tf.nn.relu, padding='SAME')
conv8 = tf.layers.conv2d(inputs=conv8, filters=64, kernel_size=[3, 3], activation=tf.nn.relu, padding='SAME')

up9 = tf.concat([tf.image.resize_nearest_neighbor(conv8, (img_height, img_width)), conv1], -1)
conv9 = tf.layers.conv2d(inputs=up9, filters=32, kernel_size=[3, 3], activation=tf.nn.relu, padding='SAME')
conv9 = tf.layers.conv2d(inputs=conv1, filters=32, kernel_size=[3, 3], activation=tf.nn.relu, padding='SAME')

preds = tf.layers.conv2d(inputs=conv9, filters=1, kernel_size=[1,1], activation=None, padding='SAME')

loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=preds))


train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

# Initialize all variables
init_op = tf.global_variables_initializer()

options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
run_metadata = tf.RunMetadata()

sess = tf.Session(config=tf.ConfigProto(
        intra_op_parallelism_threads=omp_threads, inter_op_parallelism_threads=intra_threads))

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

