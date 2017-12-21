
# These are the only things you need to change.
# Just replace the IP addresses with whatever machines you want to distribute over
# Then run this script on each of those machines.

"""
Usage:  python test_dist.py --ip=10.100.68.245 --is_sync=0
		for asynchronous TF
		python test_dist.py --ip=10.100.68.245 --is_sync=1
		for synchronous updates
		The IP address must match one of the ones in the list below. If not passed,
		then we"ll default to the current machine"s IP (which is usually correct unless you use OPA)
"""
ps_hosts = ["10.100.68.245"]
ps_ports = ["2222"]
worker_hosts = ["10.100.68.193","10.100.68.183"] #,"10.100.68.185","10.100.68.187"]
worker_ports = ["2222", "2222"] #, "2222", "2222"]

ps_list = ["{}:{}".format(x,y) for x,y in zip(ps_hosts, ps_ports)]
worker_list = ["{}:{}".format(x,y) for x,y in zip(worker_hosts, worker_ports)]
print ("Distributed TensorFlow training")
print("Parameter server nodes are: {}".format(ps_list))
print("Worker nodes are {}".format(worker_list))

import settings_dist

CHECKPOINT_DIRECTORY = "checkpoints"
NUM_STEPS = 10000

model_trained_fn = settings_dist.OUT_PATH+"model_trained.hdf5"
trained_model_fn = "trained_model"
fn = "model"
img_rows = settings_dist.IMG_ROWS/settings_dist.RESCALE_FACTOR
img_cols = settings_dist.IMG_COLS/settings_dist.RESCALE_FACTOR
num_epochs = 5
batch_size=128


####################################################################

import numpy as np
import tensorflow as tf
import os
import socket

from preprocess import * 
from helper import *

import multiprocessing

num_inter_op_threads = 2  
num_intra_op_threads = multiprocessing.cpu_count() // 2 # Use half the CPU cores

# Unset proxy env variable to avoid gRPC errors
del os.environ["http_proxy"]
del os.environ["https_proxy"]

# You can turn on the gRPC messages by setting the environment variables below
#os.environ["GRPC_VERBOSITY"]="DEBUG"
#os.environ["GRPC_TRACE"] = "all"

os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"  # Get rid of the AVX, SSE warnings

# Define parameters
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float("learning_rate", 0.0001, "Initial learning rate.")
tf.app.flags.DEFINE_integer("steps_to_validate", 1000,
					 "Validate and print loss after this many steps")
tf.app.flags.DEFINE_integer("is_sync", 0, "Synchronous updates?")
tf.app.flags.DEFINE_string("ip", socket.gethostbyname(socket.gethostname()), "IP address of this machine")

tf.app.flags.DEFINE_boolean("use_upsampling", False, "True=Use upsampling, False=Use Tranposed Convolution")

# Hyperparameters
learning_rate = FLAGS.learning_rate
steps_to_validate = FLAGS.steps_to_validate


if (FLAGS.ip in ps_hosts):
	job_name = "ps"
	task_index = ps_hosts.index(FLAGS.ip)
elif (FLAGS.ip in worker_hosts):
	job_name = "worker"
	task_index = worker_hosts.index(FLAGS.ip)
else:
	print("Error: IP {} not found in the worker or ps node list.\nUse --ip= to specify which machine this is.".format(FLAGS.ip))
	exit()

def create_done_queue(i):
  """
  Queue used to signal termination of the i"th ps shard. 
  Each worker sets their queue value to 1 when done.
  The parameter server op just checks for this.
  """
  
  with tf.device("/job:ps/task:{}".format(i)):
	return tf.FIFOQueue(len(worker_hosts), tf.int32, 
		shared_name="done_queue{}".format(i))
  
def create_done_queues():
  return [create_done_queue(i) for i in range(len(ps_hosts))]


CHANNEL_LAST = True
if CHANNEL_LAST:
	concat_axis = -1
	data_format = 'channels_last'
	
else:
	concat_axis = 1
	data_format = 'channels_first'
	
tf.keras.backend.set_image_data_format(data_format)

def model5_MultiLayer(weights=False, 
	filepath="", 
	img_rows = 224, 
	img_cols = 224, 
	n_cl_in=3,
	n_cl_out=3, 
	dropout=0.2,
	print_summary = False):
	""" difference from model: img_rows and cols, order of axis, and concat_axis"""
	
	if FLAGS.use_upsampling:
		print ('Using UpSampling2D')
	else:
		print('Using Transposed Deconvolution')

	if CHANNEL_LAST:
		inputs = tf.keras.layers.Input((img_rows, img_cols, n_cl_in), name='Images')
	else:
		inputs = tf.keras.layers.Input((n_cl_in, img_rows, img_cols), name='Images')



	params = dict(kernel_size=(3, 3), activation='relu', 
				  padding='same', data_format=data_format,
				  kernel_initializer='he_uniform') #RandomUniform(minval=-0.01, maxval=0.01, seed=816))

	conv1 = tf.keras.layers.Conv2D(name='conv1a', filters=32, **params)(inputs)
	conv1 = tf.keras.layers.Conv2D(name='conv1b', filters=32, **params)(conv1)
	pool1 = tf.keras.layers.MaxPooling2D(name='pool1', pool_size=(2, 2))(conv1)

	conv2 = tf.keras.layers.Conv2D(name='conv2a', filters=64, **params)(pool1)
	conv2 = tf.keras.layers.Conv2D(name='conv2b', filters=64, **params)(conv2)
	pool2 = tf.keras.layers.MaxPooling2D(name='pool2', pool_size=(2, 2))(conv2)

	conv3 = tf.keras.layers.Conv2D(name='conv3a', filters=128, **params)(pool2)
	conv3 = tf.keras.layers.Dropout(dropout)(conv3) ### Trying dropout layers earlier on, as indicated in the paper
	conv3 = tf.keras.layers.Conv2D(name='conv3b', filters=128, **params)(conv3)
	
	pool3 = tf.keras.layers.MaxPooling2D(name='pool3', pool_size=(2, 2))(conv3)

	conv4 = tf.keras.layers.Conv2D(name='conv4a', filters=256, **params)(pool3)
	conv4 = tf.keras.layers.Dropout(dropout)(conv4) ### Trying dropout layers earlier on, as indicated in the paper
	conv4 = tf.keras.layers.Conv2D(name='conv4b', filters=256, **params)(conv4)
	
	pool4 = tf.keras.layers.MaxPooling2D(name='pool4', pool_size=(2, 2))(conv4)

	conv5 = tf.keras.layers.Conv2D(name='conv5a', filters=512, **params)(pool4)
	

	if FLAGS.use_upsampling:
		conv5 = tf.keras.layers.Conv2D(name='conv5b', filters=256, **params)(conv5)
		up6 = tf.keras.layers.concatenate([tf.keras.layers.UpSampling2D(name='up6', size=(2, 2))(conv5), conv4], axis=concat_axis)
	else:
		conv5 = tf.keras.layers.Conv2D(name='conv5b', filters=512, **params)(conv5)
		up6 = tf.keras.layers.concatenate([tf.keras.layers.Conv2DTranspose(name='transConv6', filters=256, data_format=data_format,
						   kernel_size=(2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=concat_axis)
		
	conv6 = tf.keras.layers.Conv2D(name='conv6a', filters=256, **params)(up6)
	

	if FLAGS.use_upsampling:
		conv6 = tf.keras.layers.Conv2D(name='conv6b', filters=128, **params)(conv6)
		up7 = tf.keras.layers.concatenate([tf.keras.layers.UpSampling2D(name='up7', size=(2, 2))(conv6), conv3], axis=concat_axis)
	else:
		conv6 = tf.keras.layers.Conv2D(name='conv6b', filters=256, **params)(conv6)
		up7 = tf.keras.layers.concatenate([tf.keras.layers.Conv2DTranspose(name='transConv7', filters=128, data_format=data_format,
						   kernel_size=(2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=concat_axis)

	conv7 = tf.keras.layers.Conv2D(name='conv7a', filters=128, **params)(up7)
	

	if FLAGS.use_upsampling:
		conv7 = tf.keras.layers.Conv2D(name='conv7b', filters=64, **params)(conv7)
		up8 = tf.keras.layers.concatenate([tf.keras.layers.UpSampling2D(name='up8', size=(2, 2))(conv7), conv2], axis=concat_axis)
	else:
		conv7 = tf.keras.layers.Conv2D(name='conv7b', filters=128, **params)(conv7)
		up8 = tf.keras.layers.concatenate([tf.keras.layers.Conv2DTranspose(name='transConv8', filters=64, data_format=data_format,
						   kernel_size=(2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=concat_axis)

	
	conv8 = tf.keras.layers.Conv2D(name='conv8a', filters=64, **params)(up8)
	
	if FLAGS.use_upsampling:
		conv8 = tf.keras.layers.Conv2D(name='conv8b', filters=32, **params)(conv8)
		up9 = tf.keras.layers.concatenate([tf.keras.layers.UpSampling2D(name='up9', size=(2, 2))(conv8), conv1], axis=concat_axis)
	else:
		conv8 = tf.keras.layers.Conv2D(name='conv8b', filters=64, **params)(conv8)
		up9 = tf.keras.layers.concatenate([tf.keras.layers.Conv2DTranspose(name='transConv9', filters=32, data_format=data_format,
						   kernel_size=(2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=concat_axis)


	conv9 = tf.keras.layers.Conv2D(name='conv9a', filters=32, **params)(up9)
	conv9 = tf.keras.layers.Conv2D(name='conv9b', filters=32, **params)(conv9)

	conv10 = tf.keras.layers.Conv2D(name='Mask', filters=n_cl_out, kernel_size=(1, 1), 
					data_format=data_format, activation='sigmoid')(conv9)

	model = tf.keras.models.Model(inputs=[inputs], outputs=[conv10])

	if weights and os.path.isfile(filepath):
		print('Loading model weights from file {}'.format(filepath))
		model.load_weights(filepath)

	if print_summary:
		print (model.summary())	

	return model

def get_epoch(batch_size,imgs_train,msks_train):

	# Assuming imgs_train and msks_train are the same size
	train_size = imgs_train.shape[0]
	image_width = imgs_train.shape[1]
	image_height = imgs_train.shape[2]
	image_channels = imgs_train.shape[3]

	epoch_length = train_size - train_size%batch_size
	batch_count = epoch_length/batch_size

	# Shuffle and truncate arrays to equal 1 epoch
	zipped = zip(imgs_train,msks_train)
	np.random.shuffle(zipped)
	data,labels = zip(*zipped)
	data = np.asarray(data)[:epoch_length]
	labels = np.asarray(labels)[:epoch_length]

	# Reshape arrays into batch_count batches of length batch_size
	data = data.reshape((batch_count,batch_size,image_width,image_height,image_channels))
	labels = labels.reshape((batch_count,batch_size,image_width,image_height,image_channels))

	# Join batches of training examples with batches of labels
	epoch_of_batches = zip(data,labels)

	return epoch_of_batches


def main(_):

  config = tf.ConfigProto(inter_op_parallelism_threads=num_inter_op_threads,intra_op_parallelism_threads=num_intra_op_threads)

  run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
  run_metadata = tf.RunMetadata()  # For Tensorflow trace

  cluster = tf.train.ClusterSpec({"ps": ps_list, "worker": worker_list})
  server = tf.train.Server(cluster,job_name=job_name,task_index=task_index)

  is_sync = (FLAGS.is_sync == 1)  # Synchronous or asynchronous updates
  is_chief = (task_index == 0)  # Am I the chief node (always task 0)


  if job_name == "ps":

	sess = tf.Session(server.target, config=config)
	queue = create_done_queue(task_index)

	print("\n")
	print("*"*30)
	print("\nParameter server #{} on this machine.\n\n" \
		"Waiting on workers to finish.\n\nPress CTRL-\\ to terminate early." .format(task_index))
	print("*"*30)

	# wait until all workers are done
	for i in range(len(worker_hosts)):
		sess.run(queue.dequeue())
		print("Worker #{} reports job finished." .format(i))
	 
	print("Parameter server #{} is quitting".format(task_index))
	print("Training complete.")

  elif job_name == "worker":

  	# Load train data
	print('-'*30)
	print('Loading and preprocessing train data...')
	print('-'*30)
	imgs_train, msks_train = load_data(settings_dist.OUT_PATH,"_train")
	imgs_train, msks_train = update_channels(imgs_train, msks_train, settings_dist.IN_CHANNEL_NO, settings_dist.OUT_CHANNEL_NO, settings_dist.MODE)

	# Load test data
	print('-'*30)
	print('Loading and preprocessing test data...')
	print('-'*30)
	imgs_test, msks_test = load_data(settings_dist.OUT_PATH,"_test")
	imgs_test, msks_test = update_channels(imgs_test, msks_test, settings_dist.IN_CHANNEL_NO, settings_dist.OUT_CHANNEL_NO, settings_dist.MODE)

	print("Training images shape: {}".format(imgs_train[0].shape))
	print("Training masks shape: {}".format(msks_train[0].shape))

	print('-'*30)
	print('Creating and compiling model...')
	print('-'*30)

	
	if is_chief:
		print("I am chief worker {} with task #{}".format(worker_hosts[task_index], task_index))
	else:
		print("I am worker {} with task #{}".format(worker_hosts[task_index], task_index))

	with tf.device(tf.train.replica_device_setter(
					worker_device="/job:worker/task:{}".format(task_index),
					cluster=cluster)):
	  global_step = tf.Variable(0, name="global_step", trainable=False)

	  """
	  BEGIN: Define our model
	  """
	  # Create model
	  model = model5_MultiLayer(False, False, img_rows, img_cols, settings_dist.IN_CHANNEL_NO, settings_dist.OUT_CHANNEL_NO)
	  # Initialize placeholder objects for the loss function
	  targ = tf.placeholder(tf.float32, shape=((batch_size/len(worker_hosts)),msks_train[0].shape[0],msks_train[0].shape[1],msks_train[0].shape[2]))
	  preds = model.output
	  loss = dice_coef_loss(targ, preds)
	  """
	  END: Define our model
	  """

	  # Define gradient descent optimizer
	  optimizer = tf.train.GradientDescentOptimizer(learning_rate)

	  grads_and_vars = optimizer.compute_gradients(loss_value)
	  if is_sync:
		
		rep_op = tf.train.SyncReplicasOptimizer(optimizer,
			replicas_to_aggregate=len(worker_hosts),
			total_num_replicas=len(worker_hosts),
			use_locking=True)

		train_op = rep_op.apply_gradients(grads_and_vars, global_step=global_step)

		init_token_op = rep_op.get_init_tokens_op()

		chief_queue_runner = rep_op.get_chief_queue_runner()

	  else:
		
		train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)


	  init_op = tf.global_variables_initializer()
	  
	  saver = tf.train.Saver()

	  # These are the values we wish to print to TensorBoard
	  tf.summary.scalar("dice_loss", loss_value)
	  tf.summary.histogram("dice_loss", loss_value)
	  
	# Need to remove the checkpoint directory before each new run
	# import shutil
	# shutil.rmtree(CHECKPOINT_DIRECTORY, ignore_errors=True)

	# Send a signal to the ps when done by simply updating a queue in the shared graph
	enq_ops = []
	for q in create_done_queues():
		qop = q.enqueue(1)
		enq_ops.append(qop)

	# Only the chief does the summary
	if is_chief:
		summary_op = tf.summary.merge_all()
	else:
		summary_op = None

	# TODO:  Theoretically I can pass the summary_op into
	# the Supervisor and have it handle the TensorBoard
	# log entries. However, doing so seems to hang the code.
	# For now, I just handle the summary calls explicitly.
	import time
	sv = tf.train.Supervisor(is_chief=is_chief,
		logdir=CHECKPOINT_DIRECTORY+'/run'+time.strftime("_%Y%m%d_%H%M%S"),
		init_op=init_op,
		summary_op=None, 
		saver=saver,
		global_step=global_step,
		save_model_secs=60)  # Save the model (with weights) everty 60 seconds

	# TODO:
	# I'd like to use managed_session for this as it is more abstract
	# and probably less sensitive to changes from the TF team. However,
	# I am finding that the chief worker hangs on exit if I use managed_session.
	with sv.prepare_or_wait_for_session(server.target, config=config) as sess:
	#with sv.managed_session(server.target) as sess:
	
	  
		if is_chief and is_sync:
			sv.start_queue_runners(sess, [chief_queue_runner])
			sess.run(init_token_op)

		step = 0

		while (not sv.should_stop()) and (step < NUM_STEPS):

			data = batch[0]
			labels = batch[1]

			# For n workers, break up the batch into n sections
			# Send each worker a different section of the batch
			data_range = int(batch_size/len(worker_hosts))
			start = data_range*task_index
			end = start + data_range

			feed_dict = {model.inputs[0]:data[start:end],targ:labels[start:end]}
			loss_value,step_value,learn_rate = sess.run([train_op,global_step,learning_rate],feed_dict = feed_dict)
			
			if (step % steps_to_validate == 0):

			  if (is_chief):

				  summary = sess.run(summary_op, feed_dict=feed_dict)
				  sv.summary_computed(sess, summary)  # Update the summary

	  
		 # Send a signal to the ps when done by simply updating a queue in the shared graph
		for op in enq_ops:
			sess.run(op)   # Send the "work completed" signal to the parameter server
				
	print('Finished work on this node.')
	sv.request_stop()
	#sv.stop()


if __name__ == "__main__":
  tf.app.run()



