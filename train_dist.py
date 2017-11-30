# To run, must indicate the job_name (worker or ps) and the job number (0=chief)\
# in the command run on each server in the cluster
# Example: numactl -p 1 python train_dist.py --num_threads=50 --num_inter_threads=2\
#			 --batch_size=256 --blocktime=0 --job_name="ps" --task_index=0

# TODO: how do we control intra/interop threads in a distributed environment?
# TODO: experiment with load balancing between servers
# TODO: experiment with residual blocks
# TODO: try the 'waiting on all nodes' parameter to synchronize right off the bat
# TODO: try dilation rate in convolution layers (cannot do for deconv)
# TODO: figure out how to force prepare_or_wait_for_session to make sessions start together

from tensorflow.python.ops.control_flow_ops import with_dependencies
from preprocess import * 
import tensorflow as tf
from helper import *
import numpy as np
import argparse
import settings
import shutil
import timeit
import time
import os
import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--use_upsampling', help='use upsampling instead of transposed convolution',action='store_true', default=False)
parser.add_argument("--num_threads", type=int, default=settings.NUM_INTRA_THREADS, help="the number of threads")
parser.add_argument("--num_inter_threads", type=int, default=settings.NUM_INTER_THREADS, help="the number of interop threads")
parser.add_argument("--blocktime", type=int, default=settings.BLOCKTIME, help="blocktime")
parser.add_argument("--batch_size", type=int, default=512, help="the batch size for training")
parser.add_argument("--job_name",type=str, default="ps",help="either 'ps' or 'worker'")
parser.add_argument("--task_index",type=int, default=0,help="")
args = parser.parse_args()
batch_size = args.batch_size
num_inter_op_threads = args.num_inter_threads

num_threads = args.num_threads
num_intra_op_threads = num_threads

if (args.blocktime > 1000):
	blocktime = 'infinite'
else:
	blocktime = str(args.blocktime)

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'  # Get rid of the AVX, SSE warnings
os.environ["KMP_BLOCKTIME"] = str(blocktime)
os.environ["KMP_AFFINITY"]="granularity=thread,compact,1,0"
os.environ["OMP_NUM_THREADS"]= str(num_threads)
os.environ["TF_ADJUST_HUE_FUSED"] = '1'
os.environ['TF_ADJUST_SATURATION_FUSED'] = '1'
#os.environ['MKL_VERBOSE'] = '1'
os.environ['MKL_DYNAMIC']='1'
os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
os.environ['TF_AUTOTUNE_THRESHOLD'] = '1'
os.environ['MKL_NUM_THREADS'] = str(num_threads)
#os.environ['KMP_SETTINGS'] = '1'  # Show the settins at runtime

# Unset proxy env variable to avoid gRPC errors
del os.environ['http_proxy']
del os.environ['https_proxy']

# os.environ['MIC_ENV_PREFIX'] = 'PHI'
# os.environ['PHI_KMP_AFFINITY'] = 'compact'
# os.environ['PHI_KMP_PLACE_THREADS'] = '60c,3t'
# os.environ['PHI_OMP_NUM_THREADS'] = str(num_threads)

# The timeline trace for TF is saved to this file.
# To view it, run this python script, then load the json file by 
# starting Google Chrome browser and pointing the URI to chrome://trace
# There should be a button at the top left of the graph where
# you can load in this json file.
timeline_filename = 'timeline_ge_unet_{}_{}_{}.json'.format(blocktime, num_threads, num_intra_op_threads)

# Check if train_logs exists, delete if so
logdir = "/tmp/train_logs"
if os.path.isdir(logdir):
	shutil.rmtree(logdir)

# TODO: put all these in Settings file
ps_hosts = settings.PS_HOSTS
worker_hosts = settings.WORKER_HOSTS
learn_rate = 0.001
model_trained_fn = settings.OUT_PATH+"model_trained.hdf5"
trained_model_fn = "trained_model"
fn = "model"
img_rows = settings.IMG_ROWS/settings.RESCALE_FACTOR
img_cols = settings.IMG_COLS/settings.RESCALE_FACTOR
num_epochs = settings.EPOCHS

config = tf.ConfigProto(inter_op_parallelism_threads=num_inter_op_threads,intra_op_parallelism_threads=num_intra_op_threads)

from keras.models import Model,model_from_json,load_model
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.callbacks import History 
from keras import backend as K
import keras

CHANNEL_LAST = True
if CHANNEL_LAST:
	concat_axis = -1
	data_format = 'channels_last'
	K.set_image_dim_ordering('tf')	
else:
	concat_axis = 1
	data_format = 'channels_first'
	K.set_image_dim_ordering('th')	

def model5_MultiLayer(args=None, weights=False, 
	filepath="", 
	img_rows = 224, 
	img_cols = 224, 
	n_cl_in=3,
	n_cl_out=3, 
	dropout=0.2, 
	learning_rate = 0.01,
	print_summary = False):
	""" difference from model: img_rows and cols, order of axis, and concat_axis"""
	
	if args.use_upsampling:
		print ('Using UpSampling2D')
	else:
		print('Using Transposed Deconvolution')

	if CHANNEL_LAST:
		inputs = Input((img_rows, img_cols, n_cl_in), name='Images')
	else:
		inputs = Input((n_cl_in, img_rows, img_cols), name='Images')



	params = dict(kernel_size=(3, 3), activation='relu', 
				  padding='same', data_format=data_format,
				  kernel_initializer='he_uniform') #RandomUniform(minval=-0.01, maxval=0.01, seed=816))

	conv1 = Conv2D(name='conv1a', filters=32, **params)(inputs)
	conv1 = Conv2D(name='conv1b', filters=32, **params)(conv1)
	pool1 = MaxPooling2D(name='pool1', pool_size=(2, 2))(conv1)

	conv2 = Conv2D(name='conv2a', filters=64, **params)(pool1)
	conv2 = Conv2D(name='conv2b', filters=64, **params)(conv2)
	pool2 = MaxPooling2D(name='pool2', pool_size=(2, 2))(conv2)

	conv3 = Conv2D(name='conv3a', filters=128, **params)(pool2)
	conv3 = Dropout(dropout)(conv3) ### Trying dropout layers earlier on, as indicated in the paper
	conv3 = Conv2D(name='conv3b', filters=128, **params)(conv3)
	
	pool3 = MaxPooling2D(name='pool3', pool_size=(2, 2))(conv3)

	conv4 = Conv2D(name='conv4a', filters=256, **params)(pool3)
	conv4 = Dropout(dropout)(conv4) ### Trying dropout layers earlier on, as indicated in the paper
	conv4 = Conv2D(name='conv4b', filters=256, **params)(conv4)
	
	pool4 = MaxPooling2D(name='pool4', pool_size=(2, 2))(conv4)

	conv5 = Conv2D(name='conv5a', filters=512, **params)(pool4)
	

	if args.use_upsampling:
		conv5 = Conv2D(name='conv5b', filters=256, **params)(conv5)
		up6 = concatenate([UpSampling2D(name='up6', size=(2, 2))(conv5), conv4], axis=concat_axis)
	else:
		conv5 = Conv2D(name='conv5b', filters=512, **params)(conv5)
		up6 = concatenate([Conv2DTranspose(name='transConv6', filters=256, data_format=data_format,
			               kernel_size=(2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=concat_axis)
		
	conv6 = Conv2D(name='conv6a', filters=256, **params)(up6)
	

	if args.use_upsampling:
		conv6 = Conv2D(name='conv6b', filters=128, **params)(conv6)
		up7 = concatenate([UpSampling2D(name='up7', size=(2, 2))(conv6), conv3], axis=concat_axis)
	else:
		conv6 = Conv2D(name='conv6b', filters=256, **params)(conv6)
		up7 = concatenate([Conv2DTranspose(name='transConv7', filters=128, data_format=data_format,
			               kernel_size=(2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=concat_axis)

	conv7 = Conv2D(name='conv7a', filters=128, **params)(up7)
	

	if args.use_upsampling:
		conv7 = Conv2D(name='conv7b', filters=64, **params)(conv7)
		up8 = concatenate([UpSampling2D(name='up8', size=(2, 2))(conv7), conv2], axis=concat_axis)
	else:
		conv7 = Conv2D(name='conv7b', filters=128, **params)(conv7)
		up8 = concatenate([Conv2DTranspose(name='transConv8', filters=64, data_format=data_format,
			               kernel_size=(2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=concat_axis)

	
	conv8 = Conv2D(name='conv8a', filters=64, **params)(up8)
	
	if args.use_upsampling:
		conv8 = Conv2D(name='conv8b', filters=32, **params)(conv8)
		up9 = concatenate([UpSampling2D(name='up9', size=(2, 2))(conv8), conv1], axis=concat_axis)
	else:
		conv8 = Conv2D(name='conv8b', filters=64, **params)(conv8)
		up9 = concatenate([Conv2DTranspose(name='transConv9', filters=32, data_format=data_format,
			               kernel_size=(2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=concat_axis)


	conv9 = Conv2D(name='conv9a', filters=32, **params)(up9)
	conv9 = Conv2D(name='conv9b', filters=32, **params)(conv9)

	conv10 = Conv2D(name='Mask', filters=n_cl_out, kernel_size=(1, 1), 
					data_format=data_format, activation='sigmoid')(conv9)

	model = Model(inputs=[inputs], outputs=[conv10])

	# if weights:
	# 	optimizer=Adam(lr=0.0001, beta_1=0.9, beta_2=0.99, epsilon=1e-08, decay=0.01)
	# else:
	# 	optimizer = SGD(lr=learning_rate, momentum=0.9, decay=0.05)

	optimizer=Adam(lr=learn_rate, beta_1=0.9, beta_2=0.99, epsilon=1e-08, decay=0.00001)

	model.compile(optimizer=optimizer,
		loss=dice_coef_loss, #dice_coef_loss, #'binary_crossentropy', 
		metrics=['accuracy', dice_coef])

	if weights and os.path.isfile(filepath):
		print('Loading model weights from file {}'.format(filepath))
		model.load_weights(filepath)

	if print_summary:
		print (model.summary())	

	return model


def get_model(path):
	with open(path,'r') as f:
		loaded_model_json = f.read()
		model = model_from_json(loaded_model_json)
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

def test_dice_coef(y_true, y_pred, smooth = 1. ):
	y_true_f = K.flatten(y_true)
	y_pred_f = K.flatten(y_pred)
	intersection = K.sum(y_true_f * y_pred_f)
	coef = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
	return coef

def main(_):

	# Create cluster spec from parameter server and worker hosts
	cluster = tf.train.ClusterSpec({"ps":ps_hosts,"worker":worker_hosts})

	# Create and start a server for the local task
	server = tf.train.Server(cluster,job_name=args.job_name,task_index=args.task_index)
	
	# Load train data
	print('-'*30)
	print('Loading and preprocessing train data...')
	print('-'*30)
	imgs_train, msks_train = load_data(settings.OUT_PATH,"_train")
	imgs_train, msks_train = update_channels(imgs_train, msks_train, settings.IN_CHANNEL_NO, settings.OUT_CHANNEL_NO, settings.MODE)

	# Load test data
	print('-'*30)
	print('Loading and preprocessing test data...')
	print('-'*30)
	imgs_test, msks_test = load_data(settings.OUT_PATH,"_test")
	imgs_test, msks_test = update_channels(imgs_test, msks_test, settings.IN_CHANNEL_NO, settings.OUT_CHANNEL_NO, settings.MODE)

	print("Training images shape: {}".format(imgs_train[0].shape))
	print("Training masks shape: {}".format(msks_train[0].shape))

	print('-'*30)
	print('Creating and compiling model...')
	print('-'*30)

	run_metadata = tf.RunMetadata()  # For Tensorflow trace

	if args.job_name == "ps":

		print("Joining server")
		server.join()

	# Train if under worker
	elif args.job_name == "worker":

		# Assign ops to the local worker by default
		with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:{0}".format(args.task_index), cluster=cluster)):
			
			# Set keras learning phase to train
			keras.backend.set_learning_phase(1)

			# Don't initialize variables on the fly
			keras.backend.manual_variable_initialization(False)

			# Create model
			model = model5_MultiLayer(args, False, False, img_rows, img_cols, settings.IN_CHANNEL_NO, settings.OUT_CHANNEL_NO)
			# model_json = os.path.join(settings.OUT_PATH, fn+'.json')
			# print ("Writing model to ", model_json)
			# with open(model_json,'w') as f:
			# 	f.write(model.to_json())

			# # Load keras model in json format
			# model = get_model(settings.OUT_PATH+fn+'.json')

			# Create global_step tensor to count iterations
			# In synchronous training, global step will synchronize after the first few batches in each epoch
			global_step = tf.Variable(0,name='global_step', trainable=False)
			increment_global_step_op = tf.assign(global_step,global_step + 1)

			# Gradients should be computed after moving avg of batchnorm parameters
			# This is to prevent a data-race between barameter updates and moving avg computations
			# Mostly for use with asynchronous updates
			with tf.control_dependencies(model.updates):
				barrier = tf.no_op(name='update_barrier')

			# Synchronize optimizer
			opt = tf.train.AdamOptimizer(learn_rate)
			#opt=tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.9, beta2=0.99, epsilon=1e-08)
			optimizer = tf.train.SyncReplicasOptimizer(opt,replicas_to_aggregate = len(settings.WORKER_HOSTS),total_num_replicas = len(settings.WORKER_HOSTS))

			# Initialize placeholder objects for the loss function
			targ = tf.placeholder(tf.float32, shape=(batch_size,msks_train[0].shape[0],msks_train[0].shape[1],msks_train[0].shape[2]))
			preds = model.output
			loss = dice_coef_loss(targ, preds)

			# Define gradient updates
			with tf.control_dependencies([barrier]):
				grads = optimizer.compute_gradients(loss,model.trainable_weights)
				grad_updates = optimizer.apply_gradients(grads,global_step=global_step)

			# Assign chief queue runner
			chief_queue_runner = optimizer.get_chief_queue_runner()
			init_token_op = optimizer.get_init_tokens_op()

			# Define training operation
			train_op = with_dependencies([grad_updates],loss,name='train')

			# Save model, initialize variables
			saver = tf.train.Saver()
			summary_op = tf.summary.merge_all()
			init_op = tf.global_variables_initializer()

			# Create a "supervisor", which oversees the training process.
			# Cannot modify the graph after this point (it is marked as Final by the Supervisor)
			sv = tf.train.Supervisor(is_chief=(args.task_index == 0),logdir=logdir,init_op=init_op,summary_op=summary_op,saver=saver,global_step=global_step,save_model_secs=600)

			#with sv.prepare_or_wait_for_session(server.target) as sess:
			with sv.managed_session(server.target,config=config) as sess:

				# Bind keras session to the TF session
				K.set_session(sess)

				print('-'*30)
				print("Fitting Model")
				print('-'*30)

				# Start chief queue runner
				# This must be present to coordinate synchronous training
				if args.task_index == 0:
					sv.start_queue_runners(sess, [chief_queue_runner])

				# Run synchronous training
				step = 1
				while step <= num_epochs:

					print("Loading epoch")
					epoch = get_epoch(batch_size,imgs_train,msks_train)
					num_batches = len(epoch)
					current_batch = 1

					epoch_track = []
					epoch_start = timeit.default_timer()

					for batch in epoch:
						batch_start = timeit.default_timer()
						data = batch[0]
						labels = batch[1]
						feed_dict = {model.inputs[0]:data,targ:labels}
						loss_value,step_value = sess.run([train_op,global_step],feed_dict = feed_dict)
						sess.run(increment_global_step_op)

						# Report progress
						dice = "{0:.3f}".format(np.exp(-loss_value))
						loss_show = "{0:.3f}".format(loss_value)
						batch_time = timeit.default_timer()-batch_start
						ETE = str(round(num_batches*(float(batch_time))))[:-2] # Estimated Time per Epoch
						print("Epoch {5}/{6}, ETE: {7} s, Batch: {0}/{1}, Loss:{2}, Dice: {3}, Global Step:{4}".format(current_batch,num_batches,loss_show,dice,sess.run(global_step),step,num_epochs,ETE))
						current_batch += 1

					epoch_time = timeit.default_timer() - epoch_start
					print("Epoch time = {} s\n".format(int(epoch_time)))
					epoch_track.append(epoch_time)


					# Using these savers triggers a "Graph is finalized and cannot be modified" error
					#saver.save(sess,"./{}".format(trained_model_fn)) # POR tensorflow saver
					#model_checkpoint = model.save(model_trained_fn) # hdf5 saver

					step += 1
				print("Average time/epoch = {0} s\n".format(int(np.asarray(epoch_track).mean())))
			sv.stop()

'''
# May use this later for testing
		
			with sv
			# Make predictions on test set
			print('-'*30)
			print("Evaluating test scores")
			print('-'*30)
			image_width = imgs_test.shape[1]
			image_height = imgs_test.shape[2]
			image_channels = imgs_test.shape[3]
			for test_example in zip(imgs_test,msks_test):
				test_image = test_example[0].reshape(1,image_width,image_height,image_channels)
				ground_truth = test_example[1].reshape(1,image_width,image_height,image_channels)
				dice = []
				msk_pred = sess.run(preds,feed_dict={model.inputs[0]:test_image})
				dice.append(dice_coef(ground_truth,msk_pred))
			print("Avg dice score on test set = {}".format(np.asarray(dice).mean()))
'''
if __name__ == "__main__":
	tf.app.run()