# To run, must indicate the job_name (worker or ps) and the job number (0=chief)\
# in the command run on each server in the cluster
# Example: numactl -p 1 python train_dist.py --num_threads=50 --num_inter_threads=2\
#			 --batch_size=256 --blocktime=0 --job_name="ps" --task_index=0

# TODO: try dilation rate in convolution layers (cannot do for deconv)

from tensorflow.python.ops.control_flow_ops import with_dependencies
from preprocess import * 
import tensorflow as tf
from helper import *
import numpy as np
import argparse
import settings_dist
import shutil
import timeit
import time
import os
from tqdm import tqdm   # For the fancy progress bar


parser = argparse.ArgumentParser()
parser.add_argument('--use_upsampling', help='use upsampling instead of transposed convolution',action='store_true', default=False)
parser.add_argument("--num_threads", type=int, default=settings_dist.NUM_INTRA_THREADS, help="the number of threads")
parser.add_argument("--num_inter_threads", type=int, default=settings_dist.NUM_INTER_THREADS, help="the number of interop threads")
parser.add_argument("--blocktime", type=int, default=settings_dist.BLOCKTIME, help="blocktime")
parser.add_argument("--batch_size", type=int, default=settings_dist.BATCH_SIZE, help="the batch size for training")
parser.add_argument("--job_name",type=str, default="ps",help="either 'ps' or 'worker'")
parser.add_argument("--task_index",type=int, default=0,help="")
parser.add_argument("--epochs", type=int, default=settings_dist.EPOCHS, help="number of epochs to train")
parser.add_argument("--learningrate", type=float, default=settings_dist.LEARNINGRATE, help="learningrate")
parser.add_argument("--const_learningrate", help='decay learning rate',action='store_true',default=False)
parser.add_argument("--decay_steps", type=int, default=settings_dist.DECAY_STEPS, help="steps taken to decay learningrate by lr_fraction%")
parser.add_argument("--lr_fraction", type=float, default=settings_dist.LR_FRACTION, help="learningrate's fraction of its original value after decay_steps steps")

parser.add_argument("--worker_nodes",type=str, default=settings_dist.WORKER_HOSTS,help="list of the worker node IP addresses")
parser.add_argument("--ps_nodes",type=str, default=settings_dist.PS_HOSTS,help="list of the parameter server node IP addresses")

args = parser.parse_args()
#global batch_size
batch_size = args.batch_size
num_inter_op_threads = args.num_inter_threads
num_threads = args.num_threads
num_intra_op_threads = num_threads

# Split the test set into test_break batches
test_break = 62

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
# ps_hosts = settings_dist.PS_HOSTS
# worker_hosts = settings_dist.WORKER_HOSTS

ps_hosts = args.ps_nodes
worker_hosts = args.worker_nodes

model_trained_fn = settings_dist.OUT_PATH+"model_trained.hdf5"
trained_model_fn = "trained_model"
fn = "model"
img_rows = settings_dist.IMG_ROWS/settings_dist.RESCALE_FACTOR
img_cols = settings_dist.IMG_COLS/settings_dist.RESCALE_FACTOR
num_epochs = args.epochs

config = tf.ConfigProto(inter_op_parallelism_threads=num_inter_op_threads,intra_op_parallelism_threads=num_intra_op_threads)

run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
run_metadata = tf.RunMetadata()  # For Tensorflow trace

CHANNEL_LAST = True
if CHANNEL_LAST:
	concat_axis = -1
	data_format = 'channels_last'
	
else:
	concat_axis = 1
	data_format = 'channels_first'
	
tf.keras.backend.set_image_data_format(data_format)

def model5_MultiLayer(args=None, weights=False, 
	filepath="", 
	img_rows = 224, 
	img_cols = 224, 
	n_cl_in=3,
	n_cl_out=3, 
	dropout=0.2,
	print_summary = False):
	""" difference from model: img_rows and cols, order of axis, and concat_axis"""
	
	if args.use_upsampling:
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
	

	if args.use_upsampling:
		conv5 = tf.keras.layers.Conv2D(name='conv5b', filters=256, **params)(conv5)
		up6 = tf.keras.layers.concatenate([tf.keras.layers.UpSampling2D(name='up6', size=(2, 2))(conv5), conv4], axis=concat_axis)
	else:
		conv5 = tf.keras.layers.Conv2D(name='conv5b', filters=512, **params)(conv5)
		up6 = tf.keras.layers.concatenate([tf.keras.layers.Conv2DTranspose(name='transConv6', filters=256, data_format=data_format,
			               kernel_size=(2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=concat_axis)
		
	conv6 = tf.keras.layers.Conv2D(name='conv6a', filters=256, **params)(up6)
	

	if args.use_upsampling:
		conv6 = tf.keras.layers.Conv2D(name='conv6b', filters=128, **params)(conv6)
		up7 = tf.keras.layers.concatenate([tf.keras.layers.UpSampling2D(name='up7', size=(2, 2))(conv6), conv3], axis=concat_axis)
	else:
		conv6 = tf.keras.layers.Conv2D(name='conv6b', filters=256, **params)(conv6)
		up7 = tf.keras.layers.concatenate([tf.keras.layers.Conv2DTranspose(name='transConv7', filters=128, data_format=data_format,
			               kernel_size=(2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=concat_axis)

	conv7 = tf.keras.layers.Conv2D(name='conv7a', filters=128, **params)(up7)
	

	if args.use_upsampling:
		conv7 = tf.keras.layers.Conv2D(name='conv7b', filters=64, **params)(conv7)
		up8 = tf.keras.layers.concatenate([tf.keras.layers.UpSampling2D(name='up8', size=(2, 2))(conv7), conv2], axis=concat_axis)
	else:
		conv7 = tf.keras.layers.Conv2D(name='conv7b', filters=128, **params)(conv7)
		up8 = tf.keras.layers.concatenate([tf.keras.layers.Conv2DTranspose(name='transConv8', filters=64, data_format=data_format,
			               kernel_size=(2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=concat_axis)

	
	conv8 = tf.keras.layers.Conv2D(name='conv8a', filters=64, **params)(up8)
	
	if args.use_upsampling:
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

def main(_):

	# Create cluster spec from parameter server and worker hosts
	cluster = tf.train.ClusterSpec({"ps":ps_hosts,"worker":worker_hosts})

	# Create and start a server for the local task
	server = tf.train.Server(cluster,job_name=args.job_name,task_index=args.task_index)

	run_metadata = tf.RunMetadata()  # For Tensorflow trace

	if args.job_name == "ps":

		print("Parameter server started. To interrupt use CTRL-\\")
		server.join()
		

	# Train if under worker
	elif args.job_name == "worker":


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

		# Assign ops to the local worker by default
		with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:{0}".format(args.task_index), cluster=cluster)):
			
			# Set keras learning phase to train
			tf.keras.backend.set_learning_phase(True)

			# Don't initialize variables on the fly
			tf.keras.backend.manual_variable_initialization(False)

			# Create model
			model = model5_MultiLayer(args, False, False, img_rows, img_cols, settings_dist.IN_CHANNEL_NO, settings_dist.OUT_CHANNEL_NO)

			
			# Create global_step tensor to count iterations
			# In synchronous training, global step will synchronize after the first few batches in each epoch
			global_step = tf.Variable(0,name='global_step', trainable=False)
			increment_global_step_op = tf.assign(global_step,global_step + 1)

			# Gradients should be computed after moving avg of batchnorm parameters
			# This is to prevent a data-race between barameter updates and moving avg computations
			# Mostly for use with asynchronous updates
			with tf.control_dependencies(model.updates):
				barrier = tf.no_op(name='update_barrier')

			# Decay learning rate from initial_learn_rate to initial_learn_rate*fraction in decay_steps global steps
			if args.const_learningrate:
				learning_rate = tf.convert_to_tensor(args.learningrate, dtype=tf.float32)
			else:
				initial_learn_rate = args.learningrate
				decay_steps = args.decay_steps
				fraction = args.lr_fraction
				learning_rate = tf.train.exponential_decay(initial_learn_rate, global_step, decay_steps, fraction, staircase=False)

			# Synchronize optimizer
			opt = tf.train.AdamOptimizer(learning_rate)
			#opt=tf.train.AdamOptimizer(learning_rate=args.learning, beta1=0.95, beta2=0.99, epsilon=1e-08)
			optimizer = tf.train.SyncReplicasOptimizer(opt,replicas_to_aggregate = len(settings_dist.WORKER_HOSTS),total_num_replicas = len(settings_dist.WORKER_HOSTS))

			# Initialize placeholder objects for the loss function
			targ = tf.placeholder(tf.float32, shape=((batch_size/len(worker_hosts)),msks_train[0].shape[0],msks_train[0].shape[1],msks_train[0].shape[2]))
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

			# Evaluate test accuracy
			test_label_placeholder = tf.placeholder(tf.float32, shape=(len(msks_test)/test_break,msks_test[0].shape[0],msks_test[0].shape[1],msks_test[0].shape[2]))
			#test_preds = model.output
			test_dice = dice_coef(test_label_placeholder, preds)

			# Save model, initialize variables
			saver = tf.train.Saver()
			summary_op = tf.summary.merge_all()
			init_op = tf.global_variables_initializer()

			# Create a "supervisor", which oversees the training process.
			# Cannot modify the graph after this point (it is marked as Final by the Supervisor)
			print('Am I the chief worker: {}'.format(args.task_index == 0))

			sv = tf.train.Supervisor(is_chief=(args.task_index == 0),logdir=logdir,init_op=init_op,summary_op=summary_op,saver=saver,global_step=global_step,save_model_secs=60)

			#with sv.prepare_or_wait_for_session(server.target) as sess:
			with sv.managed_session(server.target,config=config) as sess:

				# Write to TensorBoard
				train_writer = tf.summary.FileWriter('./tensorboard_logs', sess.graph)

				print('-'*30)
				print("Fitting Model")
				print('-'*30)

				# Start chief queue runner
				# This must be present to coordinate synchronous training
				if args.task_index == 0:
					sv.start_queue_runners(sess, [chief_queue_runner])

				# Run synchronous training
				step = 1

				epoch_track = []

				total_start = timeit.default_timer()

				while (step <= num_epochs) and not sv.should_stop():

					print("Loading epoch")
					epoch = get_epoch(batch_size,imgs_train,msks_train)
					num_batches = len(epoch)
					print('Loaded')
					current_batch = 1

					epoch_start = timeit.default_timer()

					for batch in epoch:#tqdm(epoch):
					
						if sv.should_stop():
							break   # Exit early since the Supervisor node has requested a stop.

						batch_start = timeit.default_timer()
						data = batch[0]
						labels = batch[1]

						# For n workers, break up the batch into n sections
						# Send each worker a different section of the batch
						data_range = int(batch_size/len(worker_hosts))
						start = data_range*args.task_index
						end = start + data_range

						feed_dict = {model.inputs[0]:data[start:end],targ:labels[start:end]}
						loss_value,step_value,learn_rate = sess.run([train_op,global_step,learning_rate],feed_dict = feed_dict)
						#sess.run(increment_global_step_op)

						# Report progress
						dice = "{0:.3f}".format(np.exp(-loss_value))
						loss_show = "{0:.3f}".format(loss_value)
						learn_rate_show = "{0:.6f}".format(learn_rate)
						batch_time = timeit.default_timer()-batch_start
						ETE = str(round(num_batches*(float(batch_time))))[:-2] # Estimated Time per Epoch
						print("Epoch {5}/{6}, ETE: {7} s, Batch: {0}/{1}, Loss:{2}, Dice: {3}, Global Step:{4}, Learn rate:{8}".
							format(current_batch,num_batches,loss_show,dice,sess.run(global_step),step,num_epochs,ETE,learn_rate_show))
						current_batch += 1

						

					epoch_time = timeit.default_timer() - epoch_start
					print("Epoch time = {0} s\nTraining Dice = {1}".format(int(epoch_time),dice))
					epoch_track.append(epoch_time)

					#train_writer.add_summary(summary, step) # Write summary to TensorBoard

					step += 1
					#batch_size -= 20

				total_end = timeit.default_timer()

				# Evaluate test accuracy
				# Break up the test set into smaller sections to avoid segmentation faults
				# Reduce OMP_NUM_THREADS for inference to 'Resource temporarily unavailable errors'
				os.environ["OMP_NUM_THREADS"]= ""
				test_batch_size = len(imgs_test)/test_break
				i = 0
				dice_sum = 0
				while i < test_break:
					start_test = i*test_break
					stop_test = start_test + test_batch_size
					test_image_batch = imgs_test[start_test:stop_test]
					test_label_batch = msks_test[start_test:stop_test]

					test_dict = {model.inputs[0]:test_image_batch,test_label_placeholder:test_label_batch}
					test_dice_coef = sess.run([test_dice],feed_dict=test_dict)
					dice_sum += test_dice_coef[0]

					i += 1

				avg_dice = dice_sum/test_break
				print("Test Dice Coef = {0:.3f}".format(avg_dice))

				print("Average time/epoch = {0} s\n".format(int(np.asarray(epoch_track).mean())))
				print("Total time to train: {} s".format(round(total_end-total_start)))

				

			sv.stop()
				
			del sess

if __name__ == "__main__":

	from datetime import datetime
	print('Starting at {}'.format(datetime.now()))

	tf.app.run()

	print('Ending at {}'.format(datetime.now()))











