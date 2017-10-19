import os

num_threads = 50

os.environ["KMP_BLOCKTIME"] = "infinite" # This seems to keep processors at max usage
os.environ["KMP_AFFINITY"]="granularity=thread,compact,1,0"
os.environ["OMP_NUM_THREADS"]= "{}".format(num_threads)
os.environ["TF_ADJUST_HUE_FUSED"] = '1'
os.environ['TF_ADJUST_SATURATION_FUSED'] = '1'
os.environ['MKL_VERBOSE'] = '1'

os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
os.environ['TF_AUTOTUNE_THRESHOLD'] = '1'


import tensorflow as tf

from keras import backend as K

from keras.callbacks import ModelCheckpoint
from keras.callbacks import History 
from keras.models import Model

from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout, concatenate
#from keras.layers import Convolution3D, MaxPooling3D, UpSampling3D
#from keras.layers import core
from keras.optimizers import Adam

import numpy as np

#from preprocess import * 
#from helper import *
import settings

def dice_coef(y_true, y_pred, smooth = 1. ):
	y_true_f = K.flatten(y_true)
	y_pred_f = K.flatten(y_pred)
	intersection = K.sum(y_true_f * y_pred_f)
	coef = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
	return coef

def dice_coef_loss(y_true, y_pred):
	return -K.log(dice_coef(y_true, y_pred))


def model5_MultiLayer(weights=False, 
	filepath="", 
	img_rows = 224, 
	img_cols = 224, 
	n_cl_in=3,
	n_cl_out=3, 
	dropout=0.2, 
	learning_rate = 0.001,
	print_summary = False):
	""" difference from model: img_rows and cols, order of axis, and concat_axis"""
	
	inputs = Input((img_rows, img_cols, n_cl_in))
	conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', data_format='channels_last')(inputs)
	conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', data_format='channels_last')(conv1)
	pool1 = MaxPooling2D(pool_size=(2, 2), data_format='channels_last')(conv1)

	conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same', data_format='channels_last')(pool1)
	conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same', data_format='channels_last')(conv2)
	pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

	conv3 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same', data_format='channels_last')(pool2)
	conv3 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same', data_format='channels_last')(conv3)
	pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

	conv4 = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same', data_format='channels_last')(pool3)
	conv4 = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same', data_format='channels_last')(conv4)
	pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

	conv5 = Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same', data_format='channels_last')(pool4)
	conv5 = Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same', data_format='channels_last')(conv5)

	up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=3)
	conv6 = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same', data_format='channels_last')(up6)
	conv6 = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(conv6)

	up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=3)
	conv7 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same', data_format='channels_last')(up7)
	conv7 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same', data_format='channels_last')(conv7)

	up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=3)
	conv8 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same', data_format='channels_last')(up8)
	conv8 = Dropout(dropout)(conv8)
	conv8 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same', data_format='channels_last')(conv8)
	
	up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=3)
	conv9 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', data_format='channels_last')(up9)
	conv9 = Dropout(dropout)(conv9)
	conv9 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', data_format='channels_last')(conv9)

	conv10 = Conv2D(filters=n_cl_out, kernel_size=(1, 1), activation='sigmoid', data_format='channels_last')(conv9)
	model = Model(inputs=[inputs], outputs=[conv10])

	
	model.compile(optimizer=Adam(lr=learning_rate),
		loss='binary_crossentropy', #dice_coef_loss,
		metrics=['accuracy'])

	if weights and len(filepath)>0:
		model.load_weights(filepath)

	if print_summary:
		print (model.summary())	

	return model

def load_data(data_path, prefix = "_train"):
	imgs_train = np.load(os.path.join(data_path, 'imgs'+prefix+'.npy'), mmap_mode='r', allow_pickle=False)
	msks_train = np.load(os.path.join(data_path, 'msks'+prefix+'.npy'), mmap_mode='r', allow_pickle=False)

	return imgs_train, msks_train

def update_channels(imgs, msks, input_no=3, output_no=3, mode=1):
	"""
	changes the order or which channels are used to allow full testing. Uses both
	Imgs and msks as input since different things may be done to both
	---
	mode: int between 1-3
	"""

	imgs = imgs.astype('float32')
	msks = msks.astype('float32')

	shp = imgs.shape
	new_imgs = np.zeros((shp[0],shp[1],shp[2],input_no))
	new_msks = np.zeros((shp[0],shp[1],shp[2],output_no))

	if mode==1:
		new_imgs[:,:,:,0] = imgs[:,:,:,2] # flair
		new_msks[:,:,:,0] = msks[:,:,:,0]+msks[:,:,:,1]+msks[:,:,:,2]+msks[:,:,:,3]
		print('-'*10,' Whole tumor', '-'*10)
	elif mode == 2:
		#core (non enhancing)
		new_imgs[:,:,:,0] = imgs[:,:,:,0] # t1 post
		new_msks[:,:,:,0] = msks[:,:,:,3]
		print('-'*10,' Predicing enhancing tumor', '-'*10)
	elif mode == 3:
		#core (non enhancing)
		new_imgs[:,:,:,0] = imgs[:,:,:,1]# t2 post
		new_msks[:,:,:,0] = msks[:,:,:,0]+msks[:,:,:,2]+msks[:,:,:,3]# active core
		print('-'*10,' Predicing active Core', '-'*10)

	else:
		new_msks[:,:,:,0] = msks[:,:,:,0]+msks[:,:,:,1]+msks[:,:,:,2]+msks[:,:,:,3]

	return np.copy(new_imgs), np.copy(new_msks)


def train_and_predict(data_path, img_rows, img_cols, n_epoch, input_no  = 3, output_no = 3,
	fn= "model", mode = 1):
	
	# configuration session
	sess = tf.Session(config=tf.ConfigProto(
	       intra_op_parallelism_threads=num_threads, inter_op_parallelism_threads=5))
	K.set_session(sess)
	K.set_image_dim_ordering('tf')	

	print('-'*30)
	print('Loading and preprocessing train data...')
	print('-'*30)
	imgs_train, msks_train = load_data(data_path,"_train")
	imgs_train, msks_train = update_channels(imgs_train, msks_train, input_no, output_no, 
		mode)

	print('Shape = {}'.format(imgs_train.shape))
	
	print('-'*30)
	print('Loading and preprocessing test data...')
	print('-'*30)
	imgs_test, msks_test = load_data(data_path,"_test")
	imgs_test, msks_test = update_channels(imgs_test, msks_test, input_no, output_no, mode)


	print('-'*30)
	print('Creating and compiling model...')
	print('-'*30)
	model		= model5_MultiLayer(False, False, img_rows, img_cols, input_no,	output_no)
	model_fn	= os.path.join(data_path, fn+'_{epoch:03d}.hdf5')
	print ("Writing model to ", model_fn)

	model_checkpoint = ModelCheckpoint(model_fn, monitor='loss', save_best_only=False) 
	# saves all models when set to False


	print('-'*30)
	print('Fitting model...')
	print('-'*30)
	history = History()
	history = model.fit(imgs_train, msks_train, 
		batch_size=1024, #128, 
		epochs=n_epoch, 
		validation_data = (imgs_test, msks_test),
		verbose=1, 
		callbacks=[model_checkpoint])

	json_fn = os.path.join(data_path, fn+'.json')
	with open(json_fn,'w') as f:
		f.write(model.to_json())


	print('-'*30)
	print('Loading saved weights...')
	print('-'*30)
	epochNo = len(history.history['loss'])-1
	model_fn	= os.path.join(data_path, '{}_{:03d}.hdf5'.format(fn, epochNo))
	model.load_weights(model_fn)

	print('-'*30)
	print('Predicting masks on test data...')
	print('-'*30)
	msks_pred = model.predict(imgs_test, verbose=1)
	print("Done ", epochNo, np.min(msks_pred), np.max(msks_pred))
	np.save(os.path.join(data_path, 'msks_pred.npy'), msks_pred)

	scores = model.evaluate(imgs_test, msks_test, batch_size=128,verbose = 2)
	print ("Evaluation Scores", scores)

if __name__ =="__main__":
	train_and_predict(settings.OUT_PATH, settings.IMG_ROWS/settings.RESCALE_FACTOR, 
		settings.IMG_COLS/settings.RESCALE_FACTOR, 
		settings.EPOCHS, settings.IN_CHANNEL_NO, \
		settings.OUT_CHANNEL_NO, settings.MODEL_FN, settings.MODE)

