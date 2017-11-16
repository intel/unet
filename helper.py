""" For helper code """

import numpy as np
import os.path

import tensorflow as tf

from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, Dropout
from keras.layers import Convolution3D, MaxPooling3D, UpSampling3D, concatenate, AtrousConvolution2D
from keras.layers import core
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.initializers import RandomUniform

from keras import backend as K
#K.set_image_dim_ordering('tf')

def f1(y_true, y_pred):
	def recall(y_true, y_pred):
		"""Recall metric.

		Only computes a batch-wise average of recall.

		Computes the recall, a metric for multi-label classification of
		how many relevant items are selected.
		"""
		true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
		possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
		recall = true_positives / (possible_positives + K.epsilon())
		return recall

	def precision(y_true, y_pred):
		"""Precision metric.

		Only computes a batch-wise average of precision.

		Computes the precision, a metric for multi-label classification of
		how many selected items are relevant.
		"""
		true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
		predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
		precision = true_positives / (predicted_positives + K.epsilon())
		return precision
	precision = precision(y_true, y_pred)
	recall = recall(y_true, y_pred)
	return ((precision*recall)/(precision+recall))


def dice_coef(y_true, y_pred, smooth = 1. ):
	y_true_f = K.flatten(y_true)
	y_pred_f = K.flatten(y_pred)
	intersection = K.sum(y_true_f * y_pred_f)
	coef = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
	return coef

def dice_coef_loss(y_true, y_pred):
	return -K.log(dice_coef(y_true, y_pred))

def model5_MultiLayer(args=None, weights=False, 
	filepath="", 
	img_rows = 224, 
	img_cols = 224, 
	n_cl_in=3,
	n_cl_out=3, 
	dropout=0.2, 
	learning_rate = 0.001,
	print_summary = False):
	""" difference from model: img_rows and cols, order of axis, and concat_axis"""
	
	if args.use_upsampling:
		print ('Using UpSampling2D')
	else:
		print('Using Transposed Deconvolution')

	inputs = Input((img_rows, img_cols, n_cl_in), name='Image')

	params = dict(kernel_size=(3, 3), activation='relu', 
				  padding='same', 
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
		up6 = concatenate([UpSampling2D(name='up6', size=(2, 2))(conv5), conv4], axis=-1)
	else:
		conv5 = Conv2D(name='conv5b', filters=512, **params)(conv5)
		up6 = concatenate([Conv2DTranspose(name='transConv6', filters=256, 
			               kernel_size=(2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=-1)
		
	conv6 = Conv2D(name='conv6a', filters=256, **params)(up6)
	

	if args.use_upsampling:
		conv6 = Conv2D(name='conv6b', filters=128, **params)(conv6)
		up7 = concatenate([UpSampling2D(name='up7', size=(2, 2))(conv6), conv3], axis=-1)
	else:
		conv6 = Conv2D(name='conv6b', filters=256, **params)(conv6)
		up7 = concatenate([Conv2DTranspose(name='transConv7', filters=128, 
			               kernel_size=(2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=-1)

	conv7 = Conv2D(name='conv7a', filters=128, **params)(up7)
	

	if args.use_upsampling:
		conv7 = Conv2D(name='conv7b', filters=64, **params)(conv7)
		up8 = concatenate([UpSampling2D(name='up8', size=(2, 2))(conv7), conv2], axis=-1)
	else:
		conv7 = Conv2D(name='conv7b', filters=128, **params)(conv7)
		up8 = concatenate([Conv2DTranspose(name='transConv8', filters=64, 
			               kernel_size=(2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=-1)

	
	conv8 = Conv2D(name='conv8a', filters=64, **params)(up8)
	
	if args.use_upsampling:
		conv8 = Conv2D(name='conv8b', filters=32, **params)(conv8)
		up9 = concatenate([UpSampling2D(name='up9', size=(2, 2))(conv8), conv1], axis=-1)
	else:
		conv8 = Conv2D(name='conv8b', filters=64, **params)(conv8)
		up9 = concatenate([Conv2DTranspose(name='transConv9', filters=32, 
			               kernel_size=(2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=-1)


	conv9 = Conv2D(name='conv9a', filters=32, **params)(up9)
	conv9 = Conv2D(name='conv9b', filters=32, **params)(conv9)

	conv10 = Conv2D(name='Mask', filters=n_cl_out, kernel_size=(1, 1), activation='sigmoid')(conv9)

	model = Model(inputs=[inputs], outputs=[conv10])

	model.compile(optimizer=SGD(lr=learning_rate, momentum=0.9, decay=0.01),
		loss=dice_coef_loss, #'binary_crossentropy', 
		metrics=['accuracy', dice_coef])

	if weights and os.path.isfile(filepath):
		print('Loading model weights from file {}'.format(filepath))
		model.load_weights(filepath)

	if print_summary:
		print (model.summary())	

	return model
