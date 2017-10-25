import numpy as np
import tensorflow as tf

def update_channels(imgs, msks, **settings):
	"""
	changes the order or which channels are used to allow full testing. Uses both
	Imgs and msks as input since different things may be done to both
	---
	mode: int between 1-3
	"""

	shp = imgs.shape
	new_imgs = np.zeros((shp[0],shp[1],shp[2], settings['IN_CHANNEL_NO']))
	new_msks = np.zeros((shp[0],shp[1],shp[2], settings['OUT_CHANNEL_NO']))

	if settings['MODE']==1:
		new_imgs[:,:,:,0] = imgs[:,:,:,2] # flair
		new_msks[:,:,:,0] = msks[:,:,:,0]+msks[:,:,:,1]+msks[:,:,:,2]+msks[:,:,:,3]
		print('-'*10,' Whole tumor', '-'*10)

	elif settings['MODE'] == 2:
		#core (non enhancing)
		new_imgs[:,:,:,0] = imgs[:,:,:,0] # t1 post
		new_msks[:,:,:,0] = msks[:,:,:,3]
		print('-'*10,' Predicing enhancing tumor', '-'*10)

	elif settings['MODE'] == 3:
		#core (non enhancing)
		new_imgs[:,:,:,0] = imgs[:,:,:,1]# t2 post
		new_msks[:,:,:,0] = msks[:,:,:,0]+msks[:,:,:,2]+msks[:,:,:,3]  # active core
		print('-'*10,' Predicing active Core', '-'*10)

	else:
		new_msks[:,:,:,0] = msks[:,:,:,0]+msks[:,:,:,1]+msks[:,:,:,2]+msks[:,:,:,3]

	return new_imgs.astype(np.float32), new_msks.astype(np.float32)

def LoadandPreprocessData(**settings):
	'''
	Load the training and testing files from the numpy files.
	'''

	imgs_train_file = np.load(settings['name of training images numpy file'])
	msks_train_file = np.load(settings['name of training masks numpy file'])

	imgs_train_data, msks_train_data = update_channels(imgs_train_file, msks_train_file, **settings)

	imgs_test_file = np.load(settings['name of testing images numpy file'])
	msks_test_file = np.load(settings['name of testing masks numpy file'])

	imgs_test_data, msks_test_data = update_channels(imgs_test_file, msks_test_file, **settings)

	print('Number of training samples = {:,}'.format(imgs_train_data.shape[0]))
	print('Number of testing samples = {:,}'.format(imgs_test_file.shape[0]))
	print('Batch size = {:,}'.format(settings['batch_size']))

	return imgs_train_data, msks_train_data, imgs_test_data, msks_test_data

def CreatePlaceholder(imgs_data, msks_data):
	'''
	Determine how large the tensorflow placeholders need to be for the data
	'''

	assert imgs_data.shape == msks_data.shape # Make sure images and masks are same size and count

	img_height = imgs_data.shape[1]  # Needs to be an even number for max pooling to work
	img_width = imgs_data.shape[2]   # Needs to be an even number for max pooling to work
	n_channels = imgs_data.shape[3]
	# assert (img_height % 2) == 0
	# assert (img_width % 2) == 0

	n_channels = imgs_data.shape[3]
	data_shape = (None, img_height, img_width, n_channels)

	imgs_placeholder = tf.placeholder(imgs_data.dtype, shape=data_shape)
	msks_placeholder = tf.placeholder(msks_data.dtype, shape=data_shape)

	return imgs_placeholder, msks_placeholder

