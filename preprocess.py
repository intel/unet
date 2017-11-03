""" To conver dicom images into images needed for keras"""

from __future__ import print_function
import os
import numpy as np
import SimpleITK as sitk
import settings
import cv2
import random
import time

def get_data_from_dir(data_dir):
	"""
	From a given folder (in the Brats2016 folder organization), returns the different 
	volumes corresponding to t1, t1c, f 
	"""
	print ("Loading from", data_dir)
	img_path	= os.path.dirname(data_dir)
	img_dir_fn	= os.path.basename(data_dir)
	t1_fn		= ""
	t1c_fn		= ""
	flair_fn	= ""
	t2_fn		= ""
	truth_fn	= ""

	fldr1_list = os.listdir(data_dir)
	for fldr1 in fldr1_list:
		fldr1_fn = os.path.join(img_path,img_dir_fn, fldr1)
		if os.path.isdir(fldr1_fn): 
			fldr2_list = os.listdir(fldr1_fn)
			for fldr2 in fldr2_list:
				fn, ext = os.path.splitext(fldr2)
				if ext == '.mha':
					protocol_series = fldr1.split('.')[4]
					protocol = protocol_series.split('_')[0]
					if protocol == 'MR':
						series = protocol_series.split('_')[1]
						if series == 'T2':
							t2_fn = os.path.join(img_path,img_dir_fn, fldr1, fldr2)
						if series == 'Flair':
							flair_fn = os.path.join(img_path, img_dir_fn, fldr1, fldr2)
						if series == 'T1c':
							t1c_fn = os.path.join(img_path,img_dir_fn, fldr1, fldr2)
						if series == 'T1':
							t1_fn = os.path.join(img_path,img_dir_fn, fldr1, fldr2)
					else:
						truth_fn = os.path.join(img_path,img_dir_fn, fldr1, fldr2)

	#does the data have all the needed inputs: T1C, T2, Flair and truth, them use
	isComplete = False
	if len(t1c_fn)>0 and len(t1_fn) and len(flair_fn)>0 and len(t2_fn)>0 \
		and len(truth_fn)>0:
		isComplete = True
		print ("  T1 :", os.path.basename(t1_fn))
		print ("  T1c:", os.path.basename(t1c_fn)) 
		print ("  FLr:", os.path.basename(flair_fn)) 
		print ("  T2 :", os.path.basename(t2_fn))
		print ("  Tru:", os.path.basename(truth_fn))

	# Read data
	try:
		t1 = sitk.ReadImage(t1_fn)
	except Exception as e:
		print (e)
		t1 = sitk.Image()
	
	try:
		t1c = sitk.ReadImage(t1c_fn)
	except Exception as e:
		print (e)
		t1c = sitk.Image()
	
	try:
		fl = sitk.ReadImage(flair_fn)
	except Exception as e:
		print (e)
		fl = sitk.Image()

	try:
		t2 = sitk.ReadImage(t2_fn)
	except Exception as e:
		print (e)
		t2 = sitk.Image()
	
	try:
		msk = sitk.ReadImage(truth_fn);
		msk.SetOrigin(t1.GetOrigin())
		msk.SetDirection(t1.GetDirection())
		msk.SetSpacing(t1.GetSpacing())
	except Exception as e:
		print (e)
		msk = sitk.Image()

	return (t1, t1c, fl, t2, msk, isComplete);

def preprocessSITK(img, img_rows, img_cols, resize_factor=1):
	"""
		crops, rescales, does the bias field correction on an sitk image
	----
	Input: sitk image
	Output: sitk image
	"""
	si_img = img.GetSize()
	sp_img = img.GetSpacing()
	
	#crop to the desired size:
	low_boundary	= [int((si_img[0]-img_rows)/2),int((si_img[1]-img_cols)/2), 0]
	upper_boundary	= [int((si_img[0]-img_rows+1)/2),int((si_img[1]-img_cols+1)/2),0]
	pr_img = sitk.Crop(img, low_boundary, upper_boundary)

	if not resize_factor==1:
		pr_img = sitk.Shrink(pr_img,[resize_factor, resize_factor, 1])
		print ("Resizing to", pr_img.GetSize())

	return pr_img

def normalize(img_arr):
	"""
	intensity preprocessing
	"""
	#new_img_arr = (img_arr-np.min(img_arr))/(np.max(img_arr)-np.min(img_arr))*255
	new_img_arr = (img_arr-np.mean(img_arr))/np.std(img_arr)

	return new_img_arr

def create_datasets_4(img_path, img_rows, img_cols, img_slices, slice_by=5, resize_factor = 1, out_path='.'):
	"""
	creates training with 4 Inputs, and 5 outputs (1-necrosis,2-edema, 
	3-non-enhancing-tumor, 4-enhancing tumore, 5 - rest brain)
	"""

	img_list = os.listdir(img_path)

	slices_per_case = 155
	n_labels = 4
	n_inputs = 4

	img_rows_ss = img_rows/resize_factor
	img_cols_ss = img_cols/resize_factor

	#training
	tr_n_cases = 273 # max number of cases in tcia
	tr_n_slices = slices_per_case*tr_n_cases
	tr_label_counts = np.zeros(n_labels+2)

	tr_img_shape = (tr_n_slices, img_rows_ss, img_cols_ss, n_inputs)
	tr_msk_shape = (tr_n_slices, img_rows_ss, img_cols_ss, n_labels)

	tr_imgs = np.ndarray(tr_img_shape, dtype=np.float)
	tr_msks = np.ndarray(tr_msk_shape, dtype=np.float)

	#testing
	te_n_cases = 60 
	te_n_slices = slices_per_case*te_n_cases
	te_img_shape = (te_n_slices, img_rows_ss, img_cols_ss, n_inputs)
	te_msk_shape = (te_n_slices, img_rows_ss, img_cols_ss, n_labels)

	te_imgs = np.ndarray(te_img_shape, dtype=np.float)
	te_msks = np.ndarray(te_msk_shape, dtype=np.float)

	i = 0
	print('-'*30)
	print('Creating training images...')
	print('-'*30)
	tr_i		= 0
	te_i		= 0

	slicesTr	= 0
	slicesTe	= 0
	curr_sl_tr	= 0
	curr_sl_te	= 0
	curr_cs_te	= 0


	for i, img_dir_fn in enumerate(img_list):
		data_dir = os.path.join(img_path,img_dir_fn)
		# skip if is not a folder
		if not os.path.isdir(data_dir):
			continue

		# find out which on is in training 
		is_tr = True;
		if i % 5 == 0:
			is_tr = False

		print (i, "Train:", is_tr, "", end='')
		(t1p, t1, fl, t2, msk, isComplete) = get_data_from_dir(data_dir)

		#preprocess:crop and rescale
		t1	= preprocessSITK(t1, img_rows, img_cols, resize_factor)
		t1p	= preprocessSITK(t1p,img_rows, img_cols, resize_factor)
		fl	= preprocessSITK(fl, img_rows, img_cols, resize_factor)
		t2	= preprocessSITK(t2, img_rows, img_cols, resize_factor)
		msk	= preprocessSITK(msk,img_rows, img_cols, resize_factor)

		#preprocess: rescale intensity to 0 mean and 1 standard deviation
		t1Arr  = normalize(sitk.GetArrayFromImage(t1).astype('float'))
		t1pArr = normalize(sitk.GetArrayFromImage(t1p).astype('float'))
		flArr  = normalize(sitk.GetArrayFromImage(fl).astype('float'))
		t2Arr  = normalize(sitk.GetArrayFromImage(t2).astype('float'))

		imgArr = np.zeros((slices_per_case, img_rows_ss, img_cols_ss,n_inputs))	
		imgArr[:,:,:,0]	= t1Arr
		imgArr[:,:,:,1]	= t2Arr 
		imgArr[:,:,:,2]	= flArr
		imgArr[:,:,:,3]	= t1pArr

	
		mskArr = np.zeros((slices_per_case, img_rows_ss, img_cols_ss,n_labels))
		mskArrTmp = sitk.GetArrayFromImage(msk)
		mskArr[:,:,:,0] = (mskArrTmp==1).astype('float')
		mskArr[:,:,:,1] = (mskArrTmp==2).astype('float')
		mskArr[:,:,:,2] = (mskArrTmp==3).astype('float')
		mskArr[:,:,:,3] = (mskArrTmp==4).astype('float')

		n_slice = 0
		minSlice = 0
		maxSlice = slices_per_case
		for curr_slice in range(slices_per_case):#leasionSlices:
			n_slice +=1
			# is slice in training cases, but not used from training,or testin 
			#in the first state
			if n_slice % slice_by == 0:
				print ('.', sep='', end='')
				is_used = True
			else:
				is_used = False

			imgSl = imgArr[curr_slice,:,:,:]
			mskSl = mskArr[curr_slice,:,:,:]

			# set slice
			if is_tr:
				# regular training slices
				if is_used:
					if curr_sl_tr % 2 == 0:
						tr_imgs[curr_sl_tr,:,:,:] = imgSl
						tr_msks[curr_sl_tr,:,:,:] = mskSl
					else: # flip  
						tr_imgs[curr_sl_tr,:,:,:] = cv2.flip(imgSl,1).reshape(imgSl.shape)
						tr_msks[curr_sl_tr,:,:,:] = cv2.flip(mskSl,1).reshape(mskSl.shape)
					curr_sl_tr += 1

			else:
				if is_used:
					te_imgs[curr_sl_te,:,:,:] = imgSl
					te_msks[curr_sl_te,:,:,:] = mskSl
					curr_sl_te += 1

		#new line needed for the ... simple progress bar
		print ('\n')
	

		if is_tr:
			tr_i += 1
			slicesTr += maxSlice - minSlice+1 
		else:
			te_i += 1
			slicesTe += maxSlice - minSlice+1
		

	print('Done loading ',slicesTr, slicesTe, curr_sl_tr, curr_sl_te)

	### just write the actually added slices
	tr_imgs = tr_imgs[0:curr_sl_tr,:,:,:]
	tr_msks = tr_msks[0:curr_sl_tr,:,:,:]

	np.save(os.path.join(out_path,'imgs_train.npy'), tr_imgs)
	np.save(os.path.join(out_path,'msks_train.npy'), tr_msks)

	te_imgs = te_imgs[0:curr_sl_te,:,:,:]
	te_msks = te_msks[0:curr_sl_te,:,:,:]

	np.save(os.path.join(out_path,'imgs_test.npy'),  te_imgs)
	np.save(os.path.join(out_path,'msks_test.npy'),  te_msks)
	
	print('Saving to .npy files done.')
	print('Train ', curr_sl_tr)
	print('Test  ', curr_sl_te)

def load_data(data_path, prefix = "_train"):
	imgs_train = np.load(os.path.join(data_path, 'imgs'+prefix+'.npy'))
	msks_train = np.load(os.path.join(data_path, 'msks'+prefix+'.npy'))

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

	return new_imgs, new_msks

if __name__ == '__main__':

	time_start = time.time()
	
	data_path = settings.DATA_PATH
	out_path  = settings.OUT_PATH

	img_rows  = settings.IMG_ROWS
	img_cols  = settings.IMG_COLS
	img_slices = 1;

	"1 - consider all slices"
	"5 - consider very firth slices - for time purposes"
	slice_by   = settings.SLICE_BY 

	rescale_factor = settings.RESCALE_FACTOR

	#read the data and npy files to make it easy for training
	create_datasets_4(data_path, img_rows,img_cols, img_slices, slice_by, rescale_factor, 
		out_path)

	time_end = time.time()

	print ("Done in", time_end-time_start)

