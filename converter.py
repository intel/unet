import os
import nibabel as nib
import numpy.ma as ma
import settings_dist
import numpy as np
from tqdm import tqdm

root_dir = '/home/bduser/data_test/MICCAI_BraTS17_Data_Training'  # Replace with your BraTS data directory
resize = 128  # Final dimension (square), set resize = 0 if no resizing is desired
rotate = 3  # Number of counter-clockwise, 90 degree rotations
save_path = settings_dist.OUT_PATH
train_test_split = 0.85
save_interval = 25

def parse_segments(seg):
	
	# Each channel corresponds to a different region of the tumor, decouple and stack these
	
	msks_parsed = []
	for slice in range(seg.shape[-1]):
		curr = seg[:,:,slice]
		GD = ma.masked_not_equal(curr,4).filled(fill_value=0)
		edema = ma.masked_not_equal(curr,2).filled(fill_value=0)
		necrotic = ma.masked_not_equal(curr,1).filled(fill_value=0)
		none = ma.masked_not_equal(curr,0).filled(fill_value=0)

		msks_parsed.append(np.dstack((none,necrotic,edema,GD)))
	
	# Replace all tumorous areas with 1 (previously marked as 1, 2 or 4)
	mask = np.asarray(msks_parsed)
	mask[mask > 0] = 1

	return mask

def parse_images(img):
	
	slices = []
	for slice in range(img.shape[-1]):
		curr = img[:,:,slice]
		slices.append(curr)
	
	return np.asarray(slices)

def stack_img_slices(mode_track, stack_order):
	
	# Put final image channels in the order listed in stack_order
	
	full_brain = []
	for slice in range(len(mode_track['t1'])):
		current_slice = []
		for mode in stack_order:
			current_slice.append(mode_track[mode][slice,:,:])
		full_brain.append(np.dstack(current_slice))
	
	# Normalize stacked images (inference will not work if this is not performed)
	stack = np.asarray(full_brain)
	stack = (stack - np.mean(stack))/(np.std(stack))

	return stack

def resize_data(dataset, new_size):

	# Test/Train images must be the same size

	start_index = (dataset.shape[1] - new_size)/2
	end_index = dataset.shape[1] - start_index

	if rotate != 0:
		resized = np.rot90(dataset[:, start_index:end_index, start_index:end_index :], rotate, axes=(1,2))
	else:
		resized = dataset[:, start_index:end_index, start_index:end_index :]

	return resized

def save_data(imgs_all, msks_all, split, save_path):
	
	imgs_all = np.asarray(imgs_all)
	msks_all = np.asarray(msks_all)

	# Split entire dataset into train/test sets
	train_size = int(msks_all.shape[0]*split)
	new_imgs_train = imgs_all[0:train_size,:,:,:]
	new_msks_train = msks_all[0:train_size,:,:,:]
	new_imgs_test = imgs_all[train_size:,:,:,:]
	new_msks_test = msks_all[train_size:,:,:,:]

	if os.path.isfile("{}imgs_train.npy".format(save_path)):

		# Open one file at a time (these will be large) and clear buffer immediately after concatenate/save
		
		imgs_train = np.load("{}imgs_train.npy".format(save_path))
		np.save("{}imgs_train.npy".format(save_path), np.concatenate((imgs_train,new_imgs_train), axis = 0))
		imgs_train = []
		
		msks_train = np.load("{}msks_train.npy".format(save_path))
		np.save("{}msks_train.npy".format(save_path), np.concatenate((msks_train,new_msks_train), axis = 0))
		msks_train = []
		
		imgs_test = np.load("{}imgs_test.npy".format(save_path))
		np.save("{}imgs_test.npy".format(save_path), np.concatenate((imgs_test,new_imgs_test), axis = 0))
		imgs_test = []
		
		msks_test = np.load("{}msks_test.npy".format(save_path))
		np.save("{}msks_test.npy".format(save_path), np.concatenate((msks_test,new_msks_test), axis = 0))
		msks_test = []

	else:
		
		np.save("{}imgs_train.npy".format(save_path), new_imgs_train)
		np.save("{}msks_train.npy".format(save_path), new_msks_train)
		np.save("{}imgs_test.npy".format(save_path), new_imgs_test)
		np.save("{}msks_test.npy".format(save_path), new_msks_test)

imgs_all = []
msks_all = []
scan_count = 0

for subdir, dir, files in tqdm(os.walk(root_dir)):

	# Ensure all necessary files are present
	file_root = subdir.split('/')[-1] + "_"
	extension = ".nii.gz"
	img_modes = ["t1","t2","flair","t1ce"]
	need_file = [file_root + mode + extension for mode in img_modes]
	all_there = [(reqd in files) for reqd in need_file]
	if all(all_there):

		mode_track = {mode:[] for mode in img_modes}
		
		for file in files:

			if file.endswith('seg.nii.gz'):
				path = os.path.join(subdir,file)
				msk = np.array(nib.load(path).dataobj)
				parsed = resize_data(parse_segments(msk), resize)
				msks_all.extend(parsed)

			if file.endswith('t1.nii.gz'):
				path = os.path.join(subdir,file)
				img = np.array(nib.load(path).dataobj)
				mode_track['t1'] = resize_data(parse_images(img), resize)

			if file.endswith('t2.nii.gz'):
				path = os.path.join(subdir,file)
				img = np.array(nib.load(path).dataobj)
				mode_track['t2'] = resize_data(parse_images(img), resize)

			if file.endswith('t1ce.nii.gz'):
				path = os.path.join(subdir,file)
				img = np.array(nib.load(path).dataobj)
				mode_track['t1ce'] = resize_data(parse_images(img), resize)

			if file.endswith('flair.nii.gz'):
				path = os.path.join(subdir,file)
				img = np.array(nib.load(path).dataobj)
				mode_track['flair'] = resize_data(parse_images(img), resize)

		scan_count += 1
		imgs_all.extend(np.asarray(stack_img_slices(mode_track,img_modes)))

	if (scan_count%save_interval == 0) & (scan_count != 0):
		print("Total scans processed: {}".format(scan_count))
		save_data(imgs_all, msks_all, train_test_split, save_path)
		imgs_all = []
		msks_all = []

# Save any leftover files - may miss a few at the end if the dataset size changes, this will catch those
if len(imgs_all) > 0:
	save_data(imgs_all, msks_all, train_test_split, save_path)
	print("Total scans processed: {}\nDone.".format(scan_count))
