import numpy as np
from preprocess import * 
import settings
from tqdm import tqdm

print('-'*30)
print('Loading and preprocessing train data...')
print('-'*30)
imgs_train, msks_train = load_data(settings.OUT_PATH,"_train")
imgs_train, msks_train = update_channels(imgs_train, msks_train, settings.IN_CHANNEL_NO, \
		settings.OUT_CHANNEL_NO, settings.MODE)

print('-'*30)
print('Loading and preprocessing train data...')
print('-'*30)
imgs_test, msks_test = load_data(settings.OUT_PATH,"_test")
imgs_test, msks_test = update_channels(imgs_test, msks_test, settings.IN_CHANNEL_NO, \
		settings.OUT_CHANNEL_NO, settings.MODE)

for i in tqdm(range(imgs_test.shape[0])):

	np.save('{}test/img{}.npy'.format(settings.OUT_PATH, i), imgs_test[i])
	np.save('{}test/msk{}.npy'.format(settings.OUT_PATH, i), msks_test[i])

