from tensorflow.keras.utils import Sequence
import numpy as np
import os
import json
import settings
    
def get_decathlon_filelist(data_path, seed=816, split=0.85):
    """
    Get the paths for the original decathlon files
    """
    json_filename = os.path.join(data_path, "dataset.json")

    try:
        with open(json_filename, "r") as fp:
            experiment_data = json.load(fp)
    except IOError as e:
        raise Exception("File {} doesn't exist. It should be part of the "
              "Decathlon directory".format(json_filename))

    # Print information about the Decathlon experiment data
    print("*" * 30)
    print("=" * 30)
    print("Dataset name:        ", experiment_data["name"])
    print("Dataset description: ", experiment_data["description"])
    print("Tensor image size:   ", experiment_data["tensorImageSize"])
    print("Dataset release:     ", experiment_data["release"])
    print("Dataset reference:   ", experiment_data["reference"])
    print("Dataset license:     ", experiment_data["licence"])  # sic
    print("=" * 30)
    print("*" * 30)

    """
	Randomize the file list. Then separate into training and
	validation lists. We won't use the testing set since we
	don't have ground truth masks for this; instead we'll
	split the validation set into separate test and validation
	sets.
	"""
    # Set the random seed so that always get same random mix
    np.random.seed(seed)
    numFiles = experiment_data["numTraining"]
    idxList = np.arange(numFiles)  # List of file indices
    np.random.shuffle(idxList) # Shuffle the indices to randomize train/test/split
    
    trainIdx = int(np.floor(numFiles*split)) # index for the end of the training files
    trainList = idxList[:trainIdx]

    otherList = idxList[trainIdx:]
    numOther = len(otherList)
    otherIdx = numOther//2  # index for the end of the testing files
    validateList = otherList[:otherIdx]
    testList = otherList[otherIdx:]

    trainFiles = []
    for idx in trainList:
        trainFiles.append(os.path.join(data_path, experiment_data["training"][idx]["label"]))

    validateFiles = []
    for idx in validateList:
        validateFiles.append(os.path.join(data_path, experiment_data["training"][idx]["label"]))

    testFiles = []
    for idx in testList:
        testFiles.append(os.path.join(data_path, experiment_data["training"][idx]["label"]))

    print("Number of training files   = {}".format(len(trainList)))
    print("Number of validation files = {}".format(len(validateList)))
    print("Number of testing files    = {}".format(len(testList)))

    return trainFiles, validateFiles, testFiles


class DatasetGenerator(Sequence):
    """
    TensorFlow Dataset from Python/NumPy Iterator
    """
    
    def __init__(self, filenames, batch_size=8, crop_dim=[240,240], augment=False, seed=816):
        
        import nibabel as nib

        img = np.array(nib.load(filenames[0]).dataobj) # Load the first image
        self.slice_dim = 2  # We'll assume z-dimension (slice) is last
        # Determine the number of slices (we'll assume this is consistent for the other images)
        self.num_slices_per_scan = img.shape[self.slice_dim]  

        # If crop_dim == -1, then don't crop
        if crop_dim[0] == -1:
            crop_dim[0] = img.shape[0]
        if crop_dim[1] == -1:
            crop_dim[1] = img.shape[1]
        self.crop_dim = crop_dim  

        self.filenames = filenames
        self.batch_size = batch_size

        self.augment = augment
        self.seed = seed
        
        self.num_files = len(self.filenames)
        
        self.ds = self.get_dataset()

    def preprocess_img(self, img):
        """
        Preprocessing for the image
        z-score normalize
        """
        return (img - img.mean()) / img.std()

    def preprocess_label(self, label):
        """
        Predict whole tumor. If you want to predict tumor sections, then 
        just comment this out.
        """
        label[label > 0] = 1.0

        return label
    
    def augment_data(self, img, msk):
        """
        Data augmentation
        Flip image and mask. Rotate image and mask.
        """
        
        if np.random.rand() > 0.5:
            ax = np.random.choice([0,1])
            img = np.flip(img, ax)
            msk = np.flip(msk, ax)

        if np.random.rand() > 0.5:
            rot = np.random.choice([1, 2, 3])  # 90, 180, or 270 degrees

            img = np.rot90(img, rot, axes=[0,1])  # Rotate axes 0 and 1
            msk = np.rot90(msk, rot, axes=[0,1])  # Rotate axes 0 and 1

        return img, msk

    def crop_input(self, img, msk):
            """
            Randomly crop the image and mask
            """

            slices = []

            # Do we randomize?
            is_random = self.augment and np.random.rand() > 0.5

            for idx, idy in enumerate(range(2)):  # Go through each dimension

                cropLen = self.crop_dim[idx]
                imgLen = img.shape[idy]

                start = (imgLen-cropLen)//2

                ratio_crop = 0.20  # Crop up this this % of pixels for offset
                # Number of pixels to offset crop in this dimension
                offset = int(np.floor(start*ratio_crop))

                if offset > 0:
                    if is_random:
                        start += np.random.choice(range(-offset, offset))
                        if ((start + cropLen) > imgLen):  # Don't fall off the image
                            start = (imgLen-cropLen)//2
                else:
                    start = 0

                slices.append(slice(start, start+cropLen))

            return img[tuple(slices)], msk[tuple(slices)]

    def generate_batch_from_files(self):
        """
        Python generator which goes through a list of filenames to load.
        The files are 3D image (slice is dimension index 2 by default). However,
        we need to yield them as a batch of 2D slices. This generator
        keeps yielding a batch of 2D slices at a time until the 3D image is 
        complete and then moves to the next 3D image in the filenames.
        An optional `randomize_slices` allows the user to randomize the 3D image 
        slices after loading if desired.
        """
        import nibabel as nib

        np.random.seed(self.seed)  # Set a random seed

        idx = 0
        idy = 0

        while True:

            """
            Pack N_IMAGES files at a time to queue
            """
            NUM_QUEUED_IMAGES = 1 + self.batch_size // self.num_slices_per_scan  # Get enough for full batch + 1
            
            for idz in range(NUM_QUEUED_IMAGES):

                label_filename = self.filenames[idx]

                #img_filename   = label_filename.replace("_seg.nii.gz", "_flair.nii.gz") # BraTS 2018
                img_filename   = label_filename.replace("labelsTr", "imagesTr")  # Medical Decathlon
                
                img = np.array(nib.load(img_filename).dataobj)
                img = img[:,:,:,0]  # Just take FLAIR channel (channel 0)
                img = self.preprocess_img(img)

                label = np.array(nib.load(label_filename).dataobj)
                label = self.preprocess_label(label)
                
                # Crop input and label
                img, label = self.crop_input(img, label)

                if idz == 0:
                    img_stack = img
                    label_stack = label

                else:

                    img_stack = np.concatenate((img_stack,img), axis=self.slice_dim)
                    label_stack = np.concatenate((label_stack,label), axis=self.slice_dim)
                
                idx += 1 
                if idx >= len(self.filenames):
                    idx = 0
                    np.random.shuffle(self.filenames) # Shuffle the filenames for the next iteration
            
            img = img_stack
            label = label_stack

            num_slices = img.shape[self.slice_dim]
            
            if self.batch_size > num_slices:
                raise Exception("Batch size {} is greater than"
                                " the number of slices in the image {}."
                                " Data loader cannot be used.".format(self.batch_size, num_slices))

            """
            We can also randomize the slices so that no 2 runs will return the same slice order
            for a given file. This also helps get slices at the end that would be skipped
            if the number of slices is not the same as the batch order.
            """
            if self.augment:
                slice_idx = np.random.choice(range(num_slices), num_slices)
                img = img[:,:,slice_idx]  # Randomize the slices
                label = label[:,:,slice_idx]

            name = self.filenames[idx]
            
            if (idy + self.batch_size) < num_slices:  # We have enough slices for batch
                img_batch, label_batch = img[:,:,idy:idy+self.batch_size], label[:,:,idy:idy+self.batch_size]   

            else:  # We need to pad the batch with slices

                img_batch, label_batch = img[:,:,-self.batch_size:], label[:,:,-self.batch_size:]  # Get remaining slices

            if self.augment:
                img_batch, label_batch = self.augment_data(img_batch, label_batch)
                
            if len(np.shape(img_batch)) == 3:
                img_batch = np.expand_dims(img_batch, axis=-1)
            if len(np.shape(label_batch)) == 3:
                label_batch = np.expand_dims(label_batch, axis=-1)
                
            yield np.transpose(img_batch, [2,0,1,3]).astype(np.float32), np.transpose(label_batch, [2,0,1,3]).astype(np.float32)


            idy += self.batch_size
            if idy >= num_slices: # We finished this file, move to the next
                idy = 0
                idx += 1

            if idx >= len(self.filenames):
                idx = 0
                np.random.shuffle(self.filenames) # Shuffle the filenames for the next iteration
                

    def get_input_shape(self):
        """
        Get image shape
        """
        return [self.crop_dim[0], self.crop_dim[1], 1]
        
    def get_output_shape(self):
        """
        Get label shape
        """
        return [self.crop_dim[0], self.crop_dim[1], 1] 
    
    def get_dataset(self):
        """
        Return a dataset
        """
        ds = self.generate_batch_from_files()
        
        return ds  
    
    def __len__(self):
        return (self.num_slices_per_scan * self.num_files)//self.batch_size

    def __getitem__(self, idx):
        return next(self.ds)
        
    def plot_samples(self):
        """
        Plot some random samples
        """
        import matplotlib.pyplot as plt
        
        img, label = next(self.ds)
        
        print(img.shape)
 
        plt.figure(figsize=(10,10))
        
        slice_num = 3
        plt.subplot(2,2,1)
        plt.imshow(img[slice_num,:,:,0]);
        plt.title("MRI, Slice #{}".format(slice_num));

        plt.subplot(2,2,2)
        plt.imshow(label[slice_num,:,:,0]);
        plt.title("Tumor, Slice #{}".format(slice_num));

        slice_num = self.batch_size - 1
        plt.subplot(2,2,3)
        plt.imshow(img[slice_num,:,:,0]);
        plt.title("MRI, Slice #{}".format(slice_num));

        plt.subplot(2,2,4)
        plt.imshow(label[slice_num,:,:,0]);
        plt.title("Tumor, Slice #{}".format(slice_num));
