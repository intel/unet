
# coding: utf-8

# # Intel-Optimized Histology Demo
# 
# This demo uses the [colorectal histology images dataset](https://www.tensorflow.org/datasets/catalog/colorectal_histology) to train a simple convolutional neural network in TensorFlow. 
# 
# All images are RGB, 0.495 µm per pixel, digitized with an Aperio ScanScope (Aperio/Leica biosystems), magnification 20x. Histological samples are fully anonymized images of formalin-fixed paraffin-embedded human colorectal adenocarcinomas (primary tumors) from our pathology archive (Institute of Pathology, University Medical Center Mannheim, Heidelberg University, Mannheim, Germany).

# https://zenodo.org/record/53169#.X1bMe3lKguX
# Kather, J. N., Zöllner, F. G., Bianconi, F., Melchers, S. M., Schad, L. R., Gaiser, T., … Weis, C.-A. (2016). Collection of textures in colorectal cancer histology [Data set]. Zenodo. http://doi.org/10.5281/zenodo.53169
# 
# Kather JN, Weis CA, Bianconi F, Melchers SM, Schad LR, Gaiser T, Marx A, Zollner F: Multi-class texture analysis in colorectal cancer histology (2016), Scientific Reports (in press)
# 
# @article{kather2016multi,
#   title={Multi-class texture analysis in colorectal cancer histology},
#   author={Kather, Jakob Nikolas and Weis, Cleo-Aron and Bianconi, Francesco and Melchers, Susanne M and Schad, Lothar R and Gaiser, Timo and Marx, Alexander and Z{"o}llner, Frank Gerrit},
#   journal={Scientific reports},
#   volume={6},
#   pages={27988},
#   year={2016},
#   publisher={Nature Publishing Group}
# }

import os

# export TF_DISABLE_MKL=1
os.environ["TF_DISABLE_MKL"]  = "0"  # Disable Intel optimizations

# export MKLDNN_VERBOSE=1
#os.environ["MKLDNN_VERBOSE"]  = "1"     # 1 = Print log statements; 0 = silent

os.environ["OMP_NUM_THREADS"] = "12"   # Number of physical cores
os.environ["KMP_BLOCKTIME"]   = "1"    

# If hyperthreading is enabled, then use
os.environ["KMP_AFFINITY"]    = "granularity=thread,compact,1,0"

# If hyperthreading is NOT enabled, then use
#os.environ["KMP_AFFINITY"]   = "granularity=thread,compact"

import tensorflow as tf

print("TensorFlow version = {}".format(tf.__version__))

print("Does TensorFlow have the Intel optimizations: {}".format(tf.python._pywrap_util_port.IsMklEnabled()))

import numpy as np
import matplotlib.pyplot as plt

import tensorflow_datasets as tfds


(ds), ds_info =  tfds.load('colorectal_histology', data_dir=".", 
                                          shuffle_files=True, split='train', 
                                          with_info=True, as_supervised=True)

assert isinstance(ds, tf.data.Dataset)
print(ds_info)


# ## Display a few examples from the dataset


x_key, y_key = ds_info.supervised_keys
ds_temp = ds.map(lambda x, y: {x_key: x, y_key: y})
tfds.show_examples(ds_info, ds_temp);

ds_info.features['label'].names


# ## Define the data loaders
# 

n = ds_info.splits['train'].num_examples
train_split_percentage = 0.80
train_batch_size = 128
test_batch_size = 16

def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label

def augment_img(image, label):
    """Augment images: `uint8` -> `float32`."""
    
    image = tf.image.random_flip_left_right(image) # Random flip Left/Right
    image = tf.image.random_flip_up_down(image)    # Random flip Up/Down
    
    return tf.cast(image, tf.float32) / 255., label # Normalize 0 to 1 for pixel values

# Get train dataset
ds_train = ds.take(int(n * train_split_percentage))
ds_train = ds_train.map(
    augment_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(int(n * train_split_percentage))
ds_train = ds_train.batch(train_batch_size)
ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

# Get test dataset
ds_test = ds.skip(int(n * train_split_percentage)).map(
    normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_test = ds_test.cache()
ds_test = ds_test.batch(test_batch_size)
ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)


# ## Define Model
# 
# Here's a Convolutional neural network model. 


inputs = tf.keras.layers.Input(shape=ds_info.features['image'].shape)
conv = tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), padding="same", activation="relu")(inputs)
conv = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), padding="same", activation="relu")(conv)
maxpool = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(conv)

conv = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu")(maxpool)
conv = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu")(conv)
concat = tf.keras.layers.concatenate([maxpool, conv])
maxpool = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(concat)

conv = tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")(maxpool)
conv = tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(conv)
concat = tf.keras.layers.concatenate([maxpool, conv])
maxpool = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(concat)

conv = tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")(maxpool)
conv = tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(conv)
concat = tf.keras.layers.concatenate([maxpool, conv])
maxpool = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(concat)

flat = tf.keras.layers.Flatten()(maxpool)
dense = tf.keras.layers.Dense(128)(flat)
drop = tf.keras.layers.Dropout(0.5)(dense)

predict = tf.keras.layers.Dense(ds_info.features['label'].num_classes)(drop)

model = tf.keras.models.Model(inputs=[inputs], outputs=[predict])

model.compile(
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer='adam',
    metrics=[tf.metrics.SparseCategoricalAccuracy()],
)

model.summary()


# ## Train the model on the dataset


# Create a callback that saves the model
model_dir = "checkpoints"
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=model_dir, 
                                                         save_best_only=True,
                                                         verbose=1)




# Create callback for Early Stopping of training
early_stopping_callback = tf.keras.callbacks.EarlyStopping(patience=8) # Stop once validation loss plateaus for patience epochs


# In[12]:


# TensorBoard logs
tb_logs_dir = "logs"
os.makedirs(tb_logs_dir, exist_ok=True)
tb_callback = tf.keras.callbacks.TensorBoard(log_dir=tb_logs_dir)


# ## Call *fit* to train the model

# In[ ]:


def train_model(epochs):
    history = model.fit(
        ds_train,
        epochs=epochs,     
        validation_data=ds_test,
        callbacks=[checkpoint_callback, early_stopping_callback, tb_callback]
    )
    return history

epochs=5   # Run for this many epochs
history = train_model(epochs)


# ## Load the best model

# In[ ]:


print("Loading the best model")
model = tf.keras.models.load_model(model_dir)


# ## Evaluate the best model on the test dataset

# In[ ]:


print("Evaluating the best model on the test dataset")
_, accuracy = model.evaluate(ds_test)
print("\nModel accuracy on test dataset = {:.1f}%".format(100.0*accuracy))


# ## Display some predictions on the test data
# 
# We grab a random subset of the test dataset and plot the image along with the ground truth label, the TensorFlow model prediction, and the OpenVINO model prediction.

# In[ ]:


test_data = tfds.as_numpy(ds_test.shuffle(100).take(1)) # Take 1 random batch

for image, label in test_data:
    num = 8 # len(label)
    cols = 2
    plt.figure(figsize=(25,25))
    
    for idx in range(num):
        
        plt.subplot(int(np.ceil(num/cols)), cols, idx+1)
        plt.imshow(image[idx])
        plt.axis("off")
        
        # TensorFlow model prediction
        tf_predict = ds_info.features['label'].names[model.predict(image[[idx]]).argmax()]
        
        plt.title("Truth = {}\nTensorFlow Predict = {}".format(ds_info.features['label'].names[label[idx]], tf_predict))
        

