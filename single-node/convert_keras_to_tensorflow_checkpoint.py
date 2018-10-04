import keras
import tensorflow as tf

import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_filename",
                    help="Name of saved Keras model (e.g. model.h5)",
                    default=os.path.join(".", "unet2d_keras_model_upsampling_for_training.hdf5"))
parser.add_argument("--output_directory",
                    help="Directory where to save the TensorFlow Checkpoint",
                    default="saved_2dunet_model_checkpoint")

args = parser.parse_args()


def dice_coef(y_true, y_pred, smooth=1.0):
    intersection = tf.reduce_sum(y_true * y_pred, axis=(1, 2, 3))
    union = tf.reduce_sum(y_true + y_pred, axis=(1, 2, 3))
    numerator = tf.constant(2.) * intersection + smooth
    denominator = union + smooth
    coef = numerator / denominator
    return tf.reduce_mean(coef)


def dice_coef_loss(y_true, y_pred, smooth=1.0):

    y_true_f = keras.backend.flatten(y_true)
    y_pred_f = keras.backend.flatten(y_pred)
    intersection = keras.backend.sum(y_true_f * y_pred_f)
    loss = -keras.backend.log(2.0 * intersection + smooth) + \
        keras.backend.log((keras.backend.sum(y_true_f) +
                           keras.backend.sum(y_pred_f) + smooth))

    return loss


sess = keras.backend.get_session()

print("Loading saved Keras model.")

"""
If there are other custom loss and metric functions you'll need to specify them
and add them to the dictionary below.
"""
model = keras.models.load_model(args.input_filename, custom_objects={
                                "dice_coef": dice_coef, "dice_coef_loss": dice_coef_loss})



print("Saving the model to directory {}".format(args.output_directory))

saver = tf.train.Saver()

# Create directory if it doesn't exist
try:
    os.stat(args.output_directory)
except:
    os.mkdir(args.output_directory)

save_path = saver.save(sess, os.path.join(args.output_directory, "unet_model.ckpt"))
print("Checkpoint saved in path: {}".format(save_path))

