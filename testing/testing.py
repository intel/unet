#!/usr/bin/python
# ----------------------------------------------------------------------------
# Copyright 2018 Intel
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

import numpy as np
import os
import argparse
import psutil
import time
import datetime
import tensorflow as tf

from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model.signature_def_utils import predict_signature_def
from tensorflow.python.saved_model import tag_constants

parser = argparse.ArgumentParser(
    description="Sanity testing for 3D and 2D Convolution Models",add_help=True)
parser.add_argument("--dim_length",
                    type = int,
                    default=16,
                    help="Tensor cube length of side")
parser.add_argument("--num_channels",
                    type = int,
                    default=4,
                    help="Number of channels")
parser.add_argument("--num_outputs",
                    type = int,
                    default=1,
                    help="Number of outputs")

parser.add_argument("--bz",
                    type = int,
                    default=1,
                    help="Batch size")

parser.add_argument("--lr",
                    type = float,
                    default=0.001,
                    help="Learning rate")

parser.add_argument("--fms",
                    type = int,
                    default=16,
                    help="Feature maps at first level")

parser.add_argument("--num_datapoints",
                    type = int,
                    default=1024,
                    help="Number of datapoints")
parser.add_argument("--epochs",
                    type = int,
                    default=3,
                    help="Number of epochs")
parser.add_argument("--intraop_threads",
                    type = int,
                    default=psutil.cpu_count(logical=False),
                    help="Number of intraop threads")
parser.add_argument("--interop_threads",
                    type = int,
                    default=2,
                    help="Number of interop threads")
parser.add_argument("--blocktime",
                    type = int,
                    default=0,
                    help="Block time for CPU threads")
parser.add_argument("--print_model",
                    action="store_true",
                    default=False,
                    help="Print the summary of the model layers")
parser.add_argument("--use_upsampling",
                    action="store_true",
                    default=False,
                    help="Use upsampling instead of transposed convolution")
parser.add_argument("--D2",
                    action="store_true",
                    default=False,
                    help="Use 2D model and images instead of 3D.")
parser.add_argument("--single_class_output",
                    action="store_true",
                    default=False,
                    help="Use binary classifier instead of U-Net")
parser.add_argument("--mkl_verbose",
                    action="store_true",
                    default=False,
                    help="Print MKL debug statements.")
parser.add_argument("--inference",
                    action="store_true",
                    default=False,
                    help="Test inference speed. Default=Test training speed")
parser.add_argument("--ngraph",
                    action="store_true",
                    default=False,
                    help="Use ngraph")
parser.add_argument("--keras_api",
                    action="store_true",
                    default=True,
                    help="Use Keras API. False=Use tf.keras")
parser.add_argument("--channels_first",
                    action="store_true",
                    default=False,
                    help="Channels first. NCHW")

args = parser.parse_args()

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Get rid of the AVX, SSE warnings
if args.mkl_verbose:
    os.environ["MKL_VERBOSE"] = "1"  # Print out messages from MKL operations
    os.environ["MKLDNN_VERBOSE"] = "1"  # Print out messages from MKL-DNN operations
os.environ["OMP_NUM_THREADS"] = str(args.intraop_threads)
os.environ["KMP_BLOCKTIME"] = str(args.blocktime)
os.environ["KMP_AFFINITY"] = "granularity=thread,compact,1,0"


print("Started script on {}".format(datetime.datetime.now()))

print("args = {}".format(args))
os.system("uname -a")
print("TensorFlow version: {}".format(tf.__version__))

if args.keras_api:
    import keras as K
    print("Using Keras API")
else:
    from tensorflow import keras as K
    print("Using tf.keras")

if args.ngraph:
    print("Using nGraph")
    import ngraph_bridge
    if args.channels_first:
        os.environ["NGRAPH_PASS_ENABLES"]="CPUReshapeSinking:1;ReshapeElimination:1"

print("Keras API version: {}".format(K.__version__))

if args.D2:  # Define shape of the tensors (2D)
    dims = (1,2)
    if args.channels_first:
        tensor_shape = (args.num_channels,
                        args.dim_length,
                        args.dim_length)
        out_shape = (args.num_outputs,
                        args.dim_length,
                        args.dim_length)
    else:
        tensor_shape = (args.dim_length,
                        args.dim_length,
                        args.num_channels)
        out_shape = (args.dim_length,
                        args.dim_length,
                        args.num_outputs)
else:        # Define shape of the tensors (3D)
    dims=(1,2,3)
    if args.channels_first:
        tensor_shape = (args.num_channels,
                        args.dim_length,
                        args.dim_length,
                        args.dim_length)
        tensor_shape = (args.num_outputs,
                        args.dim_length,
                        args.dim_length,
                        args.dim_length)
    else:
        tensor_shape = (args.dim_length,
                        args.dim_length,
                        args.dim_length,
                        args.num_channels)
        tensor_shape = (args.dim_length,
                        args.dim_length,
                        args.dim_length,
                        args.num_outputs)


def dice_coef(y_true, y_pred, axis=(1,2,3), smooth=1.0):
   intersection = tf.reduce_sum(y_true * K.backend.round(y_pred), axis=axis)
   union = tf.reduce_sum(y_true + K.backend.round(y_pred), axis=axis)
   numerator = tf.constant(2.) * intersection + smooth
   denominator = union + smooth
   coef = numerator / denominator
   return tf.reduce_mean(coef)

def dice_coef_loss(target, prediction, axis=(1,2,3), smooth=1.0):
    """
    Sorenson Dice loss
    Using -log(Dice) as the loss since it is better behaved.
    Also, the log allows avoidance of the division which
    can help prevent underflow when the numbers are very small.
    """
    intersection = tf.reduce_sum(prediction * target, axis=axis)
    p = tf.reduce_sum(prediction, axis=axis)
    t = tf.reduce_sum(target, axis=axis)
    numerator = tf.reduce_mean(2. * intersection + smooth)
    denominator = tf.reduce_mean(t + p + smooth)
    dice_loss = -tf.math.log(numerator) + tf.math.log(denominator)

    return dice_loss

if args.channels_first:
    concat_axis = 1
    data_format = "channels_first"
else:
    concat_axis = -1
    data_format = "channels_last"

def unet3D(input_img, use_upsampling=False, n_out=1, dropout=0.2,
            print_summary = False, return_model=False):
    """
    3D U-Net
    """
    def ConvolutionBlock(x, name, fms, params):
        """
        Convolutional block of layers
        Per the original paper this is back to back 3D convs
        with batch norm and then ReLU.
        """

        x = K.layers.Conv3D(filters=fms, **params, name=name+"_conv0")(x)
        x = K.layers.BatchNormalization(name=name+"_bn0")(x)
        x = K.layers.Activation("relu", name=name+"_relu0")(x)

        x = K.layers.Conv3D(filters=fms, **params, name=name+"_conv1")(x)
        x = K.layers.BatchNormalization(name=name+"_bn1")(x)
        x = K.layers.Activation("relu", name=name)(x)

        return x

    inputs = K.layers.Input(shape=input_img, name="MRImages")

    params = dict(kernel_size=(3, 3, 3), activation=None,
                  padding="same", data_format=data_format,
                  kernel_initializer="he_uniform")

    # Transposed convolution parameters
    params_trans = dict(data_format=data_format,
                        kernel_size=(2, 2, 2), strides=(2, 2, 2),
                        padding="same")


    # BEGIN - Encoding path
    encodeA = ConvolutionBlock(inputs, "encodeA", args.fms, params)
    poolA = K.layers.MaxPooling3D(name="poolA", pool_size=(2, 2, 2))(encodeA)

    encodeB = ConvolutionBlock(poolA, "encodeB", args.fms*2, params)
    poolB = K.layers.MaxPooling3D(name="poolB", pool_size=(2, 2, 2))(encodeB)

    encodeC = ConvolutionBlock(poolB, "encodeC", args.fms*4, params)
    poolC = K.layers.MaxPooling3D(name="poolC", pool_size=(2, 2, 2))(encodeC)

    encodeD = ConvolutionBlock(poolC, "encodeD", args.fms*8, params)
    poolD = K.layers.MaxPooling3D(name="poolD", pool_size=(2, 2, 2))(encodeD)

    encodeE = ConvolutionBlock(poolD, "encodeE", args.fms*16, params)
    # END - Encoding path

    # BEGIN - Decoding path
    if use_upsampling:
        up = K.layers.UpSampling3D(name="upE", size=(2, 2, 2))(encodeE)
    else:
        up = K.layers.Conv3DTranspose(name="transconvE", filters=args.fms*8,
                                      **params_trans)(encodeE)
    concatD = K.layers.concatenate(
        [up, encodeD], axis=concat_axis, name="concatD")

    decodeC = ConvolutionBlock(concatD, "decodeC", args.fms*8, params)

    if use_upsampling:
        up = K.layers.UpSampling3D(name="upC", size=(2, 2, 2))(decodeC)
    else:
        up = K.layers.Conv3DTranspose(name="transconvC", filters=args.fms*4,
                                      **params_trans)(decodeC)
    concatC = K.layers.concatenate(
        [up, encodeC], axis=concat_axis, name="concatC")

    decodeB = ConvolutionBlock(concatC, "decodeB", args.fms*4, params)

    if use_upsampling:
        up = K.layers.UpSampling3D(name="upB", size=(2, 2, 2))(decodeB)
    else:
        up = K.layers.Conv3DTranspose(name="transconvB", filters=args.fms*2,
                                      **params_trans)(decodeB)
    concatB = K.layers.concatenate(
        [up, encodeB], axis=concat_axis, name="concatB")

    decodeA = ConvolutionBlock(concatB, "decodeA", args.fms*2, params)

    if use_upsampling:
        up = K.layers.UpSampling3D(name="upA", size=(2, 2, 2))(decodeA)
    else:
        up = K.layers.Conv3DTranspose(name="transconvA", filters=args.fms,
                                      **params_trans)(decodeA)
    concatA = K.layers.concatenate(
        [up, encodeA], axis=concat_axis, name="concatA")

    # END - Decoding path

    convOut = ConvolutionBlock(concatA, "convOut", args.fms, params)

    prediction = K.layers.Conv3D(name="PredictionMask",
                                 filters=n_out, kernel_size=(1, 1, 1),
                                 data_format=data_format,
                                 activation="sigmoid")(convOut)

    model = K.models.Model(inputs=[inputs], outputs=[prediction])

    if return_model:
        model = K.models.Model(inputs=[inputs], outputs=[prediction])

        if print_summary:
            print(model.summary())

        return prediction, model
    else:
        return prediction

def unet2D(input_tensor, use_upsampling=False,
            n_out=1, dropout=0.2, print_summary = False, return_model=False):
    """
    2D U-Net
    """
    print("2D U-Net Segmentation")

    inputs = K.layers.Input(shape=input_tensor, name="Images")

    # Convolution parameters
    params = dict(kernel_size=(3, 3), activation="relu",
                  padding="same", data_format=data_format,
                  kernel_initializer="he_uniform")

    # Transposed convolution parameters
    params_trans = dict(data_format=data_format,
                        kernel_size=(2, 2), strides=(2, 2),
                        padding="same")

    fms = args.fms

    conv1 = K.layers.Conv2D(name="conv1a", filters=fms, **params)(inputs)
    conv1 = K.layers.Conv2D(name="conv1b", filters=fms, **params)(conv1)
    pool1 = K.layers.MaxPooling2D(name="pool1", pool_size=(2, 2))(conv1)

    conv2 = K.layers.Conv2D(name="conv2a", filters=fms*2, **params)(pool1)
    conv2 = K.layers.Conv2D(name="conv2b", filters=fms*2, **params)(conv2)
    pool2 = K.layers.MaxPooling2D(name="pool2", pool_size=(2, 2))(conv2)

    conv3 = K.layers.Conv2D(name="conv3a", filters=fms*4, **params)(pool2)
    #conv3 = K.layers.Dropout(dropout)(conv3)
    conv3 = K.layers.Conv2D(name="conv3b", filters=fms*4, **params)(conv3)

    pool3 = K.layers.MaxPooling2D(name="pool3", pool_size=(2, 2))(conv3)

    conv4 = K.layers.Conv2D(name="conv4a", filters=fms*8, **params)(pool3)
    #conv4 = K.layers.Dropout(dropout)(conv4)
    conv4 = K.layers.Conv2D(name="conv4b", filters=fms*8, **params)(conv4)

    pool4 = K.layers.MaxPooling2D(name="pool4", pool_size=(2, 2))(conv4)

    conv5 = K.layers.Conv2D(name="conv5a", filters=fms*16, **params)(pool4)
    conv5 = K.layers.Conv2D(name="conv5b", filters=fms*16, **params)(conv5)

    if use_upsampling:
        up = K.layers.UpSampling2D(name="up6", size=(2, 2))(conv5)
    else:
        up = K.layers.Conv2DTranspose(name="transConv6", filters=fms*8,
                                      **params_trans)(conv5)
    up6 = K.layers.concatenate([up, conv4], axis=concat_axis)

    conv6 = K.layers.Conv2D(name="conv6a", filters=fms*8, **params)(up6)
    conv6 = K.layers.Conv2D(name="conv6b", filters=fms*8, **params)(conv6)

    if use_upsampling:
        up = K.layers.UpSampling2D(name="up7", size=(2, 2))(conv6)
    else:
        up = K.layers.Conv2DTranspose(name="transConv7", filters=fms*4,
                                      **params_trans)(conv6)
    up7 = K.layers.concatenate([up, conv3], axis=concat_axis)

    conv7 = K.layers.Conv2D(name="conv7a", filters=fms*4, **params)(up7)
    conv7 = K.layers.Conv2D(name="conv7b", filters=fms*4, **params)(conv7)

    if use_upsampling:
        up = K.layers.UpSampling2D(name="up8", size=(2, 2))(conv7)
    else:
        up = K.layers.Conv2DTranspose(name="transConv8", filters=fms*2,
                                      **params_trans)(conv7)
    up8 = K.layers.concatenate([up, conv2], axis=concat_axis)

    conv8 = K.layers.Conv2D(name="conv8a", filters=fms*2, **params)(up8)
    conv8 = K.layers.Conv2D(name="conv8b", filters=fms*2, **params)(conv8)

    if use_upsampling:
        up = K.layers.UpSampling2D(name="up9", size=(2, 2))(conv8)
    else:
        up = K.layers.Conv2DTranspose(name="transConv9", filters=fms,
                                      **params_trans)(conv8)
    up9 = K.layers.concatenate([up, conv1], axis=concat_axis)

    conv9 = K.layers.Conv2D(name="conv9a", filters=fms, **params)(up9)
    conv9 = K.layers.Conv2D(name="conv9b", filters=fms, **params)(conv9)

    pred = K.layers.Conv2D(name="PredictionMask",
                                 filters=n_out, kernel_size=(1, 1),
                                 data_format=data_format,
                                 activation="sigmoid")(conv9)

    if return_model:
        model = K.models.Model(inputs=[inputs], outputs=[pred])

        if print_summary:
            print(model.summary())

        return pred, model
    else:
        return pred

def conv3D(input_img, print_summary = False, dropout=0.2, n_out=1,
            return_model=False):
    """
    Simple 3D convolution model based on VGG-16
    """
    print("3D Convolutional Binary Classifier based on VGG-16")

    inputs = K.layers.Input(shape=input_img, name="Images")

    params = dict(kernel_size=(3, 3, 3), activation="relu",
                  padding="same", data_format=data_format,
                  kernel_initializer="he_uniform") #RandomUniform(minval=-0.01, maxval=0.01, seed=816))

    conv1 = K.layers.Conv3D(name="conv1", filters=64, **params)(inputs)
    conv2 = K.layers.Conv3D(name="conv2", filters=64, **params)(conv1)
    pool1 = K.layers.MaxPooling3D(name="pool1", pool_size=(2, 2, 2))(conv2)

    conv3 = K.layers.Conv3D(name="conv3", filters=128, **params)(pool1)
    conv4 = K.layers.Conv3D(name="conv4", filters=128, **params)(conv3)
    pool2 = K.layers.MaxPooling3D(name="pool2", pool_size=(2, 2, 2))(conv4)

    conv5 = K.layers.Conv3D(name="conv5", filters=256, **params)(pool2)
    conv6 = K.layers.Conv3D(name="conv6", filters=256, **params)(conv5)
    conv7 = K.layers.Conv3D(name="conv7", filters=256, **params)(conv6)
    pool3 = K.layers.MaxPooling3D(name="pool3", pool_size=(2, 2, 2))(conv7)

    conv8 = K.layers.Conv3D(name="conv8", filters=512, **params)(pool3)
    conv9 = K.layers.Conv3D(name="conv9", filters=512, **params)(conv8)
    conv10 = K.layers.Conv3D(name="conv10", filters=512, **params)(conv9)
    pool4 = K.layers.MaxPooling3D(name="pool4", pool_size=(2, 2, 2))(conv10)

    conv11 = K.layers.Conv3D(name="conv11", filters=512, **params)(pool4)
    conv12 = K.layers.Conv3D(name="conv12", filters=512, **params)(conv11)
    conv13 = K.layers.Conv3D(name="conv13", filters=512, **params)(conv12)
    pool5 = K.layers.MaxPooling3D(name="pool5", pool_size=(2, 2, 2))(conv13)

    flat = K.layers.Flatten()(pool5)
    dense1 = K.layers.Dense(4096, activation="relu")(flat)
    drop1 = K.layers.Dropout(dropout)(dense1)
    dense2 = K.layers.Dense(4096, activation="relu")(drop1)
    pred = K.layers.Dense(n_out, name="Prediction", activation="sigmoid")(dense2)

    if return_model:
        model = K.models.Model(inputs=[inputs], outputs=[pred])

        if print_summary:
            print(model.summary())

        return pred, model
    else:
        return pred


def conv2D(input_tensor, print_summary = False, dropout=0.2, n_out=1, return_model=False):

    """
    Simple 2D convolution model based on VGG-16
    """
    print("2D Convolutional Binary Classifier based on VGG-16")

    inputs = K.layers.Input(shape=input_tensor, name="Images")

    params = dict(kernel_size=(3, 3), activation="relu",
                  padding="same", data_format=data_format,
                  kernel_initializer="he_uniform") #RandomUniform(minval=-0.01, maxval=0.01, seed=816))

    conv1 = K.layers.Conv2D(name="conv1", filters=64, **params)(inputs)
    conv2 = K.layers.Conv2D(name="conv2", filters=64, **params)(conv1)
    pool1 = K.layers.MaxPooling2D(name="pool1", pool_size=(2, 2))(conv2)

    conv3 = K.layers.Conv2D(name="conv3", filters=128, **params)(pool1)
    conv4 = K.layers.Conv2D(name="conv4", filters=128, **params)(conv3)
    pool2 = K.layers.MaxPooling2D(name="pool2", pool_size=(2, 2))(conv4)

    conv5 = K.layers.Conv2D(name="conv5", filters=256, **params)(pool2)
    conv6 = K.layers.Conv2D(name="conv6", filters=256, **params)(conv5)
    conv7 = K.layers.Conv2D(name="conv7", filters=256, **params)(conv6)
    pool3 = K.layers.MaxPooling2D(name="pool3", pool_size=(2, 2))(conv7)

    conv8 = K.layers.Conv2D(name="conv8", filters=512, **params)(pool3)
    conv9 = K.layers.Conv2D(name="conv9", filters=512, **params)(conv8)
    conv10 = K.layers.Conv2D(name="conv10", filters=512, **params)(conv9)
    pool4 = K.layers.MaxPooling2D(name="pool4", pool_size=(2, 2))(conv10)

    conv11 = K.layers.Conv2D(name="conv11", filters=512, **params)(pool4)
    conv12 = K.layers.Conv2D(name="conv12", filters=512, **params)(conv11)
    conv13 = K.layers.Conv2D(name="conv13", filters=512, **params)(conv12)
    pool5 = K.layers.MaxPooling2D(name="pool5", pool_size=(2, 2))(conv13)

    flat = K.layers.Flatten()(pool5)
    dense1 = K.layers.Dense(4096, activation="relu")(flat)
    drop1 = K.layers.Dropout(dropout)(dense1)
    dense2 = K.layers.Dense(4096, activation="relu")(drop1)
    pred = K.layers.Dense(n_out, name="Prediction", activation="sigmoid")(dense2)

    if return_model:
        model = K.models.Model(inputs=[inputs], outputs=[pred])

        if print_summary:
            print(model.summary())

        return pred, model
    else:
        return pred


if args.single_class_output:
    if args.D2:    # 2D convnet model
        pred, model = conv2D(tensor_shape,
                       print_summary=args.print_model, n_out=args.num_outputs,
                       return_model=True)
    else:            # 3D convet model
        pred, model = conv3D(tensor_shape,
                       print_summary=args.print_model, n_out=args.num_outputs,
                       return_model=True)
else:

    if args.D2:    # 2D U-Net model
        pred, model = unet2D(tensor_shape,
                       use_upsampling=args.use_upsampling,
                       print_summary=args.print_model, n_out=args.num_outputs,
                       return_model=True)
    else:            # 3D U-Net model
        pred, model = unet3D(tensor_shape,
                       use_upsampling=args.use_upsampling,
                       print_summary=args.print_model, n_out=args.num_outputs,
                       return_model=True)

# Freeze layers
if args.inference:
   for layer in model.layers:
       layer.trainable = False

#  Performance metrics for model
if args.single_class_output:
    model.compile(loss="binary_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

else:
    model.compile(loss=dice_coef_loss,
              optimizer="adam",
              metrics=[dice_coef, "accuracy"])


def get_imgs():

    # Just feed completely random data in for the benchmark testing
    sh = [args.bz] + list(tensor_shape)
    imgs = np.random.rand(*sh)

    while True:
        yield imgs

def get_batch():

    # Just feed completely random data in for the benchmark testing
    sh = [args.bz] + list(tensor_shape)

    imgs = np.random.rand(*sh)

    if args.single_class_output:
        truths = np.random.rand(args.bz, args.num_outputs)
    else:
        truths = np.random.rand(*sh)


    while True:
        yield imgs, truths

# Same number of sample to process regardless of batch size
# So if we have a larger batch size we can take fewer steps.
total_steps = args.num_datapoints//args.bz

print("Using random data.")
if args.inference:
    print("Testing inference speed.")
else:
    print("Testing training speed.")

start_time = time.time()
if args.inference:
   for _ in range(args.epochs):
       model.predict_generator(get_imgs(), steps=total_steps, verbose=1)
else:
    model.fit_generator(get_batch(), steps_per_epoch=total_steps,
                        epochs=args.epochs, verbose=1)

if args.inference:
   import shutil
   dirName = "./tensorflow_serving_model"
   if args.single_class_output:
      dirName += "_VGG16"
   else:
      dirName += "_UNET"
   if args.D2:
      dirName += "_2D"
   else:
      dirName += "_3D"

   shutil.rmtree(dirName, ignore_errors=True)
   # Save TensorFlow serving model
   builder = saved_model_builder.SavedModelBuilder(dirName)
   # Create prediction signature to be used by TensorFlow Serving Predict API
   signature = predict_signature_def(inputs={"images": model.input},
                                      outputs={"scores": model.output})
   # Save the meta graph and the variables
   builder.add_meta_graph_and_variables(sess=K.backend.get_session(), tags=[tag_constants.SERVING],
                                        signature_def_map={"predict": signature})

   builder.save()
   print("Saved TensorFlow Serving model to: {}".format(dirName))

stop_time = time.time()

print("\n\nTotal time = {:,.3f} seconds".format(stop_time - start_time))
print("Total images = {:,}".format(args.epochs*args.num_datapoints))
print("Speed = {:,.3f} images per second".format( \
            (args.epochs*args.num_datapoints)/(stop_time - start_time)))
