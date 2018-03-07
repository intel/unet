""" For helper code """

import numpy as np
import os.path

import tensorflow as tf

USE_OLD_KERAS = False
if USE_OLD_KERAS:
	import keras


def f1(y_true, y_pred):
	def recall(y_true, y_pred):
		"""Recall metric.

		Only computes a batch-wise average of recall.

		Computes the recall, a metric for multi-label classification of
		how many relevant items are selected.
		"""
		true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
		possible_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true, 0, 1)))
		recall = true_positives / (possible_positives + tf.keras.backend.epsilon())
		return recall

	def precision(y_true, y_pred):
		"""Precision metric.

		Only computes a batch-wise average of precision.

		Computes the precision, a metric for multi-label classification of
		how many selected items are relevant.
		"""
		true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
		predicted_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_pred, 0, 1)))
		precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
		return precision
	precision = precision(y_true, y_pred)
	recall = recall(y_true, y_pred)
	return ((precision*recall)/(precision+recall))


def dice_coef(y_true, y_pred, smooth = 1. ):

	if USE_OLD_KERAS:

		y_true_f = keras.backend.flatten(y_true)
		y_pred_f = keras.backend.flatten(y_pred)
		intersection = keras.backend.sum(y_true_f * y_pred_f)
		coef = (2. * intersection + smooth) / (keras.backend.sum(y_true_f) + keras.backend.sum(y_pred_f) + smooth)
	
	else:
		y_true_f = tf.keras.backend.flatten(y_true)
		y_pred_f = tf.keras.backend.flatten(y_pred)
		intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
		coef = (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)
	return coef

def dice_coef_loss(y_true, y_pred):

	if USE_OLD_KERAS:

		smooth = 1.
		y_true_f = keras.backend.flatten(y_true)
		y_pred_f = keras.backend.flatten(y_pred)
		intersection = keras.backend.sum(y_true_f * y_pred_f)
		loss = -keras.backend.log(2.0*intersection + smooth) + \
			keras.backend.log((keras.backend.sum(y_true_f) + keras.backend.sum(y_pred_f) + smooth))

	else:

		smooth = 1.
		y_true_f = tf.keras.backend.flatten(y_true)
		y_pred_f = tf.keras.backend.flatten(y_pred)
		intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
		loss = -tf.keras.backend.log(2.0*intersection + smooth) + \
			tf.keras.backend.log((tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth))


	return loss


