import tensorflow as tf

import numpy as np

imgs_test = np.load("imgs_test_norm.npy")
msks_test = np.load("msks_test_norm.npy")

img_number = 75

def dice_score(pred, truth):

    numerator = 2*np.sum(pred*truth) + 1.0
    denominator = np.sum(pred) + np.sum(truth) + 1.0
    return numerator / denominator

with tf.Session() as sess:
  saver = tf.train.import_meta_graph("tf_checkpoint/unet_model.ckpt.meta")
  saver.restore(sess, tf.train.latest_checkpoint("tf_checkpoint"))
  graph = tf.get_default_graph()

  PredictionMask = graph.get_tensor_by_name("PredictionMask/Sigmoid:0")[0]
  prediction = np.array(sess.run([PredictionMask], feed_dict={"Images:0":imgs_test[img_number:(img_number+1),:,:,:]}))

print("Dice score = {:.4}".format(dice_score(prediction[0, :, :, 0], msks_test[img_number,:,:,0])))

#import matplotlib.pyplot as plt
# # Plot the results
# plt.subplot(1,3,1)
# plt.imshow(imgs_test[img_number,:,:,0])
# plt.title("MRI")
#
# plt.subplot(1,3,2)
# plt.imshow(msks_test[img_number,:,:,0])
# plt.title("Ground truth")
#
# plt.subplot(1,3,3)
# plt.imshow(prediction[0,:,:,0])
# plt.title("Prediction")
#
# plt.show()
