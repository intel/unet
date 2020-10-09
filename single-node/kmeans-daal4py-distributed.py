import numpy as np
from time import time
import daal4py as d4p
from PIL import Image

d4p.daalinit()
n_colors = 8

init_algo = d4p.kmeans_init(n_colors, method="plusPlusDense", distributed=True)

max_iter = 300
acc_tres = 1e-4

img = Image.open('./Yushan.jpg') #https://commons.wikimedia.org/wiki/File:%E7%8E%89%E5%B1%B1%E4%B8%BB%E5%B3%B0_02.jpg
img.load()

china = np.array(img, dtype=np.float64) / 255

# Load Image and transform to a 2D numpy array.
w, h, d = original_shape = tuple(china.shape)
assert d == 3
image_array = np.reshape(china, (w * h, d))
o_colors = 344038 #Yushan
n_slices = int(image_array.shape[0]/d4p.num_procs())

print("Number of MPI tasks: ", d4p.num_procs())

image_array = image_array[n_slices*d4p.my_procid():n_slices*d4p.my_procid()+n_slices,:]

print("Fitting model on the data")
t0 = time()

# compute initial centroids
init_result = init_algo.compute(image_array)
assert init_result.centroids.shape[0] == n_colors
# configure kmeans main object
algo = d4p.kmeans(n_colors, max_iter, distributed=True )
# compute the clusters/centroids
result = algo.compute(image_array, init_result.centroids)
# Kmeans result objects provide centroids, goalFunction, nIterations and objectiveFunction
assert result.centroids.shape[0] == n_colors
assert result.nIterations <= max_iter

print("Computation finished in in %0.3fs." % (time() - t0))
# Get labels for all points
print("Predicting color indices on the full image (k-means)")

t0 = time()
algo = d4p.kmeans(n_colors, 0, assignFlag=True)
prediction = algo.compute(image_array, result.centroids)
labels = prediction.assignments

print("  Completed in %0.3fs." % (time() - t0))
d4p.daalfini()
