import pandas as pd
import glob
import os
import numpy as np

"""
Training
"""
print("\n\nTraining")
out = {}
bzs = []
dims = []
for file in glob.glob("*_train_*.dat"):
    df = pd.read_csv(file, delimiter=" ")
    filenames = file.split("_")
    bz = int(os.path.splitext(filenames[3][2:])[0])
    dim = int(filenames[2][3:])

    out[dim, bz] = df.iloc[:,1].max()
    dims.append(dim)
    bzs.append(bz)

print("Input_Size\t",end="")
for bz in np.unique(bzs):
    print("{}\t".format(bz), end="")
print("")
for dim in np.unique(dims):
    print("{}x{}x{}\t".format(dim,dim,dim), end="")
    for bz in np.unique(bzs):

        print("{}\t".format(out[dim,bz]), end="")

    print("")


"""
Inference
"""
print("\n\nInference")
out = {}
bzs = []
dims = []
for file in glob.glob("*_inference_*.dat"):
    df = pd.read_csv(file, delimiter=" ")
    filenames = file.split("_")
    bz = int(os.path.splitext(filenames[3][2:])[0])
    dim = int(filenames[2][3:])

    out[dim, bz] = df.iloc[:,1].max()
    dims.append(dim)
    bzs.append(bz)

print("Input_Size\t",end="")
for bz in np.unique(bzs):
    print("{}\t".format(bz), end="")
print("")
for dim in np.unique(dims):
    print("{}x{}x{}\t".format(dim,dim,dim), end="")
    for bz in np.unique(bzs):

        print("{}\t".format(out[dim,bz]), end="")

    print("")
