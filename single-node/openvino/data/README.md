To create the sample dataset for testing, you'll need to download the original [Medical Decathlon](http://medicaldecathlon.com/) dataset and then run the [steps](https://github.com/IntelAI/unet/blob/master/single-node/README.md) to train the model. Once you have an HDF5 version of the dataset, you can run:

```python create_validation_sample.py --hdf5_file $DECATHLON_DIRECTORY_PATH_FOR_HDF5```

which will extract a few sample images from the validation set.

 
