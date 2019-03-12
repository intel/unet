from imports import *

import datetime
from tqdm import tqdm

from argparser import args

from model import *

from dataloader import DataGenerator

import os
import nibabel as nib

print("Started script on {}".format(datetime.datetime.now()))

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Get rid of the AVX, SSE warnings
os.environ["OMP_NUM_THREADS"] = str(args.intraop_threads)
os.environ["KMP_BLOCKTIME"] = str(args.blocktime)
os.environ["KMP_AFFINITY"] = "granularity=thread,compact,1,0"

# Optimize CPU threads for TensorFlow
config = tf.ConfigProto(
    inter_op_parallelism_threads=args.interop_threads,
    intra_op_parallelism_threads=args.intraop_threads)

sess = tf.Session(config=config)

K.backend.set_session(sess)

model = K.models.load_model(args.saved_model,
                            custom_objects={"dice_coef":dice_coef,
                            "dice_coef_loss":dice_coef_loss,
                            "sensitivity":sensitivity,
                            "specificity":specificity,
                            "combined_dice_ce_loss":combined_dice_ce_loss})

print("Loading images and masks from test set")

validation_data_params = {"dim": (args.patch_height, args.patch_width, args.patch_depth),
                          "batch_size": 1,
                          "n_in_channels": args.number_input_channels,
                          "n_out_channels": 1,
                          "train_test_split": args.train_test_split,
                          "augment": False,
                          "shuffle": False,"seed": args.random_seed}
validation_generator = DataGenerator(False, args.data_path,
                                     **validation_data_params)

m = model.evaluate_generator(validation_generator, verbose=1,
                             max_queue_size=args.num_prefetched_batches,
                             workers=args.num_data_loaders,
                             use_multiprocessing=False)

print("\n\nTest metrics")
print("============")
i = 0
for name in model.metrics_names:
    print("{} = {:.4f}".format(name, m[i]))
    i += 1

save_directory = "predictions_directory"
try:
    os.stat(save_directory)
except:
    os.mkdir(save_directory)

print("Predicting masks")

for batch_idx in tqdm(range(validation_generator.num_batches),
                      desc="Predicting on batch"):

    imgs, msks = validation_generator.get_batch(batch_idx)
    fileIDs = validation_generator.get_batch_fileIDs(batch_idx)

    preds = model.predict_on_batch(imgs)

    # Save the predictions as Nifti files so that we can
    # display them on a 3D viewer.
    for idx in tqdm(range(preds.shape[0]), desc="Saving to Nifti file"):

        img = nib.Nifti1Image(imgs[idx,:,:,:,0], np.eye(4))
        img.to_filename(os.path.join(save_directory,
                        "{}_img.nii.gz".format(fileIDs[idx])))

        msk = nib.Nifti1Image(msks[idx,:,:,:,0], np.eye(4))
        msk.to_filename(os.path.join(save_directory,
                        "{}_msk.nii.gz".format(fileIDs[idx])))

        pred = nib.Nifti1Image(preds[idx,:,:,:,0], np.eye(4))
        pred.to_filename(os.path.join(save_directory,
                         "{}_pred.nii.gz".format(fileIDs[idx])))

print("\n\n\nModel predictions saved to directory: {}".format(save_directory))
print("Stopped script on {}".format(datetime.datetime.now()))
