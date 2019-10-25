#!/bin/bash

#pip install memory_profiler
rm *.dat
rm *.log

CMDS="--epochs 3 --fms 16"

for dim_length in 32 64 128 144 200 240 320 # 400 480 512 600
do

  num=16

  # Training batch size 1
  echo "Training batch size 1, dim_length ${dim_length}"
  mprof run \
       --output unet3d_train_len${dim_length}_bz1.dat \
       python testing.py \
       --dim_length $dim_length \
       --num_datapoints $num --bz 1 $CMDS \
       2>&1 | tee train_unet_${dim_length}_bz1.log

  bash clear_caches.sh

  # Inference batch size 1
  echo "Inference batch size 1, dim_length ${dim_length}"
  mprof run \
       --output unet3d_inference_len${dim_length}_bz1.dat \
       python testing.py \
       --dim_length $dim_length \
       --num_datapoints $num $CMDS --bz 1 --inference \
       2>&1 | tee inference_unet_${dim_length}_bz1.log

  bash clear_caches.sh

  # Inference batch size 2
  echo "Inference batch size 2, dim_length ${dim_length}"
  mprof run \
       --output unet3d_inference_len${dim_length}_bz2.dat \
       python testing.py \
       --dim_length $dim_length \
       --num_datapoints $num $CMDS --bz 2 --inference \
       2>&1 | tee inference_unet_${dim_length}_bz2.log

  bash clear_caches.sh

  # Inference batch size 4
  echo "Inference batch size 4, dim_length ${dim_length}"
  mprof run \
       --output unet3d_inference_len${dim_length}_bz4.dat \
       python testing.py \
       --dim_length $dim_length \
       --num_datapoints $num $CMDS --bz 4 --inference \
       2>&1 | tee inference_unet_${dim_length}_bz4.log

  bash clear_caches.sh

  #Training batch size 2
  echo "Training batch size 2, dim_length ${dim_length}"
  mprof run \
       --output unet3d_train_len${dim_length}_bz2.dat \
       python testing.py \
       --dim_length $dim_length \
       --num_datapoints $num $CMDS --bz 2  \
       2>&1 | tee inference_unet_${dim_length}_bz2.log

  bash clear_caches.sh

  # Training batch size 4
  echo "Training batch size 4, dim_length ${dim_length}"
  mprof run \
       --output unet3d_train_len${dim_length}_bz4.dat \
       python testing.py \
       --dim_length $dim_length \
       --num_datapoints $num $CMDS --bz 4  \
       2>&1 | tee inference_unet_${dim_length}_bz4.log

  bash clear_caches.sh

done

echo "Done"
