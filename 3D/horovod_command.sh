horovodrun -np 4 -H localhost:4 --binding-args="--map-by ppr:2:socket:pe=10" --mpi-args="--report-bindings" python train_horovod.py

