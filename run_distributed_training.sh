#!/bin/sh

# Start the parameter server
numactl -p 1 python train_dist.py --job_name="ps" --task_index=0 > training.log&

# Run the Ansible playbook to start the distribute tensorflow workers
# The playbook will sync the files from the worker directories to this directory
# It will then start the train_dist.py
# You can keep track of the training progress for each worker by
# logging into that server and looking at training.log file.
ansible-playbook -i inv.yml distributed_train.yml

