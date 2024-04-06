#!/bin/bash

# Permuted MNIST
python main_mnist.py --main_optimizer 'sgd'  --cm_epochs 50 --ae_epochs 200 --lr_epochs 10 --cl_epochs 10 --main_online_lr 1e-3 --train-batch-size 128 --wd 1e-3 --num_tasks 5 --cl_dataset 'permuted_mnist' --rank 0 \
            --mlp_saved_iterations 128 --ae_offline_lr 1e-2 --ae_cl_lam 100 --ae_re_lam 1 --cm_topk 500 --ae_topk 250 --cl_method 'dco' --ae_what 'M' --push_cone_l2 0.2

# Split MNIST
python main_mnist.py --main_optimizer 'sgd' --cm_epochs 50 --ae_epochs 200 --lr_epochs 10 --cl_epochs 10 --main_online_lr 1e-3 --train-batch-size 128 --wd 1e-3 --num_tasks 5 --cl_dataset 'split_mnist' --rank 0 \
            --mlp_saved_iterations 128 --ae_offline_lr 1e-2 --ae_cl_lam 100 --ae_re_lam 1 --cm_topk 200 --ae_topk 200 --cl_method 'dco' --ae_what 'M' --push_cone_l2 0.2

# Split CIFAR-100
python main_cifar100.py --main_optimizer 'sgd' --cm_epochs 50 --ae_epochs 200 --lr_epochs 60 --cl_epochs 60 --main_online_lr 1e-3 --train-batch-size 64 --num_tasks 10 --cl_dataset 'split_cifar100' --ae_grad_norm 10 --rank 0 \
            --mlp_saved_iterations 128 --ae_offline_lr 1e-2 --ae_cl_lam 1000 --ae_re_lam 1 --cm_topk 1000 --ae_topk 250 --cl_method 'dco' --ae_what 'M' --push_cone_l2 0.2