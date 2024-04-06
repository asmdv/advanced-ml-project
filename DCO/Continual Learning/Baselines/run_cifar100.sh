#!/bin/bash

# CIFAR-100
python main_cifar100.py --main_optimizer 'sgd'  --lr_epochs 60 --cl_epochs 60 --main_online_lr 1e-2 --train-batch-size 64 --cl_method 'sgd'  --cl_dataset 'split_cifar100' --num_tasks 10 --rank 0 &
python main_cifar100.py --main_optimizer 'sgd'  --lr_epochs 60 --cl_epochs 60 --main_online_lr 1e-3 --train-batch-size 64 --cl_method 'sgd'  --cl_dataset 'split_cifar100' --num_tasks 10 --rank 1 &
python main_cifar100.py --main_optimizer 'sgd'  --lr_epochs 60 --cl_epochs 60 --main_online_lr 1e-4 --train-batch-size 64 --cl_method 'sgd'  --cl_dataset 'split_cifar100' --num_tasks 10 --rank 2 &
python main_cifar100.py --main_optimizer 'sgd'  --lr_epochs 60 --cl_epochs 60 --main_online_lr 1e-5 --train-batch-size 64 --cl_method 'sgd'  --cl_dataset 'split_cifar100' --num_tasks 10 --rank 3 &
wait

python main_cifar100.py --main_optimizer 'sgd'  --lr_epochs 60 --cl_epochs 60 --main_online_lr 1e-3 --train-batch-size 64 --cl_method 'ewc' --num_tasks 10 --ewc_lam 1e0 --cl_dataset 'split_cifar100' --rank 0 &
python main_cifar100.py --main_optimizer 'sgd'  --lr_epochs 60 --cl_epochs 60 --main_online_lr 1e-3 --train-batch-size 64 --cl_method 'ewc' --num_tasks 10 --ewc_lam 1e1 --cl_dataset 'split_cifar100' --rank 1 &
python main_cifar100.py --main_optimizer 'sgd'  --lr_epochs 60 --cl_epochs 60 --main_online_lr 1e-3 --train-batch-size 64 --cl_method 'ewc' --num_tasks 10 --ewc_lam 1e2 --cl_dataset 'split_cifar100' --rank 2 &
python main_cifar100.py --main_optimizer 'sgd'  --lr_epochs 60 --cl_epochs 60 --main_online_lr 1e-3 --train-batch-size 64 --cl_method 'ewc' --num_tasks 10 --ewc_lam 1e3 --cl_dataset 'split_cifar100' --rank 3 &
wait

python main_cifar100.py --main_optimizer 'sgd'  --lr_epochs 60 --cl_epochs 60 --main_online_lr 1e-3 --train-batch-size 64 --cl_method 'si' --num_tasks 10 --si_lam 1e-1 --cl_dataset 'split_cifar100' --rank 0 &
python main_cifar100.py --main_optimizer 'sgd'  --lr_epochs 60 --cl_epochs 60 --main_online_lr 1e-3 --train-batch-size 64 --cl_method 'si' --num_tasks 10 --si_lam  1e0 --cl_dataset 'split_cifar100' --rank 1 &
python main_cifar100.py --main_optimizer 'sgd'  --lr_epochs 60 --cl_epochs 60 --main_online_lr 1e-3 --train-batch-size 64 --cl_method 'si' --num_tasks 10 --si_lam  1e1 --cl_dataset 'split_cifar100' --rank 2 &
python main_cifar100.py --main_optimizer 'sgd'  --lr_epochs 60 --cl_epochs 60 --main_online_lr 1e-3 --train-batch-size 64 --cl_method 'si' --num_tasks 10 --si_lam  1e2 --cl_dataset 'split_cifar100' --rank 3 &
wait

python main_cifar100.py --main_optimizer 'sgd'  --lr_epochs 60 --cl_epochs 60 --fisher_update_after 50 --main_online_lr 1e-3 --train-batch-size 64 --num_tasks 10 --cl_method 'rwalk'  --rwalk_lam 1  --cl_dataset 'split_cifar100' --rank 0 &
python main_cifar100.py --main_optimizer 'sgd'  --lr_epochs 60 --cl_epochs 60 --fisher_update_after 50 --main_online_lr 1e-3 --train-batch-size 64 --num_tasks 10 --cl_method 'rwalk'  --rwalk_lam 2  --cl_dataset 'split_cifar100' --rank 1 &
python main_cifar100.py --main_optimizer 'sgd'  --lr_epochs 60 --cl_epochs 60 --fisher_update_after 50 --main_online_lr 1e-3 --train-batch-size 64 --num_tasks 10 --cl_method 'rwalk'  --rwalk_lam 5  --cl_dataset 'split_cifar100' --rank 2 &
python main_cifar100.py --main_optimizer 'sgd'  --lr_epochs 60 --cl_epochs 60 --fisher_update_after 50 --main_online_lr 1e-3 --train-batch-size 64 --num_tasks 10 --cl_method 'rwalk'  --rwalk_lam 10 --cl_dataset 'split_cifar100' --rank 3 &
wait

python main_cifar100.py --main_optimizer 'sgd'  --ae_epochs 200 --lr_epochs 60 --cl_epochs 60 --main_online_lr 1e-3 --train-batch-size 64 --num_tasks 10 --cl_dataset 'split_cifar100' --ae_grad_norm 1 --rank 0 \
            --mlp_saved_iterations 128 --ae_offline_lr 1e-2 --ae_cl_lam 1000 --ae_re_lam 100 --ae_topk 1000 --cl_method 'dco' --ae_what 'M' --push_cone_l2 0.2
wait