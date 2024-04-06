#!/bin/bash

if [ "$1" == 'permuted_mnist' ]; then
python main_mnist.py --main_optimizer 'sgd'  --lr_epochs 10 --cl_epochs 10 --main_online_lr 1e-2 --train-batch-size 128 --wd 1e-3 --cl_method 'sgd'  --cl_dataset $1 --num_tasks 5 --rank 0 &
python main_mnist.py --main_optimizer 'sgd'  --lr_epochs 10 --cl_epochs 10 --main_online_lr 1e-3 --train-batch-size 128 --wd 1e-3 --cl_method 'sgd'  --cl_dataset $1 --num_tasks 5 --rank 1 &
python main_mnist.py --main_optimizer 'sgd'  --lr_epochs 10 --cl_epochs 10 --main_online_lr 1e-4 --train-batch-size 128 --wd 1e-3 --cl_method 'sgd'  --cl_dataset $1 --num_tasks 5 --rank 2 &
python main_mnist.py --main_optimizer 'sgd'  --lr_epochs 10 --cl_epochs 10 --main_online_lr 1e-5 --train-batch-size 128 --wd 1e-3 --cl_method 'sgd'  --cl_dataset $1 --num_tasks 5 --rank 3 &
wait

python main_mnist.py --main_optimizer 'sgd'  --lr_epochs 10 --cl_epochs 10 --main_online_lr 1e-3 --train-batch-size 128 --wd 1e-3 --cl_method 'ewc' --num_tasks 5 --ewc_lam 10  --cl_dataset $1 --rank 0 &
python main_mnist.py --main_optimizer 'sgd'  --lr_epochs 10 --cl_epochs 10 --main_online_lr 1e-3 --train-batch-size 128 --wd 1e-3 --cl_method 'ewc' --num_tasks 5 --ewc_lam 20  --cl_dataset $1 --rank 1 &
python main_mnist.py --main_optimizer 'sgd'  --lr_epochs 10 --cl_epochs 10 --main_online_lr 1e-3 --train-batch-size 128 --wd 1e-3 --cl_method 'ewc' --num_tasks 5 --ewc_lam 50  --cl_dataset $1 --rank 2 &
python main_mnist.py --main_optimizer 'sgd'  --lr_epochs 10 --cl_epochs 10 --main_online_lr 1e-3 --train-batch-size 128 --wd 1e-3 --cl_method 'ewc' --num_tasks 5 --ewc_lam 100 --cl_dataset $1 --rank 3 &
wait

python main_mnist.py --main_optimizer 'sgd'  --lr_epochs 10 --cl_epochs 10 --main_online_lr 1e-3 --train-batch-size 128 --wd 1e-3 --cl_method 'si' --num_tasks 5 --si_lam 1e0  --cl_dataset $1 --rank 0 &
python main_mnist.py --main_optimizer 'sgd'  --lr_epochs 10 --cl_epochs 10 --main_online_lr 1e-3 --train-batch-size 128 --wd 1e-3 --cl_method 'si' --num_tasks 5 --si_lam 1e1  --cl_dataset $1 --rank 1 &
python main_mnist.py --main_optimizer 'sgd'  --lr_epochs 10 --cl_epochs 10 --main_online_lr 1e-3 --train-batch-size 128 --wd 1e-3 --cl_method 'si' --num_tasks 5 --si_lam 1e2  --cl_dataset $1 --rank 2 &
python main_mnist.py --main_optimizer 'sgd'  --lr_epochs 10 --cl_epochs 10 --main_online_lr 1e-3 --train-batch-size 128 --wd 1e-3 --cl_method 'si' --num_tasks 5 --si_lam 1e3  --cl_dataset $1 --rank 3 &
wait

python main_mnist.py --main_optimizer 'sgd'  --lr_epochs 10 --cl_epochs 10 --fisher_update_after 10 --main_online_lr 1e-3 --train-batch-size 128 --wd 1e-3 --num_tasks 5 --cl_method 'rwalk'  --rwalk_lam 1e-2 --cl_dataset $1 --rank 0 &
python main_mnist.py --main_optimizer 'sgd'  --lr_epochs 10 --cl_epochs 10 --fisher_update_after 10 --main_online_lr 1e-3 --train-batch-size 128 --wd 1e-3 --num_tasks 5 --cl_method 'rwalk'  --rwalk_lam 1e-1 --cl_dataset $1 --rank 1 &
python main_mnist.py --main_optimizer 'sgd'  --lr_epochs 10 --cl_epochs 10 --fisher_update_after 10 --main_online_lr 1e-3 --train-batch-size 128 --wd 1e-3 --num_tasks 5 --cl_method 'rwalk'  --rwalk_lam 1e-0 --cl_dataset $1 --rank 2 &
python main_mnist.py --main_optimizer 'sgd'  --lr_epochs 10 --cl_epochs 10 --fisher_update_after 10 --main_online_lr 1e-3 --train-batch-size 128 --wd 1e-3 --num_tasks 5 --cl_method 'rwalk'  --rwalk_lam  1e1 --cl_dataset $1 --rank 3 &
wait

python main_mnist.py --main_optimizer 'sgd'  --ae_epochs 200 --lr_epochs 10 --cl_epochs 10 --main_online_lr 1e-3 --train-batch-size 128 --wd 1e-3 --num_tasks 5 --cl_dataset $1 --rank 0 \
            --mlp_saved_iterations 128 --ae_offline_lr 1e-2 --ae_cl_lam 100 --ae_re_lam 100 --ae_topk 1000 --cl_method 'dco' --ae_what 'M' --push_cone_l2 0.2

elif [ "$1" == 'split_mnist' ]; then
python main_mnist.py --main_optimizer 'sgd'  --lr_epochs 10 --cl_epochs 10 --main_online_lr 1e-2 --train-batch-size 128 --wd 1e-3 --cl_method 'sgd'  --cl_dataset $1 --num_tasks 5 --rank 0 &
python main_mnist.py --main_optimizer 'sgd'  --lr_epochs 10 --cl_epochs 10 --main_online_lr 1e-3 --train-batch-size 128 --wd 1e-3 --cl_method 'sgd'  --cl_dataset $1 --num_tasks 5 --rank 1 &
python main_mnist.py --main_optimizer 'sgd'  --lr_epochs 10 --cl_epochs 10 --main_online_lr 1e-4 --train-batch-size 128 --wd 1e-3 --cl_method 'sgd'  --cl_dataset $1 --num_tasks 5 --rank 2 &
python main_mnist.py --main_optimizer 'sgd'  --lr_epochs 10 --cl_epochs 10 --main_online_lr 1e-5 --train-batch-size 128 --wd 1e-3 --cl_method 'sgd'  --cl_dataset $1 --num_tasks 5 --rank 3 &
wait

python main_mnist.py --main_optimizer 'sgd'  --lr_epochs 10 --cl_epochs 10 --main_online_lr 1e-3 --train-batch-size 128 --wd 1e-3 --cl_method 'ewc' --num_tasks 5 --ewc_lam 10  --cl_dataset $1 --rank 0 &
python main_mnist.py --main_optimizer 'sgd'  --lr_epochs 10 --cl_epochs 10 --main_online_lr 1e-3 --train-batch-size 128 --wd 1e-3 --cl_method 'ewc' --num_tasks 5 --ewc_lam 20  --cl_dataset $1 --rank 1 &
python main_mnist.py --main_optimizer 'sgd'  --lr_epochs 10 --cl_epochs 10 --main_online_lr 1e-3 --train-batch-size 128 --wd 1e-3 --cl_method 'ewc' --num_tasks 5 --ewc_lam 50  --cl_dataset $1 --rank 2 &
python main_mnist.py --main_optimizer 'sgd'  --lr_epochs 10 --cl_epochs 10 --main_online_lr 1e-3 --train-batch-size 128 --wd 1e-3 --cl_method 'ewc' --num_tasks 5 --ewc_lam 100 --cl_dataset $1 --rank 3 &
wait

python main_mnist.py --main_optimizer 'sgd'  --lr_epochs 10 --cl_epochs 10 --main_online_lr 1e-3 --train-batch-size 128 --wd 1e-3 --cl_method 'si' --num_tasks 5 --si_lam 1e3  --cl_dataset $1 --rank 0 &
python main_mnist.py --main_optimizer 'sgd'  --lr_epochs 10 --cl_epochs 10 --main_online_lr 1e-3 --train-batch-size 128 --wd 1e-3 --cl_method 'si' --num_tasks 5 --si_lam 1e4  --cl_dataset $1 --rank 1 &
python main_mnist.py --main_optimizer 'sgd'  --lr_epochs 10 --cl_epochs 10 --main_online_lr 1e-3 --train-batch-size 128 --wd 1e-3 --cl_method 'si' --num_tasks 5 --si_lam 1e5   --cl_dataset $1 --rank 2 &
python main_mnist.py --main_optimizer 'sgd'  --lr_epochs 10 --cl_epochs 10 --main_online_lr 1e-3 --train-batch-size 128 --wd 1e-3 --cl_method 'si' --num_tasks 5 --si_lam 1e6   --cl_dataset $1 --rank 3 &
wait

python main_mnist.py --main_optimizer 'sgd'  --lr_epochs 10 --cl_epochs 10 --fisher_update_after 10 --main_online_lr 1e-3 --train-batch-size 128 --wd 1e-3 --num_tasks 5 --cl_method 'rwalk'  --rwalk_lam 1e-2 --cl_dataset $1 --rank 0 &
python main_mnist.py --main_optimizer 'sgd'  --lr_epochs 10 --cl_epochs 10 --fisher_update_after 10 --main_online_lr 1e-3 --train-batch-size 128 --wd 1e-3 --num_tasks 5 --cl_method 'rwalk'  --rwalk_lam 1e-1 --cl_dataset $1 --rank 1 &
python main_mnist.py --main_optimizer 'sgd'  --lr_epochs 10 --cl_epochs 10 --fisher_update_after 10 --main_online_lr 1e-3 --train-batch-size 128 --wd 1e-3 --num_tasks 5 --cl_method 'rwalk'  --rwalk_lam 1e-0 --cl_dataset $1 --rank 2 &
python main_mnist.py --main_optimizer 'sgd'  --lr_epochs 10 --cl_epochs 10 --fisher_update_after 10 --main_online_lr 1e-3 --train-batch-size 128 --wd 1e-3 --num_tasks 5 --cl_method 'rwalk'  --rwalk_lam  1e1 --cl_dataset $1 --rank 3 &
wait

python main_mnist.py --main_optimizer 'sgd'  --ae_epochs 200 --lr_epochs 10 --cl_epochs 10 --main_online_lr 1e-3 --train-batch-size 128 --wd 1e-3 --num_tasks 5 --cl_dataset $1 --rank 0 \
            --mlp_saved_iterations 128 --ae_offline_lr 1e-2 --ae_cl_lam 100 --ae_re_lam 100 --ae_topk 1000 --cl_method 'dco' --ae_what 'M' --push_cone_l2 0.2
fi