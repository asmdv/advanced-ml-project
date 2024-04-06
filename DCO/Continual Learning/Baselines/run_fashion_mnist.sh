#!/bin/bash

# MNIST ---> Fashion MNIST
python main_fashion_mnist.py --main_optimizer 'sgd' --lr_epochs 10 --cl_epochs 10 --main_online_lr 1e-2 --train-batch-size 128 --cl_method 'sgd'  --cl_dataset 'to_fashion_mnist' --num_tasks 2 --rank 0 &
python main_fashion_mnist.py --main_optimizer 'sgd' --lr_epochs 10 --cl_epochs 10 --main_online_lr 1e-3 --train-batch-size 128 --cl_method 'sgd'  --cl_dataset 'to_fashion_mnist' --num_tasks 2 --rank 1 &
python main_fashion_mnist.py --main_optimizer 'sgd' --lr_epochs 10 --cl_epochs 10 --main_online_lr 1e-4 --train-batch-size 128 --cl_method 'sgd'  --cl_dataset 'to_fashion_mnist' --num_tasks 2 --rank 2 &
python main_fashion_mnist.py --main_optimizer 'sgd' --lr_epochs 10 --cl_epochs 10 --main_online_lr 1e-5 --train-batch-size 128 --cl_method 'sgd'  --cl_dataset 'to_fashion_mnist' --num_tasks 2 --rank 3 &
wait

python main_fashion_mnist.py --main_optimizer 'sgd' --lr_epochs 10 --cl_epochs 10 --main_online_lr 1e-3 --train-batch-size 128 --cl_method 'ewc' --num_tasks 2 --ewc_lam 1e2 --cl_dataset 'to_fashion_mnist' --rank 0 &
python main_fashion_mnist.py --main_optimizer 'sgd' --lr_epochs 10 --cl_epochs 10 --main_online_lr 1e-3 --train-batch-size 128 --cl_method 'ewc' --num_tasks 2 --ewc_lam 1e3 --cl_dataset 'to_fashion_mnist' --rank 1 &
python main_fashion_mnist.py --main_optimizer 'sgd' --lr_epochs 10 --cl_epochs 10 --main_online_lr 1e-3 --train-batch-size 128 --cl_method 'ewc' --num_tasks 2 --ewc_lam 1e4 --cl_dataset 'to_fashion_mnist' --rank 2 &
python main_fashion_mnist.py --main_optimizer 'sgd' --lr_epochs 10 --cl_epochs 10 --main_online_lr 1e-3 --train-batch-size 128 --cl_method 'ewc' --num_tasks 2 --ewc_lam 1e5 --cl_dataset 'to_fashion_mnist' --rank 3 &
wait

python main_fashion_mnist.py --main_optimizer 'sgd' --lr_epochs 10 --cl_epochs 10 --main_online_lr 1e-3 --train-batch-size 128 --cl_method 'si' --num_tasks 2 --si_lam 1e2  --cl_dataset 'to_fashion_mnist' --rank 0 &
python main_fashion_mnist.py --main_optimizer 'sgd' --lr_epochs 10 --cl_epochs 10 --main_online_lr 1e-3 --train-batch-size 128 --cl_method 'si' --num_tasks 2 --si_lam 1e3  --cl_dataset 'to_fashion_mnist' --rank 1 &
python main_fashion_mnist.py --main_optimizer 'sgd' --lr_epochs 10 --cl_epochs 10 --main_online_lr 1e-3 --train-batch-size 128 --cl_method 'si' --num_tasks 2 --si_lam 1e4  --cl_dataset 'to_fashion_mnist' --rank 2 &
python main_fashion_mnist.py --main_optimizer 'sgd' --lr_epochs 10 --cl_epochs 10 --main_online_lr 1e-3 --train-batch-size 128 --cl_method 'si' --num_tasks 2 --si_lam 1e5  --cl_dataset 'to_fashion_mnist' --rank 3 &
wait

python main_fashion_mnist.py --main_optimizer 'sgd' --lr_epochs 10 --cl_epochs 10 --fisher_update_after 10 --main_online_lr 1e-3 --train-batch-size 128 --num_tasks 2 --cl_method 'rwalk'  --rwalk_lam 1e-2 --cl_dataset 'to_fashion_mnist' --rank 0 &
python main_fashion_mnist.py --main_optimizer 'sgd' --lr_epochs 10 --cl_epochs 10 --fisher_update_after 10 --main_online_lr 1e-3 --train-batch-size 128 --num_tasks 2 --cl_method 'rwalk'  --rwalk_lam 1e-1 --cl_dataset 'to_fashion_mnist' --rank 1 &
python main_fashion_mnist.py --main_optimizer 'sgd' --lr_epochs 10 --cl_epochs 10 --fisher_update_after 10 --main_online_lr 1e-3 --train-batch-size 128 --num_tasks 2 --cl_method 'rwalk'  --rwalk_lam 1e-0 --cl_dataset 'to_fashion_mnist' --rank 2 &
python main_fashion_mnist.py --main_optimizer 'sgd' --lr_epochs 10 --cl_epochs 10 --fisher_update_after 10 --main_online_lr 1e-3 --train-batch-size 128 --num_tasks 2 --cl_method 'rwalk'  --rwalk_lam  1e1 --cl_dataset 'to_fashion_mnist' --rank 3 &
wait

python main_fashion_mnist.py --main_optimizer 'sgd' --ae_epochs 50 --lr_epochs 10 --cl_epochs 10 --main_online_lr 1e-3 --train-batch-size 128 --num_tasks 2 --cl_dataset 'to_fashion_mnist' --ae_grad_norm 2000 --rank 0 \
            --mlp_saved_iterations 128 --ae_offline_lr 1e-2 --ae_cl_lam 10 --ae_re_lam 100 --ae_topk 1000 --cl_method 'dco' --ae_what 'M' --push_cone_l2 2.0 --l2_regularization 0.001


# Fashion MNIST ---> MNIST
python main_fashion_mnist.py --main_optimizer 'sgd' --lr_epochs 10 --cl_epochs 10 --main_online_lr 1e-2 --train-batch-size 128 --cl_method 'sgd'  --cl_dataset 'from_fashion_mnist' --num_tasks 2 --rank 0 &
python main_fashion_mnist.py --main_optimizer 'sgd' --lr_epochs 10 --cl_epochs 10 --main_online_lr 1e-3 --train-batch-size 128 --cl_method 'sgd'  --cl_dataset 'from_fashion_mnist' --num_tasks 2 --rank 1 &
python main_fashion_mnist.py --main_optimizer 'sgd' --lr_epochs 10 --cl_epochs 10 --main_online_lr 1e-4 --train-batch-size 128 --cl_method 'sgd'  --cl_dataset 'from_fashion_mnist' --num_tasks 2 --rank 2 &
python main_fashion_mnist.py --main_optimizer 'sgd' --lr_epochs 10 --cl_epochs 10 --main_online_lr 1e-5 --train-batch-size 128 --cl_method 'sgd'  --cl_dataset 'from_fashion_mnist' --num_tasks 2 --rank 3 &
wait

python main_fashion_mnist.py --main_optimizer 'sgd' --lr_epochs 10 --cl_epochs 10 --main_online_lr 1e-3 --train-batch-size 128 --cl_method 'ewc' --num_tasks 2 --ewc_lam 1e2 --cl_dataset 'from_fashion_mnist' --rank 0 &
python main_fashion_mnist.py --main_optimizer 'sgd' --lr_epochs 10 --cl_epochs 10 --main_online_lr 1e-3 --train-batch-size 128 --cl_method 'ewc' --num_tasks 2 --ewc_lam 1e3 --cl_dataset 'from_fashion_mnist' --rank 1 &
python main_fashion_mnist.py --main_optimizer 'sgd' --lr_epochs 10 --cl_epochs 10 --main_online_lr 1e-3 --train-batch-size 128 --cl_method 'ewc' --num_tasks 2 --ewc_lam 1e4 --cl_dataset 'from_fashion_mnist' --rank 2 &
python main_fashion_mnist.py --main_optimizer 'sgd' --lr_epochs 10 --cl_epochs 10 --main_online_lr 1e-3 --train-batch-size 128 --cl_method 'ewc' --num_tasks 2 --ewc_lam 1e5 --cl_dataset 'from_fashion_mnist' --rank 3 &
wait

python main_fashion_mnist.py --main_optimizer 'sgd' --lr_epochs 10 --cl_epochs 10 --main_online_lr 1e-3 --train-batch-size 128 --cl_method 'si' --num_tasks 2 --si_lam 1e2  --cl_dataset 'from_fashion_mnist' --rank 0 &
python main_fashion_mnist.py --main_optimizer 'sgd' --lr_epochs 10 --cl_epochs 10 --main_online_lr 1e-3 --train-batch-size 128 --cl_method 'si' --num_tasks 2 --si_lam 1e3  --cl_dataset 'from_fashion_mnist' --rank 1 &
python main_fashion_mnist.py --main_optimizer 'sgd' --lr_epochs 10 --cl_epochs 10 --main_online_lr 1e-3 --train-batch-size 128 --cl_method 'si' --num_tasks 2 --si_lam 1e4  --cl_dataset 'from_fashion_mnist' --rank 2 &
python main_fashion_mnist.py --main_optimizer 'sgd' --lr_epochs 10 --cl_epochs 10 --main_online_lr 1e-3 --train-batch-size 128 --cl_method 'si' --num_tasks 2 --si_lam 1e5  --cl_dataset 'from_fashion_mnist' --rank 3 &
wait

python main_fashion_mnist.py --main_optimizer 'sgd' --lr_epochs 10 --cl_epochs 10 --fisher_update_after 10 --main_online_lr 1e-3 --train-batch-size 128 --num_tasks 2 --cl_method 'rwalk'  --rwalk_lam 1e-2 --cl_dataset 'from_fashion_mnist' --rank 0 &
python main_fashion_mnist.py --main_optimizer 'sgd' --lr_epochs 10 --cl_epochs 10 --fisher_update_after 10 --main_online_lr 1e-3 --train-batch-size 128 --num_tasks 2 --cl_method 'rwalk'  --rwalk_lam 1e-1 --cl_dataset 'from_fashion_mnist' --rank 1 &
python main_fashion_mnist.py --main_optimizer 'sgd' --lr_epochs 10 --cl_epochs 10 --fisher_update_after 10 --main_online_lr 1e-3 --train-batch-size 128 --num_tasks 2 --cl_method 'rwalk'  --rwalk_lam 1e-0 --cl_dataset 'from_fashion_mnist' --rank 2 &
python main_fashion_mnist.py --main_optimizer 'sgd' --lr_epochs 10 --cl_epochs 10 --fisher_update_after 10 --main_online_lr 1e-3 --train-batch-size 128 --num_tasks 2 --cl_method 'rwalk'  --rwalk_lam  1e1 --cl_dataset 'from_fashion_mnist' --rank 3 &
wait

python main_fashion_mnist.py --main_optimizer 'sgd' --ae_epochs 50 --lr_epochs 10 --cl_epochs 10 --main_online_lr 1e-3 --train-batch-size 128 --num_tasks 2 --cl_dataset 'from_fashion_mnist' --ae_grad_norm 2000 --rank 0 \
            --mlp_saved_iterations 128 --ae_offline_lr 1e-2 --ae_cl_lam 10 --ae_re_lam 100 --ae_topk 1000 --cl_method 'dco' --ae_what 'M' --push_cone_l2 2.0 --l2_regularization 0.001