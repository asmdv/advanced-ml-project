# Baseline
python main_mnist.py --main_optimizer 'sgd'  --lr_epochs 5 --cl_epochs 5 --main_online_lr 1e-3 --train-batch-size 128 --wd 1e-3 --cl_method 'ewc' --num_tasks 5 --ewc_lam 10  --cl_dataset permuted_mnist --rank 0 &
wait