# Dynamic Architecture (CL ERROR = 0)
python ../main_mnist.py --main_optimizer 'sgd'  --lr_epochs 5 --cl_epochs 5 --main_online_lr 1e-3 --train-batch-size 128 --wd 1e-3 --cl_method 'ewc' --num_tasks 5 --ewc_lam 10  --cl_dataset permuted_mnist --rank 0  --cl_error_threshold 0 --max_allowed_added_layers 20 --freeze --added_layer_conf 1,0,0 &

python ../main_mnist.py --main_optimizer 'sgd'  --lr_epochs 5 --cl_epochs 5 --main_online_lr 1e-3 --train-batch-size 128 --wd 1e-3 --cl_method 'ewc' --num_tasks 5 --ewc_lam 10  --cl_dataset permuted_mnist --rank 0  --cl_error_threshold 0 --max_allowed_added_layers 20 --freeze --added_layer_conf 2,0,0 &

python ../main_mnist.py --main_optimizer 'sgd'  --lr_epochs 5 --cl_epochs 5 --main_online_lr 1e-3 --train-batch-size 128 --wd 1e-3 --cl_method 'ewc' --num_tasks 5 --ewc_lam 10  --cl_dataset permuted_mnist --rank 0  --cl_error_threshold 0 --max_allowed_added_layers 20 --freeze --added_layer_conf 3,0,0 &

python ../main_mnist.py --main_optimizer 'sgd'  --lr_epochs 5 --cl_epochs 5 --main_online_lr 1e-3 --train-batch-size 128 --wd 1e-3 --cl_method 'ewc' --num_tasks 5 --ewc_lam 10  --cl_dataset permuted_mnist --rank 0  --cl_error_threshold 0 --max_allowed_added_layers 20 --freeze --added_layer_conf 4,0,0 &

python ../main_mnist.py --main_optimizer 'sgd'  --lr_epochs 5 --cl_epochs 5 --main_online_lr 1e-3 --train-batch-size 128 --wd 1e-3 --cl_method 'ewc' --num_tasks 5 --ewc_lam 10  --cl_dataset permuted_mnist --rank 0  --cl_error_threshold 0 --max_allowed_added_layers 20 --freeze --added_layer_conf 5,0,0 &
wait