

# Permuted Mnist

## SGD (0 added layers)
python main_mnist.py --main_optimizer 'sgd'  --lr_epochs 10 --cl_epochs 10 --main_online_lr 1e-2 --train-batch-size 128 --wd 1e-3 --cl_method 'sgd'  --cl_dataset permuted_mnist --num_tasks 5 --rank 0 --cl_error_threshold 10 --max_allowed_added_layers 0

## SGD (1 added layers)

### Freeze
python main_mnist.py --main_optimizer 'sgd'  --lr_epochs 10 --cl_epochs 10 --main_online_lr 1e-2 --train-batch-size 128 --wd 1e-3 --cl_method 'sgd'  --cl_dataset permuted_mnist --num_tasks 5 --rank 0 --cl_error_threshold 10 --max_allowed_added_layers 1 --freeze

### No Freeze
python main_mnist.py --main_optimizer 'sgd'  --lr_epochs 10 --cl_epochs 10 --main_online_lr 1e-2 --train-batch-size 128 --wd 1e-3 --cl_method 'sgd'  --cl_dataset permuted_mnist --num_tasks 5 --rank 0 --cl_error_threshold 10 --max_allowed_added_layers 1 --no-freeze


## SGD (2 added layers)
### Freeze
python main_mnist.py --main_optimizer 'sgd'  --lr_epochs 10 --cl_epochs 10 --main_online_lr 1e-2 --train-batch-size 128 --wd 1e-3 --cl_method 'sgd'  --cl_dataset permuted_mnist --num_tasks 5 --rank 0 --cl_error_threshold 10 --max_allowed_added_layers 2 --freeze

### No Freeze
python main_mnist.py --main_optimizer 'sgd'  --lr_epochs 10 --cl_epochs 10 --main_online_lr 1e-2 --train-batch-size 128 --wd 1e-3 --cl_method 'sgd'  --cl_dataset permuted_mnist --num_tasks 5 --rank 0 --cl_error_threshold 10 --max_allowed_added_layers 2 --no-freeze

## EWC (0 added layers)
python main_mnist.py --main_optimizer 'sgd'  --lr_epochs 10 --cl_epochs 10 --main_online_lr 1e-3 --train-batch-size 128 --wd 1e-3 --cl_method 'ewc' --num_tasks 5 --ewc_lam 10  --cl_dataset permuted_mnist --rank 0 --cl_error_threshold 8 --max_allowed_added_layers 0

## EWC (1 added layers)
### Freeze
python main_mnist.py --main_optimizer 'sgd'  --lr_epochs 10 --cl_epochs 10 --main_online_lr 1e-3 --train-batch-size 128 --wd 1e-3 --cl_method 'ewc' --num_tasks 5 --ewc_lam 10  --cl_dataset permuted_mnist --rank 0 --cl_error_threshold 8 --max_allowed_added_layers 1 --freeze

### No Freeze
python main_mnist.py --main_optimizer 'sgd'  --lr_epochs 10 --cl_epochs 10 --main_online_lr 1e-3 --train-batch-size 128 --wd 1e-3 --cl_method 'ewc' --num_tasks 5 --ewc_lam 10  --cl_dataset permuted_mnist --rank 0 --cl_error_threshold 8 --max_allowed_added_layers 1 --no-freeze


## EWC (2 added layers)

### Freeze
python main_mnist.py --main_optimizer 'sgd'  --lr_epochs 10 --cl_epochs 10 --main_online_lr 1e-3 --train-batch-size 128 --wd 1e-3 --cl_method 'ewc' --num_tasks 5 --ewc_lam 10  --cl_dataset permuted_mnist --rank 0 --cl_error_threshold 8 --max_allowed_added_layers 2

### No Freeze
python main_mnist.py --main_optimizer 'sgd'  --lr_epochs 10 --cl_epochs 10 --main_online_lr 1e-3 --train-batch-size 128 --wd 1e-3 --cl_method 'ewc' --num_tasks 5 --ewc_lam 10  --cl_dataset permuted_mnist --rank 0 --cl_error_threshold 8 --max_allowed_added_layers 2 --no-freeze

# Split mnist
## SGD (0 added layers)
python main_mnist.py --main_optimizer 'sgd'  --lr_epochs 10 --cl_epochs 10 --main_online_lr 1e-2 --train-batch-size 128 --wd 1e-3 --cl_method 'sgd'  --cl_dataset split_mnist --num_tasks 5 --rank 0 --cl_error_threshold 10 --max_allowed_added_layers 0

## SGD (2 added layers)
### Freeze
python main_mnist.py --main_optimizer 'sgd'  --lr_epochs 10 --cl_epochs 10 --main_online_lr 1e-2 --train-batch-size 128 --wd 1e-3 --cl_method 'sgd'  --cl_dataset split_mnist --num_tasks 5 --rank 0 --cl_error_threshold 10 --max_allowed_added_layers 2 --freeze

### No Freeze
python main_mnist.py --main_optimizer 'sgd'  --lr_epochs 10 --cl_epochs 10 --main_online_lr 1e-2 --train-batch-size 128 --wd 1e-3 --cl_method 'sgd'  --cl_dataset split_mnist --num_tasks 5 --rank 0 --cl_error_threshold 10 --max_allowed_added_layers 2 --no-freeze



