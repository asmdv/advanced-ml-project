

# Permuted Mnist

## SGD (0 added layers)
### No replay
python main_mnist.py --main_optimizer 'sgd'  --lr_epochs 10 --cl_epochs 10 --main_online_lr 1e-2 --train-batch-size 128 --wd 1e-3 --cl_method 'sgd'  --cl_dataset permuted_mnist --num_tasks 5 --rank 0 --cl_error_threshold 10 --max_allowed_added_layers 0
### Replay 5
python main_mnist.py --main_optimizer 'sgd'  --lr_epochs 10 --cl_epochs 10 --main_online_lr 1e-2 --train-batch-size 128 --wd 1e-3 --cl_method 'sgd'  --cl_dataset permuted_mnist --num_tasks 5 --rank 0 --cl_error_threshold 10 --max_allowed_added_layers 0 --replay_buffer_batch_size 5

### Replay 20
python main_mnist.py --main_optimizer 'sgd'  --lr_epochs 10 --cl_epochs 10 --main_online_lr 1e-2 --train-batch-size 128 --wd 1e-3 --cl_method 'sgd'  --cl_dataset permuted_mnist --num_tasks 5 --rank 0 --cl_error_threshold 10 --max_allowed_added_layers 0 --replay_buffer_batch_size 20

### Replay 64
python main_mnist.py --main_optimizer 'sgd'  --lr_epochs 10 --cl_epochs 10 --main_online_lr 1e-2 --train-batch-size 128 --wd 1e-3 --cl_method 'sgd'  --cl_dataset permuted_mnist --num_tasks 5 --rank 0 --cl_error_threshold 10 --max_allowed_added_layers 0 --replay_buffer_batch_size 64

### Replay 128
python main_mnist.py --main_optimizer 'sgd'  --lr_epochs 10 --cl_epochs 10 --main_online_lr 1e-2 --train-batch-size 128 --wd 1e-3 --cl_method 'sgd'  --cl_dataset permuted_mnist --num_tasks 5 --rank 0 --cl_error_threshold 10 --max_allowed_added_layers 0 --replay_buffer_batch_size 128

# EWC
### No replay
python main_mnist.py --main_optimizer 'sgd'  --lr_epochs 10 --cl_epochs 10 --main_online_lr 1e-3 --train-batch-size 128 --wd 1e-3 --cl_method 'ewc' --num_tasks 5 --ewc_lam 10  --cl_dataset permuted_mnist --rank 0 --cl_error_threshold 8 --max_allowed_added_layers 0

### Replay 5
python main_mnist.py --main_optimizer 'sgd'  --lr_epochs 10 --cl_epochs 10 --main_online_lr 1e-3 --train-batch-size 128 --wd 1e-3 --cl_method 'ewc' --num_tasks 5 --ewc_lam 10  --cl_dataset permuted_mnist --rank 0 --cl_error_threshold 8 --max_allowed_added_layers 0 --replay_buffer_batch_size 5

### Replay 20
python main_mnist.py --main_optimizer 'sgd'  --lr_epochs 10 --cl_epochs 10 --main_online_lr 1e-3 --train-batch-size 128 --wd 1e-3 --cl_method 'ewc' --num_tasks 5 --ewc_lam 10  --cl_dataset permuted_mnist --rank 0 --cl_error_threshold 8 --max_allowed_added_layers 0 --replay_buffer_batch_size 20

### Replay 128
python main_mnist.py --main_optimizer 'sgd'  --lr_epochs 10 --cl_epochs 10 --main_online_lr 1e-3 --train-batch-size 128 --wd 1e-3 --cl_method 'ewc' --num_tasks 5 --ewc_lam 10  --cl_dataset permuted_mnist --rank 0 --cl_error_threshold 8 --max_allowed_added_layers 0 --replay_buffer_batch_size 128



# Split Mnist
## SGD
### No replay
python main_mnist.py --main_optimizer 'sgd'  --lr_epochs 10 --cl_epochs 10 --main_online_lr 1e-2 --train-batch-size 128 --wd 1e-3 --cl_method 'sgd'  --cl_dataset split_mnist --num_tasks 5 --rank 0 --cl_error_threshold 10 --max_allowed_added_layers 0
### Replay 5
python main_mnist.py --main_optimizer 'sgd'  --lr_epochs 10 --cl_epochs 10 --main_online_lr 1e-2 --train-batch-size 128 --wd 1e-3 --cl_method 'sgd'  --cl_dataset split_mnist --num_tasks 5 --rank 0 --cl_error_threshold 10 --max_allowed_added_layers 0 --replay_buffer_batch_size 5

### Replay 20
python main_mnist.py --main_optimizer 'sgd'  --lr_epochs 10 --cl_epochs 10 --main_online_lr 1e-2 --train-batch-size 128 --wd 1e-3 --cl_method 'sgd'  --cl_dataset split_mnist --num_tasks 5 --rank 0 --cl_error_threshold 10 --max_allowed_added_layers 0 --replay_buffer_batch_size 20

### Replay 64
python main_mnist.py --main_optimizer 'sgd'  --lr_epochs 10 --cl_epochs 10 --main_online_lr 1e-2 --train-batch-size 128 --wd 1e-3 --cl_method 'sgd'  --cl_dataset split_mnist --num_tasks 5 --rank 0 --cl_error_threshold 10 --max_allowed_added_layers 0 --replay_buffer_batch_size 64

### Replay 128
python main_mnist.py --main_optimizer 'sgd'  --lr_epochs 10 --cl_epochs 10 --main_online_lr 1e-2 --train-batch-size 128 --wd 1e-3 --cl_method 'sgd'  --cl_dataset split_mnist --num_tasks 5 --rank 0 --cl_error_threshold 10 --max_allowed_added_layers 0 --replay_buffer_batch_size 128


## EWC
### No replay
python main_mnist.py --main_optimizer 'sgd'  --lr_epochs 10 --cl_epochs 10 --main_online_lr 1e-3 --train-batch-size 128 --wd 1e-3 --cl_method 'ewc' --num_tasks 5 --ewc_lam 10  --cl_dataset split_mnist --rank 0 --cl_error_threshold 8 --max_allowed_added_layers 0

### Replay 5
python main_mnist.py --main_optimizer 'sgd'  --lr_epochs 10 --cl_epochs 10 --main_online_lr 1e-3 --train-batch-size 128 --wd 1e-3 --cl_method 'ewc' --num_tasks 5 --ewc_lam 10  --cl_dataset split_mnist --rank 0 --cl_error_threshold 8 --max_allowed_added_layers 0 --replay_buffer_batch_size 5

### Replay 20
python main_mnist.py --main_optimizer 'sgd'  --lr_epochs 10 --cl_epochs 10 --main_online_lr 1e-3 --train-batch-size 128 --wd 1e-3 --cl_method 'ewc' --num_tasks 5 --ewc_lam 10  --cl_dataset split_mnist --rank 0 --cl_error_threshold 8 --max_allowed_added_layers 0 --replay_buffer_batch_size 20

### Replay 128
python main_mnist.py --main_optimizer 'sgd'  --lr_epochs 10 --cl_epochs 10 --main_online_lr 1e-3 --train-batch-size 128 --wd 1e-3 --cl_method 'ewc' --num_tasks 5 --ewc_lam 10  --cl_dataset split_mnist --rank 0 --cl_error_threshold 8 --max_allowed_added_layers 0 --replay_buffer_batch_size 128










## SGD (1 added layers)

### Freeze
python main_mnist.py --main_optimizer 'sgd'  --lr_epochs 10 --cl_epochs 10 --main_online_lr 1e-2 --train-batch-size 128 --wd 1e-3 --cl_method 'sgd'  --cl_dataset permuted_mnist --num_tasks 5 --rank 0 --cl_error_threshold 10 --max_allowed_added_layers 1 --freeze --added_layer_conf 5,0,0

### No Freeze
python main_mnist.py --main_optimizer 'sgd'  --lr_epochs 10 --cl_epochs 10 --main_online_lr 1e-2 --train-batch-size 128 --wd 1e-3 --cl_method 'sgd'  --cl_dataset permuted_mnist --num_tasks 5 --rank 0 --cl_error_threshold 10 --max_allowed_added_layers 2 --no-freeze --added_layer_conf 2,0,0


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
python main_mnist.py --main_optimizer 'sgd'  --lr_epochs 10 --cl_epochs 10 --main_online_lr 1e-3 --train-batch-size 128 --wd 1e-3 --cl_method 'ewc' --num_tasks 5 --ewc_lam 10  --cl_dataset permuted_mnist --rank 0 --cl_error_threshold 8 --max_allowed_added_layers 1 --no-freeze --added_layer_conf 1,0,0


## EWC (2 added layers)

### Freeze
python main_mnist.py --main_optimizer 'sgd'  --lr_epochs 10 --cl_epochs 10 --main_online_lr 1e-3 --train-batch-size 128 --wd 1e-3 --cl_method 'ewc' --num_tasks 5 --ewc_lam 10  --cl_dataset permuted_mnist --rank 0 --cl_error_threshold 8 --max_allowed_added_layers 2

### No Freeze
python main_mnist.py --main_optimizer 'sgd'  --lr_epochs 10 --cl_epochs 10 --main_online_lr 1e-3 --train-batch-size 128 --wd 1e-3 --cl_method 'ewc' --num_tasks 5 --ewc_lam 10  --cl_dataset permuted_mnist --rank 0 --cl_error_threshold 8 --max_allowed_added_layers 2 --no-freeze


## DCO
python main_mnist.py --main_optimizer 'sgd'  --ae_epochs 200 --lr_epochs 10 --cl_epochs 10 --main_online_lr 1e-3 --train-batch-size 128 --wd 1e-3 --num_tasks 5 --cl_dataset permuted_mnist --rank 0 \
            --mlp_saved_iterations 128 --ae_offline_lr 1e-2 --ae_cl_lam 100 --ae_re_lam 100 --ae_topk 1000 --cl_method 'dco' --ae_what 'M' --push_cone_l2 0.2 --cl_error_threshold 8 --max_allowed_added_layers 0 --freeze


# Split mnist
## SGD (0 added layers)
python main_mnist.py --main_optimizer 'sgd'  --lr_epochs 10 --cl_epochs 10 --main_online_lr 1e-2 --train-batch-size 128 --wd 1e-3 --cl_method 'sgd'  --cl_dataset split_mnist --num_tasks 5 --rank 0 --cl_error_threshold 10 --max_allowed_added_layers 0

## SGD (2 added layers)
### Freeze
python main_mnist.py --main_optimizer 'sgd'  --lr_epochs 10 --cl_epochs 10 --main_online_lr 1e-2 --train-batch-size 128 --wd 1e-3 --cl_method 'sgd'  --cl_dataset split_mnist --num_tasks 5 --rank 0 --cl_error_threshold 10 --max_allowed_added_layers 2 --freeze

### No Freeze
python main_mnist.py --main_optimizer 'sgd'  --lr_epochs 10 --cl_epochs 10 --main_online_lr 1e-2 --train-batch-size 128 --wd 1e-3 --cl_method 'sgd'  --cl_dataset split_mnist --num_tasks 5 --rank 0 --cl_error_threshold 10 --max_allowed_added_layers 2 --no-freeze




python main_mnist.py --main_optimizer 'sgd'  --lr_epochs 10 --cl_epochs 10 --main_online_lr 1e-2 --train-batch-size 128 --wd 1e-3 --cl_method 'sgd'  --cl_dataset permuted_mnist --num_tasks 5 --rank 0 --cl_error_threshold 10 --max_allowed_added_layers 0 --replay_buffer_batch_size 20










## Experiments

## SGD 1 max 2 layers freeze
python main_mnist.py --main_optimizer 'sgd'  --lr_epochs 10 --cl_epochs 10 --main_online_lr 1e-2 --train-batch-size 128 --wd 1e-3 --cl_method 'sgd'  --cl_dataset permuted_mnist --num_tasks 5 --rank 0 --cl_error_threshold 10 --max_allowed_added_layers 1 --freeze --added_layer_conf 2,0,0

## SGD 1 max 1 layer freeze
ython main_mnist.py --main_optimizer 'sgd'  --lr_epochs 10 --cl_epochs 10 --main_online_lr 1e-2 --train-batch-size 128 --wd 1e-3 --cl_method 'sgd'  --cl_dataset permuted_mnist --num_tasks 5 --rank 0 --cl_error_threshold 10 --max_allowed_added_layers 1 --freeze --added_layer_conf 1,0,0


## SGD 2 max 2 layers freeze
python main_mnist.py --main_optimizer 'sgd'  --lr_epochs 10 --cl_epochs 10 --main_online_lr 1e-2 --train-batch-size 128 --wd 1e-3 --cl_method 'sgd'  --cl_dataset permuted_mnist --num_tasks 5 --rank 0 --cl_error_threshold 10 --max_allowed_added_layers 2 --freeze --added_layer_conf 2,0,0

## SGD 2 max 1 layer freeze
ython main_mnist.py --main_optimizer 'sgd'  --lr_epochs 10 --cl_epochs 10 --main_online_lr 1e-2 --train-batch-size 128 --wd 1e-3 --cl_method 'sgd'  --cl_dataset permuted_mnist --num_tasks 5 --rank 0 --cl_error_threshold 10 --max_allowed_added_layers 2 --freeze --added_layer_conf 1,0,0


## SGD 1 max 2 layers no freeze
python main_mnist.py --main_optimizer 'sgd'  --lr_epochs 10 --cl_epochs 10 --main_online_lr 1e-2 --train-batch-size 128 --wd 1e-3 --cl_method 'sgd'  --cl_dataset permuted_mnist --num_tasks 5 --rank 0 --cl_error_threshold 10 --max_allowed_added_layers 1 --no-freeze --added_layer_conf 2,0,0


## SGD 1 max 1 layer no freeze
python main_mnist.py --main_optimizer 'sgd'  --lr_epochs 10 --cl_epochs 10 --main_online_lr 1e-2 --train-batch-size 128 --wd 1e-3 --cl_method 'sgd'  --cl_dataset permuted_mnist --num_tasks 5 --rank 0 --cl_error_threshold 10 --max_allowed_added_layers 1 --no-freeze --added_layer_conf 1,0,0


# Replay
## SGD 1 max 2 layers freeze
python main_mnist.py --main_optimizer 'sgd'  --lr_epochs 10 --cl_epochs 10 --main_online_lr 1e-2 --train-batch-size 128 --wd 1e-3 --cl_method 'sgd'  --cl_dataset permuted_mnist --num_tasks 5 --rank 0 --cl_error_threshold 10 --max_allowed_added_layers 1 --freeze --added_layer_conf 2,0,0 --replay_buffer_batch_size 20

## SGD 1 max 1 layer freeze
ython main_mnist.py --main_optimizer 'sgd'  --lr_epochs 10 --cl_epochs 10 --main_online_lr 1e-2 --train-batch-size 128 --wd 1e-3 --cl_method 'sgd'  --cl_dataset permuted_mnist --num_tasks 5 --rank 0 --cl_error_threshold 10 --max_allowed_added_layers 1 --freeze --added_layer_conf 1,0,0 --replay_buffer_batch_size 20




## SGD 1 max 2 layers freeze
python main_mnist.py --main_optimizer 'sgd'  --lr_epochs 10 --cl_epochs 10 --main_online_lr 1e-2 --train-batch-size 128 --wd 1e-3 --cl_method 'sgd'  --cl_dataset split_mnist --num_tasks 5 --rank 0 --cl_error_threshold 10 --max_allowed_added_layers 1 --freeze --added_layer_conf 2,0,0

## SGD 1 max 1 layer freeze
python main_mnist.py --main_optimizer 'sgd'  --lr_epochs 10 --cl_epochs 10 --main_online_lr 1e-2 --train-batch-size 128 --wd 1e-3 --cl_method 'sgd'  --cl_dataset split_mnist --num_tasks 5 --rank 0 --cl_error_threshold 10 --max_allowed_added_layers 1 --freeze --added_layer_conf 1,0,0


### Replay
## SGD 1 max 2 layers freeze replay
python main_mnist.py --main_optimizer 'sgd'  --lr_epochs 10 --cl_epochs 10 --main_online_lr 1e-2 --train-batch-size 128 --wd 1e-3 --cl_method 'sgd'  --cl_dataset split_mnist --num_tasks 5 --rank 0 --cl_error_threshold 10 --max_allowed_added_layers 1 --freeze --added_layer_conf 2,0,0 --replay_buffer_batch_size 20

## SGD 1 max 1 layer freeze replay
ython main_mnist.py --main_optimizer 'sgd'  --lr_epochs 10 --cl_epochs 10 --main_online_lr 1e-2 --train-batch-size 128 --wd 1e-3 --cl_method 'sgd'  --cl_dataset split_mnist --num_tasks 5 --rank 0 --cl_error_threshold 10 --max_allowed_added_layers 1 --freeze --added_layer_conf 1,0,0 --replay_buffer_batch_size 20


1 layer freeze none
20