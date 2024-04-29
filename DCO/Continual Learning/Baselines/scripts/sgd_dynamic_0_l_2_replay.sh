# Replay
python ../main_mnist.py --main_optimizer 'sgd'  --lr_epochs 5 --cl_epochs 5 --main_online_lr 1e-2 --train-batch-size 128 --wd 1e-3 --cl_method 'sgd'  --cl_dataset permuted_mnist --num_tasks 5 --rank 0  --cl_error_threshold 0 --max_allowed_added_layers 20 --freeze --added_layer_conf 2,0,0 --replay_buffer_batch_size 5 &

python ../main_mnist.py --main_optimizer 'sgd'  --lr_epochs 5 --cl_epochs 5 --main_online_lr 1e-2 --train-batch-size 128 --wd 1e-3 --cl_method 'sgd'  --cl_dataset permuted_mnist --num_tasks 5 --rank 0  --cl_error_threshold 0 --max_allowed_added_layers 20 --freeze --added_layer_conf 2,0,0 --replay_buffer_batch_size 10 &

python ../main_mnist.py --main_optimizer 'sgd'  --lr_epochs 5 --cl_epochs 5 --main_online_lr 1e-2 --train-batch-size 128 --wd 1e-3 --cl_method 'sgd'  --cl_dataset permuted_mnist --num_tasks 5 --rank 0  --cl_error_threshold 0 --max_allowed_added_layers 20 --freeze --added_layer_conf 2,0,0 --replay_buffer_batch_size 20 &

python ../main_mnist.py --main_optimizer 'sgd'  --lr_epochs 5 --cl_epochs 5 --main_online_lr 1e-2 --train-batch-size 128 --wd 1e-3 --cl_method 'sgd'  --cl_dataset permuted_mnist --num_tasks 5 --rank 0  --cl_error_threshold 0 --max_allowed_added_layers 20 --freeze --added_layer_conf 2,0,0 --replay_buffer_batch_size 40 &

python ../main_mnist.py --main_optimizer 'sgd'  --lr_epochs 5 --cl_epochs 5 --main_online_lr 1e-2 --train-batch-size 128 --wd 1e-3 --cl_method 'sgd'  --cl_dataset permuted_mnist --num_tasks 5 --rank 0  --cl_error_threshold 0 --max_allowed_added_layers 20 --freeze --added_layer_conf 2,0,0 --replay_buffer_batch_size 64 &
wait