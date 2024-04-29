# Replay
python ../main_mnist.py --main_optimizer 'sgd'  --lr_epochs 2 --cl_epochs 2 --main_online_lr 1e-2 --train-batch-size 128 --wd 1e-3 --cl_method 'sgd'  --cl_dataset permuted_mnist --num_tasks 5 --rank 0 --replay_buffer_batch_size 5

python ../main_mnist.py --main_optimizer 'sgd'  --lr_epochs 10 --cl_epochs 10 --main_online_lr 1e-2 --train-batch-size 128 --wd 1e-3 --cl_method 'sgd'  --cl_dataset permuted_mnist --num_tasks 5 --rank 0 --replay_buffer_batch_size 10 &

python ../main_mnist.py --main_optimizer 'sgd'  --lr_epochs 10 --cl_epochs 10 --main_online_lr 1e-2 --train-batch-size 128 --wd 1e-3 --cl_method 'sgd'  --cl_dataset permuted_mnist --num_tasks 5 --rank 0 --replay_buffer_batch_size 20 &

python ../main_mnist.py --main_optimizer 'sgd'  --lr_epochs 10 --cl_epochs 10 --main_online_lr 1e-2 --train-batch-size 128 --wd 1e-3 --cl_method 'sgd'  --cl_dataset permuted_mnist --num_tasks 5 --rank 0 --replay_buffer_batch_size 40 &

python ../main_mnist.py --main_optimizer 'sgd'  --lr_epochs 10 --cl_epochs 10 --main_online_lr 1e-2 --train-batch-size 128 --wd 1e-3 --cl_method 'sgd'  --cl_dataset permuted_mnist --num_tasks 5 --rank 0 --replay_buffer_batch_size 64 &
wait
