# options.py: parse options
import torch
import argparse

# parser
parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser(description='Continual learning arguments')

# arguments
parser.add_argument('--cl_dataset'      , type=str, default='split_cifar100', metavar='CLD')
parser.add_argument('--cl_method'       , type=str, default='dco', metavar='CLM')
parser.add_argument('--num_tasks'       , type=int, default=5 , metavar='N')

parser.add_argument('--lr_epochs'       , type=int, default=30 , metavar='N')
parser.add_argument('--post-batch-size' , type=int, default=128, metavar='N')
parser.add_argument('--train-batch-size', type=int, default=128, metavar='N')
parser.add_argument('--test-batch-size' , type=int, default=1024, metavar='N')
parser.add_argument('--main_para', action='store_true', default=True)
parser.add_argument('--ae_para', action='store_true', default=True)
parser.add_argument('--main_lr_gamma'  , type=float, default= 1.0, metavar='LD')

parser.add_argument('--ae_what', type=str, default='M')
parser.add_argument('--ae_topk', type=int, default=1000, metavar='K')
parser.add_argument('--mlp_saved_epochs', type=int, default=10, metavar='S')

parser.add_argument('--ae_epochs', type=int, default=200, metavar='N')
parser.add_argument('--in_epochs', type=int, default= 1, metavar='N')
parser.add_argument('--cl_epochs', type=int, default=10, metavar='N')
parser.add_argument('--prox_epochs', type=int, default=10, metavar='N')

parser.add_argument('--is_mlps_saved', action='store_true', default=False) # TODO: FIXME set to False
parser.add_argument('--mlp_saved_iterations', type=int, default=512, metavar='N')
parser.add_argument('--ae_saved_epochs', type=int, default=100, metavar='N')

parser.add_argument('--main_online_lr' , type=float, default=1e-4, metavar='LR')
# ewc
parser.add_argument('--ewc_lam'      , type=float, default= 100, metavar='CL')
# si
parser.add_argument('--si_lam'      , type=float, default= 100, metavar='CL')
parser.add_argument('--si_epsilon'      , type=float, default= 0.1, metavar='CL')
# fisher
parser.add_argument('--fisher_ema_decay', type=float, default= 0.9, metavar='CL')
parser.add_argument('--fisher_update_after', type=int, default= 100, metavar='CL')
# rwalk
parser.add_argument('--rwalk_epsilon'      , type=float, default= 0.1, metavar='CL')
parser.add_argument('--rwalk_lam'      , type=float, default= 1, metavar='CL')
# agem
parser.add_argument('--episodic_batch_size', type=int, default= 256, metavar='CL')
parser.add_argument('--episodic_mem_size'  , type=int, default= 256, metavar='CL')
# autoencoder
parser.add_argument('--ae_online_lr'   , type=float, default=1e-2, metavar='LR')
parser.add_argument('--ae_offline_lr'  , type=float, default=1e-2, metavar='LR')
parser.add_argument('--ae_online_ps'   , type=float, default= 0.0, metavar='PS')
parser.add_argument('--ae_offline_ps'  , type=float, default= 0.0, metavar='PS')
parser.add_argument('--ae_re_lam'      , type=float, default= 100, metavar='RE')
parser.add_argument('--ae_cl_lam'      , type=float, default= 100, metavar='CL')
parser.add_argument('--ae_weight_decay', type=float, default= 0.0, metavar='WD')
parser.add_argument('--ae_lr_gamma'    , type=float, default= 1.0, metavar='LD')
parser.add_argument('--ae_mean', action='store_true', default=False)
parser.add_argument('--ae_grad_norm'      , type=float, default= 20, metavar='CL')
parser.add_argument('--l2_regularization', type=float, default= 0.0, metavar='L2RE')

parser.add_argument('--seed', type=int, default=1, metavar='S')
parser.add_argument('--rank', type=int, default=0, metavar='S')

parser.add_argument('--is_grad', action='store_true', default=False)
parser.add_argument('--is_svd',  action='store_true', default=False)
parser.add_argument('--sample_posterior',  action='store_true', default=False)

# push inside cone
parser.add_argument('--push_cone_l2'   , type=float, default=0.0, metavar='Cone_L2')
parser.add_argument('--cone_batch_size', type=int, default= 128, metavar='CBS')
parser.add_argument('--main_optimizer', type=str, default='adam')
parser.add_argument('--wd', type=float, default=0.0, metavar='WD')