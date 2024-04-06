import time
import torch
import trainer

import utils
from models import InvAuto
import itertools

def train_invauto(args, task_id, mod_main, mod_main_center, tr_loader, visdom_obj):
    ''' the algorithm for trainining a linear autoencoder to find the top directions of optimizer's trajectory '''
    # 1. define a linear autoencoder
    if args.main_para:
        mod_ae = InvAuto(args, mod_main.module, args.ae_topk, is_invauto=True, is_svd=args.is_svd, is_bias=False)
    else:
        mod_ae = InvAuto(args, mod_main, args.ae_topk, is_invauto=True, is_svd=args.is_svd, is_bias=False)
    mod_ae = mod_ae.to(args.device)
    if args.ae_para:
        for name in args.ae_what:
            for i, E in enumerate(mod_ae.layers_E[name]):
                mod_ae.layers_E[name][i] = torch.nn.DataParallel(E)
            for i, D in enumerate(mod_ae.layers_D[name]):
                mod_ae.layers_D[name][i] = torch.nn.DataParallel(D)

    # opt_main = torch.optim.SGD(mod_main.parameters(), lr = args.main_online_lr)
    opt_main = torch.optim.SGD(mod_main.parameters(), lr = 1e-3)
    print('          ---> change the optimizer of main model to SGD')
    print('The size of model is [%.5f MB]'%(mod_ae.mlp_num_params* 32* 1.25e-7))
    # 2. train the linear autoencoder
    mod_main.train()
    opt_ae = torch.optim.Adam(mod_ae.parameters(), lr = args.ae_offline_lr, weight_decay=args.ae_weight_decay)
    opt_ae_scheduler = torch.optim.lr_scheduler.StepLR(opt_ae, step_size = 100, gamma=args.ae_lr_gamma)
    mod_ae.mlp_counter = args.mlp_saved_iterations // 2 # (divided by \tau = 2)
    for epoch in range(1, args.ae_epochs+1):
        start_time = time.time()
        # 2.1: save the intermediate gradients of main model (MAIN)
        for data_idx in itertools.count(1):
            try:
                data, target = next(tr_loader_t1_iter)
            except:
                tr_loader_t1_iter = tr_loader.__iter__()
                data, target = next(tr_loader_t1_iter)
            main_loss = trainer.train(args, mod_main, opt_main, data, target, is_grad_acc=True)
            main_loss.backward()                
            if data_idx % 2 == 1: # step 1: sample gradients
                mod_ae.ae_snap_mlp(1)
            if data_idx % 2 == 0: # step 2: sample gradients and update the model (\tau = 2)
                mod_ae.ae_snap_mlp(2)
                opt_main.step()
                opt_main.zero_grad()
            if data_idx >= args.mlp_saved_iterations:
                mod_ae.ae_save_mlps(postfix='')
                mod_main.module.pull2point(mod_main_center, pull_strength=0.1) # pull to the center variavle
                break

        # 2.2: train on the intermediate gradients of main model (AE)
        opt_ae.zero_grad()
        ae_loss1, ae_loss2 = mod_ae.ae_learn_offline() # --> we only use ae_loss2
        ae_loss2.backward()
        opt_ae.step()
        opt_ae_scheduler.step()
        visdom_obj.line([ae_loss2.item()], [task_id* args.ae_epochs + epoch-1], update='append', opts={'title':'[Autoencoder] L2 loss'}, win='offline_ae_l2_loss', env='gpu:%d'%args.rank)
        print('\t  ---> Each epoch takes %.2fs'%(time.time()-start_time))

    return mod_ae, opt_ae