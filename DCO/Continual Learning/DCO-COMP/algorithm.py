import time
import torch
import trainer

import utils
from models import InvAuto, CommonModel
import itertools
import torch.nn.functional as F

def create_invauto(args, task_id, mod_main, mod_main_center, tr_loader, visdom_obj, prev_invauto=None):
    if args.main_para:
        # mod_ae = InvAuto(args, mod_main.module, args.ae_topk//(task_id+1), is_invauto=True, is_svd=args.is_svd, is_bias=False, prev_invauto=prev_invauto)
        mod_ae = InvAuto(args, mod_main.module, args.ae_topk, is_invauto=True, is_svd=args.is_svd, is_bias=False, prev_invauto=prev_invauto)
    else:
        # mod_ae = InvAuto(args, mod_main, args.ae_topk//(task_id+1), is_invauto=True, is_svd=args.is_svd, is_bias=False, prev_invauto=prev_invauto)
        mod_ae = InvAuto(args, mod_main, args.ae_topk, is_invauto=True, is_svd=args.is_svd, is_bias=False, prev_invauto=prev_invauto)
    mod_ae = mod_ae.to(args.device)   
    if args.ae_para:
        for name in args.ae_what:
            for i, E in enumerate(mod_ae.layers_E[name]):
                mod_ae.layers_E[name][i] = torch.nn.DataParallel(E)
            for i, D in enumerate(mod_ae.layers_D[name]):
                mod_ae.layers_D[name][i] = torch.nn.DataParallel(D)
    opt_ae = torch.optim.Adam(mod_ae.parameters(), lr = args.ae_offline_lr, weight_decay=args.ae_weight_decay)
    return mod_ae, opt_ae

def train_comauto(args, mod_main, mod_aes):
    mod_cm = CommonModel(mod_main.module, args.cm_topk).to(args.device)
    # if args.ae_para:
    #     for i, E in enumerate(mod_cm.layers_E):
    #         mod_cm.layers_E[i] = torch.nn.DataParallel(E)
    #     for i, D in enumerate(mod_cm.layers_D):
    #         mod_cm.layers_D[i] = torch.nn.DataParallel(D)
    mod_sgs = []
    for _ in mod_aes[1:]:
        mod_sg = CommonModel(mod_main.module, args.cm_topk // (len(mod_aes)-1)).to(args.device)
        # if args.ae_para:
        #     for i, E in enumerate(mod_sg.layers_E):
        #         mod_sg.layers_E[i] = torch.nn.DataParallel(E)
        #     for i, D in enumerate(mod_sg.layers_D):
        #         mod_sg.layers_D[i] = torch.nn.DataParallel(D)
        mod_sgs += [mod_sg]
    
    sg_all_datas_left = []
    sg_all_datas_righ = []
    for mod_ae in mod_aes[1:]:
        ae_params = []
        for p1, p2 in zip(mod_ae.parameters(), mod_aes[0].parameters()):
            ae_params += [torch.cat((p1.data.detach().view(p1.data.size(0), -1),
                                     p2.data.detach().view(p2.data.size(0), -1)),
                                     dim = 0)]
        sg_datas_left = []
        sg_datas_righ = []
        for i in range(len(ae_params) // 2):
            left   = ae_params[2* i+0].detach()
            righ   = ae_params[2* i+1].detach()
            sg_datas_left += [left]
            sg_datas_righ += [righ]
        sg_all_datas_left += [sg_datas_left]
        sg_all_datas_righ += [sg_datas_righ]
    

    opt_sgs = torch.optim.Adam(itertools.chain(*[mod_sg.parameters() for mod_sg in mod_sgs]), lr = 2* args.ae_offline_lr, weight_decay=args.ae_weight_decay)
    opt_cm = torch.optim.Adam(mod_cm.parameters(), lr = 2* args.ae_offline_lr, weight_decay=args.ae_weight_decay)
    for epoch in range(args.cm_epochs):
        opt_cm.zero_grad()
        opt_sgs.zero_grad()
        losses = []
        for ae in range(len(mod_aes[1:])):
            mid_out_sg = mid_out_cm = 0
            datas_left = sg_all_datas_left[ae]
            datas_righ = sg_all_datas_righ[ae]
            ae_params = []
            for p1, p2 in zip(mod_sgs[ae].parameters(), mod_cm.parameters()):
                ae_params += [torch.cat((p1.view(p1.size(0), -1),
                                        p2.view(p2.size(0), -1)),
                                        dim = 0)]
            comp_datas_left = []
            comp_datas_righ = []
            for i in range(len(ae_params) // 2):
                left   = ae_params[2* i+0]
                righ   = ae_params[2* i+1]
                comp_datas_left += [left]
                comp_datas_righ += [righ]

            total_loss = 0
            mbs = 20
            for i in range(mbs):
                bb, be = int(i* (datas_left[0].size(0) // mbs)), int((i+1)* (datas_left[0].size(0) // mbs))
                m1 = m2 = m3 = 0
                for l1, l2, r1, r2 in zip(datas_left, comp_datas_left, datas_righ, comp_datas_righ):
                    m1  = l1[bb:be].mm(l2.t()) # batch_size* topk
                    m2  = r1[bb:be].mm(r2.t()) # batch_size* topk
                    m3 += m1 * m2       # batch_size* topk
                    loss = 0
                for l1, l2, r1, r2 in zip(datas_left, comp_datas_left, datas_righ, comp_datas_righ):
                    t1 = (l2.mm(l2.t())).unsqueeze(0) * (r2.mm(r2.t())).unsqueeze(0) * m3.unsqueeze(2).bmm(m3.unsqueeze(1))
                    t2 = -2* (l1[bb:be].mm(l2.t()) * r1[bb:be].mm(r2.t()) * m3)
                    loss += (t1.sum() + t2.sum()) # / batch_size
                loss.backward()
                total_loss += loss.item()
            losses += [total_loss]
        print(epoch, losses, sum(losses))
        opt_cm.step()
        opt_sgs.step()
    return mod_cm, mod_sgs

def train_invauto(args, task_id, mod_main, mod_main_center, tr_loader, visdom_obj, prev_invauto=None):
    ''' the algorithm for trainining a linear autoencoder to find the top directions of optimizer's trajectory '''
    # 1. define a linear autoencoder
    if args.main_para:
        mod_ae = InvAuto(args, mod_main.module, args.ae_topk//(task_id+1), is_invauto=True, is_svd=args.is_svd, is_bias=False, prev_invauto=prev_invauto)
    else:
        mod_ae = InvAuto(args, mod_main, args.ae_topk//(task_id+1), is_invauto=True, is_svd=args.is_svd, is_bias=False, prev_invauto=prev_invauto)

    mod_ae = mod_ae.to(args.device)
    if args.ae_para:
        for name in args.ae_what:
            for i, E in enumerate(mod_ae.layers_E[name]):
                mod_ae.layers_E[name][i] = torch.nn.DataParallel(E)
            for i, D in enumerate(mod_ae.layers_D[name]):
                mod_ae.layers_D[name][i] = torch.nn.DataParallel(D)
    opt_main = torch.optim.SGD(mod_main.parameters(), lr = 1e-3)
    print('          ---> change the optimizer of main model to SGD')
    print('The size of model is [%.5f MB]'%(mod_ae.mlp_num_params* 32* 1.25e-7))

    # 2. train the linear autoencoder
    mod_main.train()
    if prev_invauto is not None:
        # opt_scalar = torch.optim.SGD([p for n, p in mod_ae.named_parameters() if 'scalars' in n], lr = 0.1, momentum=0.9, weight_decay=args.ae_weight_decay)
        # opt_ae = torch.optim.SGD([p for n, p in mod_ae.named_parameters() if not 'scalars' in n], lr = 0.1, momentum=0.9, weight_decay=args.ae_weight_decay)
        opt_scalar = torch.optim.Adam([p for n, p in mod_ae.named_parameters() if 'scalars' in n], lr = 0.2* args.ae_offline_lr, weight_decay=args.ae_weight_decay)
        opt_scalar_scheduler = torch.optim.lr_scheduler.StepLR(opt_scalar, step_size = 1, gamma=0.99)
    opt_ae = torch.optim.Adam([p for n, p in mod_ae.named_parameters() if not 'scalars' in n], lr = args.ae_offline_lr, weight_decay=args.ae_weight_decay)
    opt_ae_scheduler = torch.optim.lr_scheduler.StepLR(opt_ae, step_size = 1, gamma=0.99)
    mod_ae.mlp_counter = args.mlp_saved_iterations // 2 # (divided by \tau = 2)
    if prev_invauto is not None:
        total_epochs = args.ae_epochs
    else:
        total_epochs = args.ae_epochs

    # 2.1: save the intermediate gradients of main model (MAIN)
    for epoch in range(1, total_epochs+1):
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
        start_time = time.time()
        opt_ae.zero_grad()
        if prev_invauto is not None:
            opt_scalar.zero_grad()
        ae_loss1, ae_loss2 = mod_ae.ae_learn_offline() # --> we only use ae_loss2
        ae_loss2.backward()
        if prev_invauto is not None:
            for s in mod_ae.scalars:
                print(s.max(), s.min())  
            opt_scalar.step()
            opt_scalar_scheduler.step()
        opt_ae.step()
        opt_ae_scheduler.step()
        visdom_obj.line([ae_loss1.item()], [task_id* args.ae_epochs + epoch-1], update='append', opts={'title':'[Autoencoder] L2 loss'}, win='offline_ae_l2_loss', env='gpu:%d'%args.rank)
        print('\t  ---> Each epoch takes %.2fs'%(time.time()-start_time))

    return mod_ae, opt_ae

def evaluate_invauto(args, task_id, mod_main, mod_main_center, mod_ae, tr_loader, visdom_obj):
    sum_mid_out = 0
    opt_main = torch.optim.SGD(mod_main.parameters(), lr = 1e-3)
    mod_main.train()
    for epoch in range(3):
        for data, target in tr_loader:
            opt_main.zero_grad()
            main_loss = trainer.train(args, mod_main, opt_main, data, target)
            main_loss.backward()
            opt_main.step()
            ae_mid_out   = mod_ae.ae_grad_encode(mod_main, mod_main_center).detach()
            sum_mid_out += ae_mid_out** 2
        mod_main.module.pull2point(mod_main_center, pull_strength=0.1) # pull to the center variavle
    return sum_mid_out