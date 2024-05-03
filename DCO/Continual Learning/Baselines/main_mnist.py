import copy
import time
import sys
import torch
import torch.nn.functional as F
import plotter
import utils
import trainer
from models import MLP, Conv
from options import parser
from data import get_dataset, get_data_loader, get_label_data_loader, map_dataset
from data import DATASET_CONFIGS, MODEL_CONFIGS
import train_utils
import path_utils
import numpy as np
import dill as pickle
from torch.utils.data import Subset
import copy
def run_main(args, experiment_name):
    global_time = time.time()
    checkpoint = False
    if args.checkpoint:
        checkpoint = True
        print(f"Loading the checkpoint at {args.checkpoint}")
        with open(args.checkpoint, 'rb') as file:
            save_object = pickle.load(file)
        args = save_object["args"]
        local_test_loss = save_object["local_test_loss"]
    else:
        print(f"Arguments: {args}")
        save_object = {"errors": [], "n_tasks": args.num_tasks, "test_batch_loss": [], "test_batch_error": [], "args": args, "expansion_epochs": []}
        local_test_loss = [[] for _ in range(args.num_tasks)]
    # if checkpoint:
    #     rbcl = save_object["rbcl"]
    if args.replay_buffer_batch_size:
        rbcl = train_utils.ReplayBufferCL(n_tasks=args.num_tasks, max_size=3 * args.replay_buffer_batch_size,
                                          batch_size=args.replay_buffer_batch_size)
    else:
        rbcl = None

    # // 1.2 Main model //
    """
        (1) by default the biases of all architectures are turned off.
        (2) be careful to switch between about model.train() and model.eval() in case of dropout layers
    """
    cl_error_threshold = args.cl_error_threshold
    dataset_name = args.cl_dataset
    if 'mnist' in args.cl_dataset:
        model_conf = MODEL_CONFIGS[dataset_name]
        input_size, output_size = DATASET_CONFIGS[dataset_name]['size'] ** 2, DATASET_CONFIGS[dataset_name]['classes']
        mod_main = MLP(args, input_size, output_size, hidden_size=model_conf['s_layer'],
                       hidden_layer_num=model_conf['n_layer'])

        # # train_utils.handle_new_hidden_layer_logic(mod_main, args, model_conf, 0, 1)
        # mod_main.add_hidden_layerV3(args.added_layer_conf[1], count=args.added_layer_conf[0],
        #                              same=args.added_layer_conf[1] == model_conf['s_layer'], task=1)
        # print("Before")
        # mod_main.print_grad_req_for_all_params()
        # print(mod_main.tasks)
        # print(mod_main.tasks_output)
        # mod_main.freeze(1)
        # print("After freeze")
        # mod_main.print_grad_req_for_all_params()
        # mod_main.unfreeze(1)
        # print("After unfreeze")
        # mod_main.print_grad_req_for_all_params()
        # return
    elif 'cifar' in args.cl_dataset:
        mod_main = Conv(args)
    else:
        raise ValueError('No matched dataset')
    mod_main = mod_main.to(args.device)
    if args.main_para and args.cl_method == 'dco':
        mod_main = torch.nn.DataParallel(mod_main)
    if args.cl_dataset == 'split_cifar100':
        first_task_lr = 0.01
        log_interval = 10
    else:
        first_task_lr = args.main_online_lr
        log_interval = 1
    if args.main_optimizer == 'sgd':
        opt_main = torch.optim.SGD(mod_main.parameters(), lr=first_task_lr, momentum=0.9, weight_decay=args.wd)
    else:
        opt_main = torch.optim.Adam(mod_main.parameters(), lr=first_task_lr, weight_decay=args.wd)
    opt_main_scheduler = torch.optim.lr_scheduler.StepLR(opt_main, step_size=10, gamma=args.main_lr_gamma)

    # // 1.3 visdom: https://github.com/facebookresearch/visdom //
    """ a open-source visualization tool from facebook (tested on 0.1.8.8 version) """
    # try:
    #     visdom_obj = utils.get_visdom(args, experiment_name)
    # except Exception as e:
    #     print(e)
    #     print('[Visdom] ===> De-activated')
    #
    # // 1.4 Define task datasets and their dataloders //
    """
        (1) structure of the task loader is: [0, dataset_1, dataset_2, ..., dataset_n]
        (2) for permuted_mnist: the permutation pattern is controlled by np.random.seed(m_task) in `data.py`
        (3) for split mnist and split cifar-100: the sequence of tasks are defined by a sequence of labels
    """
    tr_loaders, te_loaders = [0], [0]
    tr_loaders_full, te_loaders_full = [0], [0]


    for m_task in range(1, args.num_tasks + 1):
        if dataset_name == 'permuted_mnist':
            tr_dataset = get_dataset(dataset_name, m_task, True)
            te_dataset = get_dataset(dataset_name, m_task, False)

            train_indices_subset, test_indices_subset = torch.randperm(len(tr_dataset))[:100], torch.randperm(len(te_dataset))[:100]
            tr_dataset_subset = Subset(tr_dataset, train_indices_subset)
            te_dataset_subset = Subset(te_dataset, test_indices_subset)

            train_indices, test_indices = torch.randperm(len(tr_dataset))[:len(tr_dataset) // 2], torch.randperm(len(te_dataset))[:len(te_dataset) // 2]

            tr_dataset = Subset(tr_dataset, train_indices)
            te_dataset = Subset(te_dataset, test_indices)

            tr_loaders += [get_data_loader(tr_dataset, args.train_batch_size,
                                           cuda=('cuda' in args.device))]
            te_loaders += [get_data_loader(te_dataset, args.train_batch_size,
                                           cuda=('cuda' in args.device))]

            tr_loaders_full += [get_data_loader(tr_dataset_subset, len(tr_dataset_subset),
                                           cuda=('cuda' in args.device))]
            te_loaders_full += [get_data_loader(te_dataset_subset, len(te_dataset_subset),
                                           cuda=('cuda' in args.device))]

        elif dataset_name == 'split_mnist' or dataset_name == 'split_cifar10':
            tr_dataset = get_dataset(dataset_name, m_task, True)
            te_dataset = get_dataset(dataset_name, m_task, False)

            m_label = m_task - 1

            # train_indices_subset, test_indices_subset = torch.randperm(len(tr_dataset))[:100], torch.randperm(len(te_dataset))[:100]
            # tr_dataset_subset = Subset(tr_dataset, train_indices_subset)
            # te_dataset_subset = Subset(te_dataset, test_indices_subset)
            #
            # train_indices, test_indices = torch.randperm(len(tr_dataset))[:len(tr_dataset) // 2], torch.randperm(len(te_dataset))[:len(te_dataset) // 2]
            #
            # tr_dataset = Subset(tr_dataset, train_indices)
            # te_dataset = Subset(te_dataset, test_indices)
            # print(tr_dataset)
            # return

            tr_loaders += [get_label_data_loader(tr_dataset, args.train_batch_size,
                                                 cuda=('cuda' in args.device),
                                                 labels=[m_label * 2, m_label * 2 + 1])]
            te_loaders += [get_label_data_loader(te_dataset, args.train_batch_size,
                                                 cuda=('cuda' in args.device),
                                                 labels=[m_label * 2, m_label * 2 + 1])]


            tr_loaders_full += [get_label_data_loader(tr_dataset, args.train_batch_size,
                                                 cuda=('cuda' in args.device),
                                                 labels=[m_label * 2, m_label * 2 + 1])]
            te_loaders_full += [get_label_data_loader(te_dataset, args.train_batch_size,
                                                 cuda=('cuda' in args.device),
                                                 labels=[m_label * 2, m_label * 2 + 1])]
        elif dataset_name == 'split_cifar100':
            m_label = m_task - 1
            tr_loaders += [get_label_data_loader(get_dataset(dataset_name, m_task, True), args.train_batch_size,
                                                 cuda=('cuda' in args.device),
                                                 labels=[l for l in range(m_label * 10, m_label * 10 + 10)])]
            te_loaders += [get_label_data_loader(get_dataset(dataset_name, m_task, False), args.train_batch_size,
                                                 cuda=('cuda' in args.device),
                                                 labels=[l for l in range(m_label * 10, m_label * 10 + 10)])]
    print('[MAIN/CL] ===> Training main model for %d epochs' % args.lr_epochs)
    print('          ---> The number of training data points for each epoch is %d' % len(tr_loaders[1].dataset))
    print('          ---> The number of  testing data points for each epoch is %d' % len(te_loaders[1].dataset))
    assert not (len(tr_loaders[1].dataset) == len(
        te_loaders[1].dataset))  # just in case the trainining dataset and testing dataset are messed up

    global_epoch = 0
    added_layers_count = 0
    max_allowed_added_layers = args.max_allowed_added_layers
    save_object["batches_in_epoch"] = len(tr_loaders[1])

    # // 2.1 Preparation before the 1st task //
    """ Algorithms:
            (1) SI (sometimes referred to as PI): the cores of SI are Eq.(4) and Eq.(5)
            (2) RWALK: Similar to SI. More details should be referred to agem/model.py (lines 1280~1311)
        The above two methods (I call them path integral methods) need preparation before the 1st task
    """
    if args.cl_method == 'si':
        small_omega = 0
        big_omegas = 0
        param_main_start = utils.ravel_model_params(mod_main, False, 'cpu')
    elif args.cl_method == 'rwalk':
        old_param = utils.ravel_model_params(mod_main, False, 'cpu')
        running_fisher = utils.ravel_model_params(mod_main, False, 'cpu')
        running_fisher.zero_()
        small_omega = 0
        big_omegas = 0
        tmp_fisher_vars = 0
    else:
        pass
    starting_point = utils.ravel_model_params(mod_main, False, 'cpu')

    # // 2.2 Train for the 1st task //
    """
        (1) cur_itertation: this counter is only used for RWALK to upadte `running_fisher` periodically
        (2) small_omega: track the contribution to the change of loss function of each individual parameter
        (3) big_omega: track the distance between current parameters and previous parameters
    """
    # worse_perfomance = train_utils.each_batch_test(args, save_object, mod_main, te_loaders_full, 0,
    #                                                local_test_loss)

    print(f"Training task 1")

    cur_iteration = 0
    for epoch in range(0, args.lr_epochs):
        epoch_time = train_utils.calc_time()
        random_replay_batch_ids = train_utils.get_random_replay_batch_ids(1, len(tr_loaders[1]) - 1, args)
        for batch_idx, (data, target) in enumerate(tr_loaders[1], 1):
            if batch_idx in random_replay_batch_ids:
                train_utils.add_to_replay_buffer(rbcl, 1, data, target, args)

            cur_iteration += 1
            if args.cl_method == 'si' or args.cl_method == 'rwalk':
                param1 = utils.ravel_model_params(mod_main, False, 'cpu')
                main_loss = trainer.train(args, mod_main, opt_main, data, target, task=m_task - 1)
                main_loss.backward()
                grad = utils.ravel_model_params(mod_main, True, 'cpu')  # 'plain' gradients without regularization
                opt_main.step()
                param2 = utils.ravel_model_params(mod_main, False, 'cpu')
                small_omega += -grad * (param2 - param1)
                if args.cl_method == 'rwalk':
                    """ this part has a slightly different update rule for running fisher and its temporary variables: 
                        (1) updates of small_omega and big_omega: agem/fc_permute_mnist.py (lines 319 ~ 334) --> important
                        (2) update of running_fisher and temp_fisher_vars : agem/model (lines 1093 and 1094) and agem/model.py (lines 1087)
                        (3) fisher_ema_decay should be 0.9 from agem codes of both permuted mnist and split cifar-100 
                        CAERFUL don't forget to check if it is '+=' for big_omega and small_omega """
                    tmp_fisher_vars += grad ** 2
                    if cur_iteration == 1:  # initilaization for running fisher
                        running_fisher = grad ** 2
                    if cur_iteration % args.fisher_update_after == 0:
                        # 1. update big omega
                        cur_param = utils.ravel_model_params(mod_main, False, 'cpu')
                        delta = running_fisher * ((cur_param - old_param) ** 2) + args.rwalk_epsilon
                        big_omegas += torch.max(small_omega / delta, torch.zeros_like(small_omega)).to(args.device)
                        # 2. update running fisher
                        running_fisher = (1 - args.fisher_ema_decay) * running_fisher + (
                                1.0 / args.fisher_update_after) * args.fisher_ema_decay * tmp_fisher_vars
                        # 3. assign current parameters as old parameters
                        old_param = cur_param
                        # 4. reset small omega to zero
                        small_omega = 0
            else:
                main_loss = trainer.train(args, mod_main, opt_main, data, target, task=0)
                main_loss.backward()
                opt_main.step()

            worse_perfomance = train_utils.each_batch_test(args, save_object, mod_main, te_loaders_full, 0,
                                                           local_test_loss)

        opt_main_scheduler.step()
        if epoch % log_interval == 0:
            errors = []
            for i in range(1, args.num_tasks + 1):
                cur_error, test_loss = trainer.test(args, mod_main, te_loaders[i], i, global_epoch + epoch, task=i - 1)
                errors += [cur_error]
                # # visdom_obj.line([cur_error], [global_epoch + epoch], update='append',
                #                 opts={'title': '%d-Task Error' % i}, win='cur_error_%d' % i, name='T',
                #                 env=f'{experiment_name}')
                # Checking only task 1
                # if epoch % args.lr_epochs == 0 and i == 1 and cur_error > cl_error_threshold:
                #     print("Current error is bigger than threshold. Adding the new layer.")
                #     # adding_new_hidden_layer = True
                #     mod_main, added_layers_count = handle_new_hidden_layer_logic(mod_main, args, result_list, model_conf, added_layers_count)
                #     # break
                # elif epoch % args.lr_epochs == 0 and i == 1:
                #     print(f"Success. Task 1 Error: {cur_error:.2f}. No need for adding layer")

            current_point = utils.ravel_model_params(mod_main, False, 'cpu')
            l2_norm = (current_point - starting_point).norm().item()
            # visdom_obj.line([l2_norm], [global_epoch + epoch], update='append', opts={'title': 'L2 Norm'},
            #                 win='l2_norm', name='T', env=f'{experiment_name}')
            # visdom_obj.line([sum(errors) / args.num_tasks], [global_epoch + epoch], update='append',
            #                 opts={'title': 'Average Error'}, win='avg_error', name='T', env=f'{experiment_name}')
            save_object["errors"] += [errors]
        epoch_time = train_utils.calc_time(epoch_time, "Epoch")


    # 3.1 Preparation before the 2nd and consequent tasks
    """
        SGD       : no prepration needed
        EWC       : need to save digonal Fisher matrix in F. The collection of {F} --> Fs
        SI        : no prepration needed
        RWALK     : copy current parameter values to old_param
        GEM(A-GEM): prepare to save examples in gem_dataset
        DCO       : need to save the optimial parameters of the model and train a linear autoencoer for every task
    """
    if args.cl_method == 'sgd':
        pass
    elif args.cl_method == 'ewc':
        mod_main_centers = []
        Fs = []
    elif args.cl_method == 'si':
        mod_main_centers = []
    elif args.cl_method == 'rwalk':
        mod_main_centers = []
        old_param = utils.ravel_model_params(mod_main, False, 'cpu')
    elif 'gem' in args.cl_method:
        gem_dataset = []
    elif args.cl_method == 'dco':
        mod_aes, opt_aes = [0], [0]
        mod_main_centers = [0]

    if args.main_optimizer == 'sgd':
        opt_main = torch.optim.SGD(mod_main.parameters(), lr=args.main_online_lr, momentum=0.9,
                                   weight_decay=args.wd)
    else:
        opt_main = torch.optim.Adam(mod_main.parameters(), lr=args.main_online_lr, weight_decay=args.wd)

    # 3.2 Train for the 2nd and consequent tasks
    for m_task in range(2, args.num_tasks + 1):
        print(f"Training task {m_task}")
        if args.cl_method == 'sgd':
            pass
        elif args.cl_method == 'ewc':
            mod_main_centers, Fs = upgrade_mod_main_ewc(mod_main_centers, Fs, mod_main, dataset_name, m_task, args,
                                                        opt_main)
        elif args.cl_method == 'si':
            """
                (1) delta     : track the distance between current parameters and previous parameters
                (2) big_omegas: acuumulate `small_omega/delta` through training
                We reset the opitmizer of SI at the end of each task (see left-top paragraph on page 6 of SI paper) 
            """
            if args.main_optimizer == 'sgd':
                opt_main = torch.optim.SGD(mod_main.parameters(), lr=args.main_online_lr, momentum=0.9,
                                           weight_decay=args.wd)
            else:
                opt_main = torch.optim.Adam(mod_main.parameters(), lr=args.main_online_lr, weight_decay=args.wd)
            delta = (utils.ravel_model_params(mod_main, False, 'cpu') - param_main_start) ** 2 + args.si_epsilon
            big_omegas += torch.max(small_omega / delta, torch.zeros_like(small_omega)).to(
                args.device)  # check if I need to devide delta by 2
            small_omega = 0
            param_main_start = utils.ravel_model_params(mod_main, False, 'cpu')
            mod_main_centers += [utils.ravel_model_params(mod_main, False, args.device)]
        elif args.cl_method == 'rwalk':
            """
                (1) normalized fisher: agem/model.py (lines 1101 ~ 1111)
                (2) normlaized score : check this on page 8 of RWALK paper;
                                       agem/model.py (lines 1049 ~ 1065) does not seem to be consitent with the description in the paper
                (3) reset small_omega, big_omega and temp_fisher: agem/model.py (lines 1280 ~ 1311)
                DON'T mess up with normalized and unnormalized scores and fishers
            """
            # normalized score
            if m_task == 2:
                score_vars = big_omegas
            else:
                score_vars = (score_vars + big_omegas) / 2
            max_score = score_vars.max()
            min_score = score_vars.min()
            normalize_scores = (score_vars - min_score) / (max_score - min_score + args.rwalk_epsilon)
            normalize_scores = normalize_scores.to(args.device)

            # normalized fisher
            fisher_at_minima = running_fisher
            max_fisher = fisher_at_minima.max()
            min_fisher = fisher_at_minima.min()
            normalize_fisher_at_minima = (fisher_at_minima - min_fisher) / (
                    max_fisher - min_fisher + args.rwalk_epsilon)
            normalize_fisher_at_minima = normalize_fisher_at_minima.to(args.device)

            # reset
            small_omega = 0
            big_omegas = 0
            tmp_fisher_vars = 0
            mod_main_centers += [utils.ravel_model_params(mod_main, False, args.device)]

        elif 'gem' in args.cl_method:
            """
                gem_dataset:    all the sample from previous tasks are saved in this list
                                the episodic memory per task for a-gem is set to 256 and and 512 for MNIST and CIFAR-100 respectively
                                (by the way, the description of the amount of episodic memory is partially missing in A-GEM paper, page 7, line 4)
                feed_batch_size: 1 for GEM (one data point after another)
                                 256 and 1300 on MNIST and CIFAR-100 respectively for A-GEM
            """
            # print('== GEM Method ==')
            episodic_mem_size, episodic_batch_size = args.episodic_mem_size, args.episodic_batch_size
            if dataset_name == 'permuted_mnist':
                agem_loader = get_data_loader(get_dataset(dataset_name, m_task - 1, True), batch_size=1, cuda=False)
            elif dataset_name == 'split_mnist' or dataset_name == 'split_cifar10':
                m_label = m_task - 2
                agem_loader = get_label_data_loader(get_dataset(dataset_name, m_task - 1, True), 1, cuda=False,
                                                    labels=[m_label * 2, m_label * 2 + 1])
            elif dataset_name == 'split_cifar100':
                m_label = m_task - 2
                agem_loader = get_label_data_loader(get_dataset(dataset_name, m_task - 1, True), 1, cuda=False,
                                                    labels=[l for l in range(m_label * 10, m_label * 10 + 10)])
            for num, (data, target) in enumerate(agem_loader, 1):
                gem_dataset += [(data.view(data.size(1), data.size(2), data.size(3)), target)]
                if num >= episodic_mem_size:
                    break  # stop when the episodic memory for current task is filled
            feed_batch_size = 1 if args.cl_method == 'gem' else min(args.episodic_batch_size,
                                                                    episodic_mem_size * (m_task - 1))
            agem_loader = get_data_loader(map_dataset(gem_dataset), batch_size=feed_batch_size, cuda=False)
            # if feed_batch_size == episodic_mem_size:
            #     agem_iter = agem_loader.__iter__()
            #     agem_data, agem_target = next(agem_iter)

        elif args.cl_method == 'dco':
            """
                Notice that we train for extra `args.prox_epochs` epochs for our method
            """
            mod_main_centers, mod_aes, opt_aes, cl_opt_main = train_utils.upgrade_dco(mod_main_centers, mod_aes, opt_aes,
                                                                          mod_main, tr_loaders, args, m_task,
                                                                          opt_main, None)
        else:
            raise ValueError('No named method')

        # training
        epoch_time = train_utils.calc_time()
        cur_iteration = 0
        for cl_epoch in range(args.cl_epochs):
            random_replay_batch_ids = train_utils.get_random_replay_batch_ids(1, len(tr_loaders[1]) - 5, args)
            batch_idx = 0
            data, target = next(iter(tr_loaders[m_task]))
            while batch_idx < len(tr_loaders[m_task]) - 1:
                # path = f"{experiment_name}/t.pt"
                # torch.save(mod_main.state_dict(), path)
                # mod_main_ref = copy.deepcopy(mod_main.state_dict())
                # opt_main_copy = copy.deepcopy(opt_main)
                # mod_main_copy = copy.deepcopy(mod_main)
                # batch_t = train_utils.calc_time()
                # Adding replay buffer
                # t = train_utils.calc_time()
                if batch_idx in random_replay_batch_ids:
                    train_utils.add_to_replay_buffer(rbcl, m_task, data, target, args)
                samples = train_utils.get_samples(rbcl, m_task, args)
                # t = train_utils.calc_time(t, "Get_samples")

                cur_iteration += 1
                if args.cl_method == 'sgd':
                    # print("SGD task ", m_task - 1)
                    train_utils.train_sgd_cl(args, mod_main, opt_main, data, target, task=m_task - 1)
                    train_utils.train_sgd_cl_replay(samples, args, mod_main, opt_main)
                    # return
                    # t = train_utils.calc_time(t, "train_sgd_cl")
                elif args.cl_method == 'ewc':
                    train_utils.train_ewc_cl(args, mod_main, opt_main, data, target, mod_main_centers, Fs,
                                             task=m_task - 1)
                    train_utils.train_ewc_cl_replay(samples, args, mod_main, opt_main, mod_main_centers, Fs)

                elif args.cl_method == 'si':
                    """ SI algorithm adds per-parameter regularization loss to the total loss """
                    param1 = utils.ravel_model_params(mod_main, False, 'cpu')
                    main_loss = trainer.train(args, mod_main, opt_main, data, target, task=m_task - 1)
                    main_loss.backward()
                    grad = utils.ravel_model_params(mod_main, True, 'cpu')
                    loss, cur_p = 0, 0
                    for param in mod_main.parameters():
                        size = param.numel()
                        cur_loss = F.mse_loss(param, mod_main_centers[-1][cur_p: cur_p + size].view_as(param),
                                              reduction='none')
                        loss += (big_omegas[cur_p: cur_p + size].view_as(param) * cur_loss).sum() * args.si_lam
                        cur_p += size
                    loss.backward()
                    opt_main.step()
                    param2 = utils.ravel_model_params(mod_main, False, 'cpu')
                    small_omega += -grad * (param2 - param1)
                elif args.cl_method == 'rwalk':
                    """ Double-check:
                            (1) cur_iteration: make sure it is added by 1 for each iteration and reset to 0 at the beginning of task
                            (2) updates: the updates of running fisher etc. should be the same as before. Don't forget to check '+=' for big_omegas
                    """
                    param1 = utils.ravel_model_params(mod_main, False, 'cpu')
                    main_loss = trainer.train(args, mod_main, opt_main, data, target, task=m_task - 1)
                    main_loss.backward()
                    grad = utils.ravel_model_params(mod_main, True, 'cpu')
                    loss, cur_p = 0, 0
                    for param in mod_main.parameters():
                        size = param.numel()
                        reg = (normalize_scores + normalize_fisher_at_minima)[cur_p: cur_p + size].view_as(param)
                        cur_loss = F.mse_loss(param, mod_main_centers[-1][cur_p: cur_p + size].view_as(param),
                                              reduction='none')
                        loss += (reg * cur_loss).sum() * args.rwalk_lam
                        cur_p += size
                    loss.backward()
                    opt_main.step()
                    param2 = utils.ravel_model_params(mod_main, False, 'cpu')
                    small_omega += -grad * (param2 - param1)
                    tmp_fisher_vars += grad ** 2
                    if cur_iteration == 1:  # initilaization for running fisher
                        running_fisher = grad ** 2
                    if cur_iteration % args.fisher_update_after == 0:
                        # 1. update big omega
                        cur_param = utils.ravel_model_params(mod_main, False, 'cpu')
                        delta = running_fisher * ((cur_param - old_param) ** 2) + args.rwalk_epsilon
                        big_omegas += torch.max(small_omega / delta, torch.zeros_like(small_omega)).to(args.device)
                        # 2. update running fisher
                        running_fisher = (1 - args.fisher_ema_decay) * running_fisher + (
                                1.0 / args.fisher_update_after) * args.fisher_ema_decay * tmp_fisher_vars
                        # 3. assign current parameters as old parameters
                        old_param = cur_param
                        # 4. reset small omega to zero
                        small_omega = 0
                elif 'gem' in args.cl_method:
                    """ Double-check:
                            (1) pay attention to the gradients we are manipulating. i.e., Don't mess up the original gradient with the projected gradient.
                            (2) remember to call mod_main.zero_grad() otherwise the gradients are going to be accumulated.
                            (3) check trainer.train() and trainer.test() for GEM and A-GEM for multi-head setting 
                                and make sure it has the propoer masked output for such replay methods
                        Updates:
                            (1) Update of A-GEM: see A-GEM paper Equation (11)
                            (2) Update of GEM  : it solves a quadratic programming problem with `quadprog`, a cpu-based pyhton library, for every iteration.
                                                 Thus GEM may be intractable for deep neural networks.
                    """
                    main_loss = trainer.train(args, mod_main, opt_main, data, target, task=m_task - 1)
                    main_loss.backward()
                    m_grad = utils.ravel_model_params(mod_main, True, args.device)
                    mod_main.zero_grad()
                    if args.cl_method == 'agem':
                        agem_iter = agem_loader.__iter__()
                        agem_data, agem_target = next(agem_iter)
                        agem_target = agem_target.view(-1)
                        main_loss = trainer.train(args, mod_main, opt_main, agem_data, agem_target, task=m_task - 1)
                        main_loss.backward()
                        gem_grad = utils.ravel_model_params(mod_main, True, args.device)
                        mod_main.zero_grad()
                        dot_product = torch.sum(gem_grad * m_grad)
                        if dot_product < 0:
                            m_grad = m_grad - gem_grad * dot_product / (gem_grad.norm() ** 2)
                    else:
                        gem_grad = []
                        t1 = time.time()
                        for gem_data, gem_target in agem_loader:
                            gem_target = gem_target.view(-1)
                            main_loss = trainer.train(args, mod_main, opt_main, gem_data, gem_target,
                                                      task=m_task - 1)
                            main_loss.backward()
                            gem_grad += [utils.ravel_model_params(mod_main, True, args.device)]
                            mod_main.zero_grad()
                        gem_grad = torch.stack(gem_grad)
                        t2 = time.time()
                        m_grad = utils.project2cone2(m_grad,
                                                     gem_grad)
                        t3 = time.time()
                        print(t2 - t1, t3 - t2)
                    utils.assign_model_params(m_grad, mod_main, True)
                    opt_main.step()
                elif args.cl_method == 'dco':
                    ae_loss, grad_norm = train_utils.train_dco_cl(args, mod_main, opt_main, data, target, m_task,
                                                                  mod_main_centers, cl_opt_main, mod_aes, opt_aes,
                                                                  task=m_task - 1)
                    ae_loss, grad_norm = train_utils.train_dco_cl_replay(samples, args, mod_main, opt_main, data,
                                                                         target, m_task, mod_main_centers,
                                                                         cl_opt_main, mod_aes, opt_aes)
                    # for i in range(1, m_task):
                    #     _, _, diff = mod_main.module.pull2point(mod_main_centers[i], pull_strength=args.ae_offline_ps) # pull to the center variavle
                else:
                    raise ValueError('No named method')

                worse_perfomance = train_utils.each_batch_test(args, save_object, mod_main, te_loaders_full, m_task-1, local_test_loss)
                if worse_perfomance:
                    print(f"Recieved worse performance at batch {batch_idx}")
                    print(f"Rolling back to model on previous batch.")
                    # mod_main = copy.deepcopy(mod_main_copy)
                    # mod_main.load_state_dict(mod_main_ref)
                    # opt_main = copy.deepcopy(opt_main_copy)
                    # old_weights = torch.load(path)
                    # print(old_weights)
                    # return
                    # print(old_weights == mod_main.state_dict())
                    # return
                    # mod_main.load_state_dict(torch.load(path))
                    mod_main, added_layers_count = train_utils.handle_new_hidden_layer_logic(mod_main, args,
                                                              model_conf, added_layers_count,
                                                              m_task - 1)
                    save_object["expansion_epochs"].append(global_epoch + args.lr_epochs + (m_task - 2) * args.cl_epochs + cl_epoch)
                    print("Expansion epochs", save_object["expansion_epochs"])
                    if args.cl_method == 'ewc':
                        mod_main_centers = []
                        Fs = []
                        for _task in range(1, m_task + 1):
                            mod_main_centers, Fs = upgrade_mod_main_ewc(mod_main_centers, Fs, mod_main,
                                                                        dataset_name, _task, args, opt_main)
                    elif args.cl_method == 'dco':
                        mod_aes, opt_aes = [0], [0]
                        mod_main_centers = [0]
                        for _task in range(1, m_task + 1):
                            mod_main_centers, mod_aes, opt_aes, cl_opt_main = train_utils.upgrade_dco(mod_main_centers, mod_aes,
                                                                                          opt_aes, mod_main,
                                                                                          tr_loaders, args, m_task,
                                                                                          opt_main, None)

                    starting_point = utils.ravel_model_params(mod_main, False, 'cpu')
                    continue

                batch_idx += 1
                data, target = next(iter(tr_loaders[m_task]))


            if cl_epoch % log_interval == 0:
                errors = []
                for i in range(1, args.num_tasks + 1):
                    cur_error, test_loss = trainer.test(args, mod_main, te_loaders[i], i,
                                             global_epoch + args.lr_epochs + (m_task - 2) * args.cl_epochs + cl_epoch,
                                             task=i - 1)
                    errors += [cur_error]
                    # visdom_obj.line([cur_error], [global_epoch + args.lr_epochs + (m_task - 2) * args.cl_epochs + cl_epoch],
                    #                 update='append', opts={'title': '%d-Task Error' % i}, win='cur_error_%d' % i,
                    #                 name='T', env=f'{experiment_name}')

                # if args.cl_method == 'dco':
                    # for i in range(m_task - 1):
                        # visdom_obj.line([ae_loss[i].item()],
                        #                 [global_epoch + (m_task - 1) * args.cl_epochs + cl_epoch], update='append',
                        #                 opts={'title': '%d-AE Loss' % i}, win='ae_loss_%d' % i, name='T',
                        #                 env=f'{experiment_name}')
                    # print('The grad norm is', grad_norm)
                    # try:
                        # visdom_obj.line([grad_norm], [global_epoch + (m_task - 1) * args.cl_epochs + cl_epoch],
                        #                 update='append', opts={'title': 'Grad Norm'}, win='grad_norm', name='T',
                        #                 env=f'{experiment_name}')

                    # except:
                        # visdom_obj.line([grad_norm.item()],
                        #                 [global_epoch + (m_task - 1) * args.cl_epochs + cl_epoch], update='append',
                        #                 opts={'title': 'Grad Norm'}, win='grad_norm', name='T',
                        #                 env=f'{experiment_name}')
                current_point = utils.ravel_model_params(mod_main, False, 'cpu')
                l2_norm = (current_point - starting_point).norm().item()
                save_object["errors"] += [errors]
                # visdom_obj.line([l2_norm], [global_epoch + (m_task - 1) * args.cl_epochs + cl_epoch],
                #                 update='append', opts={'title': 'L2 Norm'}, win='l2_norm', name='T',
                #                 env=f'{experiment_name}')
                # visdom_obj.line([sum(errors) / args.num_tasks],
                #                 [global_epoch + (m_task - 1) * args.cl_epochs + cl_epoch], update='append',
                #                 opts={'title': 'Average Error'}, win='avg_error', name='T',
                #                 env=f'{experiment_name}')

            save_object["cur_task"] = m_task
            save_object["cur_epoch"] = cl_epoch
            # save_object["rbcl"] = rbcl
            save_object["local_test_loss"] = local_test_loss
            with open(f'{experiment_name}/checkpoint.pt', 'wb') as file:
                pickle.dump(save_object, file)
            # torch.save(save_object,
            #            f'{experiment_name}/checkpoint.pt')
            plotter.plot_error_from_data(save_object, save_path=f'{experiment_name}/plots')
            plotter.plot_local_batch_error_from_data(save_object, save_path=f'{experiment_name}/plots')
            epoch_time = train_utils.calc_time(epoch_time, "Epoch")
    errors = []
    for i in range(1, args.num_tasks + 1):
        cur_error, test_loss = trainer.test(args, mod_main, te_loaders[i], i,
                                 global_epoch + args.lr_epochs + (args.num_tasks-1) * args.cl_epochs, task=i - 1)
        errors += [cur_error]
        # visdom_obj.line([cur_error], [global_epoch + args.lr_epochs + (args.num_tasks-1) * args.cl_epochs],
        #                 update='append', opts={'title': '%d-Task Error' % i}, win='cur_error_%d' % i,
        #                 name='T', env=f'{experiment_name}')

    current_point = utils.ravel_model_params(mod_main, False, 'cpu')
    l2_norm = (current_point - starting_point).norm().item()
    save_object["errors"] += [errors]
    # visdom_obj.line([l2_norm], [global_epoch + args.lr_epochs + (args.num_tasks-1) * args.cl_epochs],
    #                 update='append', opts={'title': 'L2 Norm'}, win='l2_norm', name='T',
    #                 env=f'{experiment_name}')
    # visdom_obj.line([sum(errors) / args.num_tasks],
    #                 [global_epoch + args.lr_epochs + (args.num_tasks-1) * args.cl_epochs], update='append',
    #                 opts={'title': 'Average Error'}, win='avg_error', name='T',
    #                 env=f'{experiment_name}')

    with open(f'{experiment_name}/final.pt', 'wb') as file:
        pickle.dump(save_object, file)
    plotter.plot_error_from_data(save_object, save_path=f'{experiment_name}/plots')
    plotter.plot_local_batch_error_from_data(save_object, save_path=f'{experiment_name}/plots')
    print("Total time required: ")
    """
        To check the restuls, in Python3 with torch package imported: 
            (1) load average errors : average_errors = torch.load('res-%d.pt'%args.rank)
            (2) print average errors: print(average_errors[args.cl_method])
    """




def main():
    # // 1.1 Arguments //
    print('[INIT] ===> Defining models in process')
    args = parser.parse_args()
    args.device = 'cuda:%d' % args.rank if torch.cuda.is_available() else 'cpu'
    utils.save_options(args)
    torch.manual_seed(args.seed)  # https://pytorch.org/docs/stable/notes/randomness.html

    args.added_layer_conf = handle_layer_conf_args(args.added_layer_conf)

    if (args.added_layer_conf[0] != 0 and args.max_allowed_added_layers == 0) or (args.added_layer_conf[0] == 0 and args.max_allowed_added_layers != 0):
        raise Exception(f"Arguments provided are incompatible. Please change either them:\n--added_layer_conf[0] => {args.added_layer_conf[0]}\n--max_allowed_added_layers => {args.max_allowed_added_layers}")

    # Create experiment folder
    experiment_name = path_utils.create_experiment_path(args)

    # 1.1.1 Set up log file
    orig_stdout = sys.stdout

    f = open(f'{experiment_name}/log.txt', 'a')
    original = sys.stdout
    sys.stdout = path_utils.Tee(sys.stdout, f)


    run_main(args, experiment_name)
#  -------------- \\ Main codes begin here \\ -------------

def upgrade_mod_main_ewc(mod_main_centers, Fs, mod_main, dataset_name, m_task, args, opt_main):
    """
        (1) set batch_size = 1 for `ewc_loader` (a seperate dataloader for EWC method)
        (2) set m_label = m_task - 2 (because we are looking at the last task)
        (3) we save the elements of diagonal Fisher matrix in `ewc_grads`
            and we divide it by the number of data points in ewc_loader at the end
    """
    mod_main_center = copy.deepcopy(list(mod_main.parameters()))
    if dataset_name == 'permuted_mnist':
        ewc_loader = get_data_loader(get_dataset(dataset_name, m_task - 1, True), batch_size=1, cuda=False)

    elif dataset_name == 'split_mnist' or dataset_name == 'split_cifar10':
        m_label = m_task - 2
        ewc_loader = get_label_data_loader(get_dataset(dataset_name, m_task - 1, True), 1, cuda=False,
                                           labels=[m_label * 2, m_label * 2 + 1])
    elif dataset_name == 'split_cifar100':
        m_label = m_task - 2
        ewc_loader = get_label_data_loader(get_dataset(dataset_name, m_task - 1, True), 1, cuda=False,
                                           labels=[l for l in range(m_label * 10, m_label * 10 + 10)])
    ewc_grads = []
    for param in mod_main.parameters():
        ewc_grads += [torch.zeros_like(param)]
    for num, (data, target) in enumerate(ewc_loader, 1):
        main_loss = trainer.train(args, mod_main, opt_main, data, target, task=m_task - 1)
        try:
            main_loss.backward()
        except Exception as e:
            pass
        for param, grad in zip(mod_main.parameters(), ewc_grads):
            if param.grad is not None:
                grad.add_(1 / len(ewc_loader.dataset), param.grad ** 2)
    Fs += [ewc_grads]
    mod_main_centers += [mod_main_center]
    return mod_main_centers, Fs




def handle_layer_conf_args(layer_arg):
    layer_arg = layer_arg.split(",")
    for i in range(len(layer_arg)):
        layer_arg[i] = int(layer_arg[i])
    return layer_arg

if __name__ == '__main__':
    main()
