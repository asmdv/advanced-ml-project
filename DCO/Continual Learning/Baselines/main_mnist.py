import copy
import math
import time
import sys
import os
import string
import shutil
import torch
import torch.nn.functional as F
import datetime
import random
import plotter
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchrl.data import ReplayBuffer, ListStorage, LazyTensorStorage, LazyMemmapStorage
import utils
import trainer
import visdom
from tensordict import tensorclass
import tempfile
from models import MLP, Conv
from options import parser
from data import get_dataset, get_data_loader, get_label_data_loader, map_dataset
from data import DATASET_CONFIGS, MODEL_CONFIGS
from algorithm import train_invauto
import train_utils



class Tee(object):
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()  # If you want the output to be visible immediately

    def flush(self):
        for f in self.files:
            f.flush()


def handle_new_hidden_layer_logic(mod_main, args, model_conf, added_layers_count):
    if not (added_layers_count < args.max_allowed_added_layers):
        return mod_main, added_layers_count
    mod_local = mod_main.module if isinstance(mod_main, torch.nn.DataParallel) else mod_main
    if args.freeze:
        mod_local.freeze_all_but_last()

    if args.added_layer_conf[1] == 0:
        args.added_layer_conf[1] = model_conf['s_layer']
    mod_local.add_hidden_layerV2(len(mod_local.layers) - 3 - args.added_layer_conf[2] * 2, args.added_layer_conf[1], count=args.added_layer_conf[0], same=args.added_layer_conf[1] == model_conf['s_layer'])

    # mod_local.add_hidden_layer(len(mod_local.layers) - 3, model_conf['s_layer'])
    # mod_local.add_hidden_layer(len(mod_local.layers) - 3, model_conf['s_layer'])
    mod_main = mod_main.to(args.device)
    print("New Hidden Layer added.")

    return mod_main, added_layers_count + 1




def generate_random_string(length):
    letters = string.ascii_letters
    return ''.join(random.choice(letters) for _ in range(length))


class ReplayBufferCL():
    def __init__(self, n_tasks, max_size, batch_size):
        self.buffer = []
        salt = generate_random_string(8)
        for t in range(n_tasks):
            overwrite_directory(f"./tempdir/{salt}/buffer_{t}")
            rb = ReplayBuffer(storage=LazyMemmapStorage(max_size=max_size, scratch_dir=f"./tempdir/buffer_{t}"), batch_size=batch_size)
            self.buffer.append(rb)


def create_experiment_path(args):
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d_%H_%M_%S")
    freeze_name = "freeze" if args.freeze else "no-freeze"
    experiment_name = f"experiments/exp_{args.cl_dataset}_{args.cl_method}_n_tasks_{args.num_tasks}_epochs_{args.lr_epochs}_{args.cl_epochs}_threshold_{args.cl_error_threshold}_max_layers_{args.max_allowed_added_layers}_{freeze_name}_layers_{'_'.join(str(x) for x in args.added_layer_conf)}_rb_{args.replay_buffer_batch_size}_{formatted_time}"
    create_directory_if_not_exists(experiment_name)
    create_directory_if_not_exists(f"{experiment_name}/plots")
    return experiment_name


#  -------------- \\ Main codes begin here \\ -------------
def main():
    # // 1.1 Arguments //
    print('[INIT] ===> Defining models in process')
    args = parser.parse_args()
    args.device = 'cuda:%d' % args.rank if torch.cuda.is_available() else 'cpu'
    utils.save_options(args)
    torch.manual_seed(args.seed)  # https://pytorch.org/docs/stable/notes/randomness.html
    save_object = {"errors": [], "n_tasks": args.num_tasks}

    if args.replay_buffer_batch_size:
        rbcl = ReplayBufferCL(n_tasks=args.num_tasks, max_size=3 * args.replay_buffer_batch_size, batch_size=args.replay_buffer_batch_size)

    args.added_layer_conf = handle_layer_conf_args(args.added_layer_conf)
    if (args.added_layer_conf[0] != 0 and args.max_allowed_added_layers == 0) or (args.added_layer_conf[0] == 0 and args.max_allowed_added_layers != 0):
        raise Exception(f"Arguments provided are incompatible. Please change either them:\n--added_layer_conf[0] => {args.added_layer_conf[0]}\n--max_allowed_added_layers => {args.max_allowed_added_layers}")

    # Create experiment folder
    experiment_name = create_experiment_path(args)

    # 1.1.1 Set up log file
    orig_stdout = sys.stdout
    f = open(f'{experiment_name}/log.txt', 'w')
    original = sys.stdout
    sys.stdout = Tee(sys.stdout, f)


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
    try:
        visdom_obj = utils.get_visdom(args, experiment_name)
    except Exception as e:
        print(e)
        print('[Visdom] ===> De-activated')

    # // 1.4 Define task datasets and their dataloders // 
    """
        (1) structure of the task loader is: [0, dataset_1, dataset_2, ..., dataset_n]
        (2) for permuted_mnist: the permutation pattern is controlled by np.random.seed(m_task) in `data.py`
        (3) for split mnist and split cifar-100: the sequence of tasks are defined by a sequence of labels
    """
    tr_loaders, te_loaders = [0], [0]
    for m_task in range(1, args.num_tasks + 1):
        if dataset_name == 'permuted_mnist':
            tr_loaders += [get_data_loader(get_dataset(dataset_name, m_task, True), args.train_batch_size,
                                           cuda=('cuda' in args.device))]
            te_loaders += [get_data_loader(get_dataset(dataset_name, m_task, False), args.train_batch_size,
                                           cuda=('cuda' in args.device))]
        elif dataset_name == 'split_mnist' or dataset_name == 'split_cifar10':
            m_label = m_task - 1
            tr_loaders += [get_label_data_loader(get_dataset(dataset_name, m_task, True), args.train_batch_size,
                                                 cuda=('cuda' in args.device), labels=[m_label * 2, m_label * 2 + 1])]
            te_loaders += [get_label_data_loader(get_dataset(dataset_name, m_task, False), args.train_batch_size,
                                                 cuda=('cuda' in args.device), labels=[m_label * 2, m_label * 2 + 1])]
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

    adding_new_hidden_layer = False
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
    print(f"Training task 1")


    cur_iteration = 0
    for epoch in range(1, args.lr_epochs + 1):
        random_replay_batch_ids = train_utils.get_random_replay_batch_ids(1, len(tr_loaders[1]), args)
        for batch_idx, (data, target) in enumerate(tr_loaders[1], 1):
            if batch_idx in random_replay_batch_ids:
                train_utils.add_to_replay_buffer(rbcl, m_task, data, target, args)

            cur_iteration += 1
            if args.cl_method == 'si' or args.cl_method == 'rwalk':
                param1 = utils.ravel_model_params(mod_main, False, 'cpu')
                main_loss = trainer.train(args, mod_main, opt_main, data, target)
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
                main_loss = trainer.train(args, mod_main, opt_main, data, target)
                main_loss.backward()
                opt_main.step()
        opt_main_scheduler.step()
        if epoch % log_interval == 0:
            errors = []
            for i in range(1, args.num_tasks + 1):
                cur_error = trainer.test(args, mod_main, te_loaders[i], i, global_epoch + epoch)
                errors += [cur_error]
                visdom_obj.line([cur_error], [global_epoch + epoch], update='append',
                                opts={'title': '%d-Task Error' % i}, win='cur_error_%d' % i, name='T',
                                env=f'{experiment_name}')
                # Checking only task 1
                # if epoch % args.lr_epochs == 0 and i == 1 and cur_error > cl_error_threshold:
                #     print("Current error is bigger than threshold. Adding the new layer.")
                #     # adding_new_hidden_layer = True
                #     mod_main, added_layers_count = handle_new_hidden_layer_logic(mod_main, args, result_list, model_conf, added_layers_count)
                #     # break
                # elif epoch % args.lr_epochs == 0 and i == 1:
                #     print(f"Success. Task 1 Error: {cur_error:.2f}. No need for adding layer")

            # if adding_new_hidden_layer:
            #     global_epoch += epoch
            #     break
            current_point = utils.ravel_model_params(mod_main, False, 'cpu')
            l2_norm = (current_point - starting_point).norm().item()
            visdom_obj.line([l2_norm], [global_epoch + epoch], update='append', opts={'title': 'L2 Norm'},
                            win='l2_norm', name='T', env=f'{experiment_name}')
            visdom_obj.line([sum(errors) / args.num_tasks], [global_epoch + epoch], update='append',
                            opts={'title': 'Average Error'}, win='avg_error', name='T', env=f'{experiment_name}')
            save_object["errors"] += [errors]

    # if adding_new_hidden_layer:
    #     mod_main = handle_new_hidden_layer_logic(mod_main, args, result_list, model_conf)
    #     added_layers_count += 1
    #     continue
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
            mod_main_centers, Fs = upgrade_mod_main_ewc(mod_main_centers, Fs, mod_main, dataset_name, m_task, args, opt_main)
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
            mod_main_centers, mod_aes, opt_aes, cl_opt_main = upgrade_dco(mod_main_centers, mod_aes, opt_aes, mod_main, tr_loaders, args, m_task, opt_main, visdom_obj)
        else:
            raise ValueError('No named method')

        # training
        cur_iteration = 0
        for cl_epoch in range(args.cl_epochs):
            random_replay_batch_ids = train_utils.get_random_replay_batch_ids(1, len(tr_loaders[1]), args)
            for batch_idx, (data, target) in enumerate(tr_loaders[m_task]):
                # Adding replay buffer
                if batch_idx in random_replay_batch_ids:
                    train_utils.add_to_replay_buffer(rbcl, m_task, data, target, args)

                samples = train_utils.get_samples(rbcl, m_task)

                cur_iteration += 1
                if args.cl_method == 'sgd':
                    train_utils.train_sgd_cl(args, mod_main, opt_main, data, target)
                    train_utils.train_sgd_cl_replay(samples, args, mod_main, opt_main)

                elif args.cl_method == 'ewc':
                    train_utils.train_ewc_cl(args, mod_main, opt_main, data, target, mod_main_centers, Fs)
                    train_utils.train_ewc_cl_replay(samples, args, mod_main, opt_main, mod_main_centers, Fs)

                elif args.cl_method == 'si':
                    """ SI algorithm adds per-parameter regularization loss to the total loss """
                    param1 = utils.ravel_model_params(mod_main, False, 'cpu')
                    main_loss = trainer.train(args, mod_main, opt_main, data, target)
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
                    main_loss = trainer.train(args, mod_main, opt_main, data, target)
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
                    main_loss = trainer.train(args, mod_main, opt_main, data, target)
                    main_loss.backward()
                    m_grad = utils.ravel_model_params(mod_main, True, args.device)
                    mod_main.zero_grad()
                    if args.cl_method == 'agem':
                        agem_iter = agem_loader.__iter__()
                        agem_data, agem_target = next(agem_iter)
                        agem_target = agem_target.view(-1)
                        main_loss = trainer.train(args, mod_main, opt_main, agem_data, agem_target)
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
                            main_loss = trainer.train(args, mod_main, opt_main, gem_data, gem_target)
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
                    ae_loss, grad_norm = train_utils.train_dco_cl(args, mod_main, opt_main, data, target, m_task, mod_main_centers, cl_opt_main, mod_aes, opt_aes)
                    ae_loss, grad_norm = train_utils.train_dco_cl_replay(samples, args, mod_main, opt_main, data, target, m_task, mod_main_centers, cl_opt_main, mod_aes, opt_aes)
                    # for i in range(1, m_task):
                    #     _, _, diff = mod_main.module.pull2point(mod_main_centers[i], pull_strength=args.ae_offline_ps) # pull to the center variavle
                else:
                    raise ValueError('No named method')
            if cl_epoch % log_interval == 0:
                errors = []
                for i in range(1, args.num_tasks + 1):
                    cur_error = trainer.test(args, mod_main, te_loaders[i], i,
                                             global_epoch + (m_task - 1) * args.cl_epochs + (cl_epoch + 1))
                    errors += [cur_error]
                    visdom_obj.line([cur_error], [global_epoch + (m_task - 1) * args.cl_epochs + (cl_epoch + 1)],
                                    update='append', opts={'title': '%d-Task Error' % i}, win='cur_error_%d' % i,
                                    name='T', env=f'{experiment_name}')

                    # cl_epoch == (args.cl_epochs - 1) and
                    # cl_epoch == (args.cl_epochs - 1) and
                    if i <= m_task and cur_error > cl_error_threshold:
                        print(f"Current error is bigger than threshold for task {i}.")
                        adding_new_hidden_layer = True
                        # break
                    elif i <= m_task:
                        print(f"Success. Task {i} Error: {cur_error:.2f}. No need for adding layer")

                if args.cl_method == 'dco':
                    for i in range(m_task - 1):
                        visdom_obj.line([ae_loss[i].item()],
                                        [global_epoch + (m_task - 1) * args.cl_epochs + cl_epoch], update='append',
                                        opts={'title': '%d-AE Loss' % i}, win='ae_loss_%d' % i, name='T',
                                        env=f'{experiment_name}')
                    print('The grad norm is', grad_norm)
                    try:
                        visdom_obj.line([grad_norm], [global_epoch + (m_task - 1) * args.cl_epochs + cl_epoch],
                                        update='append', opts={'title': 'Grad Norm'}, win='grad_norm', name='T',
                                        env=f'{experiment_name}')
                    except:
                        visdom_obj.line([grad_norm.item()],
                                        [global_epoch + (m_task - 1) * args.cl_epochs + cl_epoch], update='append',
                                        opts={'title': 'Grad Norm'}, win='grad_norm', name='T',
                                        env=f'{experiment_name}')
                current_point = utils.ravel_model_params(mod_main, False, 'cpu')
                l2_norm = (current_point - starting_point).norm().item()
                save_object["errors"] += [errors]
                visdom_obj.line([l2_norm], [global_epoch + (m_task - 1) * args.cl_epochs + cl_epoch],
                                update='append', opts={'title': 'L2 Norm'}, win='l2_norm', name='T',
                                env=f'{experiment_name}')
                visdom_obj.line([sum(errors) / args.num_tasks],
                                [global_epoch + (m_task - 1) * args.cl_epochs + cl_epoch], update='append',
                                opts={'title': 'Average Error'}, win='avg_error', name='T',
                                env=f'{experiment_name}')

                if adding_new_hidden_layer:
                    mod_main, added_layers_count = handle_new_hidden_layer_logic(mod_main, args,
                                                                                 model_conf, added_layers_count)
                    if args.cl_method == 'ewc':
                        mod_main_centers = []
                        Fs = []
                        for _task in range(1, m_task + 1):
                            mod_main_centers, Fs = upgrade_mod_main_ewc(mod_main_centers, Fs, mod_main, dataset_name, _task, args, opt_main)
                    elif args.cl_method == 'dco':
                        mod_aes, opt_aes = [0], [0]
                        mod_main_centers = [0]
                        for _task in range(1, m_task + 1):
                            mod_main_centers, mod_aes, opt_aes, cl_opt_main = upgrade_dco(mod_main_centers, mod_aes, opt_aes, mod_main, tr_loaders, args, m_task, opt_main, visdom_obj)


                    starting_point = utils.ravel_model_params(mod_main, False, 'cpu')
                    adding_new_hidden_layer = False

            torch.save(save_object,
                       f'{experiment_name}/checkpoint.pt')

        # # Doing replay - old way
        # if args.cl_method == 'sgd':
        #     handle_replay_sgd(m_task, args, mod_main, opt_main, rbcl)
        # elif args.cl_method == 'ewc':
        #     handle_replay_ewc(m_task, args, mod_main, opt_main, rbcl, mod_main_centers, Fs)
        # elif args.cl_method == 'dco':
        #     handle_replay_dco(m_task, args, mod_main, opt_main, rbcl, mod_main_centers, cl_opt_main, mod_aes, opt_aes)

    errors = []
    for i in range(1, args.num_tasks + 1):
        cur_error = trainer.test(args, mod_main, te_loaders[i], i,
                                 global_epoch + (args.num_tasks) * args.cl_epochs + 1)
        errors += [cur_error]
        visdom_obj.line([cur_error], [global_epoch + (args.num_tasks) * args.cl_epochs + 1],
                        update='append', opts={'title': '%d-Task Error' % i}, win='cur_error_%d' % i,
                        name='T', env=f'{experiment_name}')

        # cl_epoch == (args.cl_epochs - 1) and
        # cl_epoch == (args.cl_epochs - 1) and
        if i <= m_task and cur_error > cl_error_threshold:
            print(f"Current error is bigger than threshold for task {i}.")
            adding_new_hidden_layer = True
            # break
        elif i <= m_task:
            print(f"Success. Task {i} Error: {cur_error:.2f}. No need for adding layer")
    current_point = utils.ravel_model_params(mod_main, False, 'cpu')
    l2_norm = (current_point - starting_point).norm().item()
    save_object["errors"] += [errors]
    visdom_obj.line([l2_norm], [global_epoch + (args.num_tasks) * args.cl_epochs + 1],
                    update='append', opts={'title': 'L2 Norm'}, win='l2_norm', name='T',
                    env=f'{experiment_name}')
    visdom_obj.line([sum(errors) / args.num_tasks],
                    [global_epoch + (args.num_tasks) * args.cl_epochs + 1], update='append',
                    opts={'title': 'Average Error'}, win='avg_error', name='T',
                    env=f'{experiment_name}')

                # if adding_new_hidden_layer:
                #     global_epoch += (m_task-1)* args.cl_epochs+cl_epoch
                #     break

                # comment

        # if adding_new_hidden_layer:
        #     break
    # if adding_new_hidden_layer:
    #     mod_main, added_layers_count = handle_new_hidden_layer_logic(mod_main, args, result_list, model_conf, added_layers_count)
    #     continue
    torch.save(save_object,
               f'{experiment_name}/final.pt')
    plotter.plot_error_from_data(save_object, save_path=f'{experiment_name}/plots')

    """
        To check the restuls, in Python3 with torch package imported: 
            (1) load average errors : average_errors = torch.load('res-%d.pt'%args.rank)
            (2) print average errors: print(average_errors[args.cl_method])
    """







# def handle_replay_sgd(m_task, args, mod_main, opt_main, rbcl):
#     print("Running replay sgd")
#     for _ in range(5):
#         print(f"Replay iteration {_}. Tasks: ", end="")
#         for prev_task in range(m_task):
#             print(f"{prev_task}", end=" ")
#             replayBufferData = rbcl.buffer[prev_task].sample()
#             train_utils.train_sgd_cl(args, mod_main, opt_main, replayBufferData.images, replayBufferData.labels)
#         print()
#
# def handle_replay_ewc(m_task, args, mod_main, opt_main, rbcl, mod_main_centers, Fs):
#     print("Running replay ewc")
#     for _ in range(5):
#         print(f"Replay iteration {_}. Tasks: ", end="")
#         for prev_task in range(m_task):
#             print(f"{prev_task}", end=" ")
#             replayBufferData = rbcl.buffer[prev_task].sample()
#             train_utils.train_ewc_cl(args, mod_main, opt_main, replayBufferData.images, replayBufferData.labels, mod_main_centers, Fs)
#         print()

# def handle_replay_dco(m_task, args, mod_main, opt_main, rbcl, mod_main_centers, cl_opt_main, mod_aes, opt_aes):
#     print("Running replay ewc")
#     for _ in range(5):
#         print(f"Replay iteration {_}. Tasks: ", end="")
#         for prev_task in range(m_task):
#             print(f"{prev_task}", end=" ")
#             replayBufferData = rbcl.buffer[prev_task].sample()
#             train_dco_cl(args, mod_main, opt_main, replayBufferData.images, replayBufferData.labels, m_task, mod_main_centers, cl_opt_main, mod_aes, opt_aes)
#         print()

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
        main_loss = trainer.train(args, mod_main, opt_main, data, target)
        main_loss.backward()
        for param, grad in zip(mod_main.parameters(), ewc_grads):
            if param.grad is not None:
                grad.add_(1 / len(ewc_loader.dataset), param.grad ** 2)
    Fs += [ewc_grads]
    mod_main_centers += [mod_main_center]
    return mod_main_centers, Fs


def upgrade_dco(mod_main_centers, mod_aes, opt_aes, mod_main, tr_loaders, args, m_task, opt_main, visdom_obj):
    # Method 1: pushing inside the cone
    # mod_main_center = copy.deepcopy(list(mod_main.parameters()))
    # for _ in range(args.prox_epochs-1):
    #     for batch_idx, (data, target) in enumerate(tr_loaders[m_task-1], 1):
    #         main_loss = trainer.train(args, mod_main, opt_main, data, target)
    #         main_loss.backward()
    #         opt_main.step()
    #         mod_main.module.pull2point(mod_main_center, pull_strength=0.1) # pull to the center variavle
    # for data, target in tr_loaders[m_task-1]:
    #     main_loss = trainer.train(args, mod_main, opt_main, data, target)
    #     main_loss.backward()
    #     opt_main.step()
    #     mod_main.module.pull2point(mod_main_center, pull_strength=0.1) # pull to the center variavle
    # center_estimations = utils.ravel_model_params(mod_main, False, args.device)

    # Method 2: pushing inside the cone
    mod_main_center = copy.deepcopy(list(mod_main.parameters()))
    corner1 = utils.ravel_model_params(mod_main, False, 'cpu')
    corner1.zero_()
    corner2 = corner1.clone()
    for ep in range(args.prox_epochs):
        for batch_idx, (data, target) in enumerate(tr_loaders[m_task - 1], 1):
            main_loss = trainer.train(args, mod_main, opt_main, data, target)
            main_loss.backward()
            opt_main.step()
            if ep == 0 and batch_idx <= 16:
                corner1.add_(1 / 16, utils.ravel_model_params(mod_main, False, 'cpu'))
            if ep == args.prox_epochs - 1 and batch_idx > len(tr_loaders[m_task - 1]) - 16:
                corner2.add_(1 / 16, utils.ravel_model_params(mod_main, False, 'cpu'))
    move_step = corner2 - corner1
    center_estimations = corner1 + move_step / move_step.norm() * args.push_cone_l2

    utils.assign_model_params(center_estimations, mod_main, is_grad=False)
    mod_main_centers += [copy.deepcopy(list(mod_main.parameters()))]
    mod_ae, opt_ae = train_invauto(args, m_task - 2, mod_main, mod_main_centers[m_task - 1],
                                   tr_loaders[m_task - 1], visdom_obj)
    mod_aes += [mod_ae]
    opt_aes += [opt_ae]
    print('[AE/CL] ===> Using AE model for Continual Learning')
    cl_opt_main = torch.optim.Adam(mod_main.parameters(), lr=args.main_online_lr)
    return mod_main_centers, mod_aes, opt_aes, cl_opt_main
def create_directory_if_not_exists(directory_path):
    """
    Create a directory if it doesn't exist.

    Parameters:
    directory_path (str): Path of the directory to be created.
    """
    if not os.path.exists(directory_path):
        try:
            os.makedirs(directory_path)
            print(f"Directory '{directory_path}' created successfully.")
        except OSError as err:
            print(f"Error: Creating directory '{directory_path}' - {err}")
    else:
        print(f"Directory '{directory_path}' already exists.")

def overwrite_directory(directory_path):
    if os.path.exists(directory_path):
        try:
            shutil.rmtree(directory_path)
        except OSError as err:
            print(f"Error: Unable to remove directory '{directory_path}' - {err}")
    try:
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created successfully.")
    except OSError as err:
        print(f"Error: Creating directory '{directory_path}' - {err}")

def handle_layer_conf_args(layer_arg):
    layer_arg = layer_arg.split(",")
    for i in range(len(layer_arg)):
        layer_arg[i] = int(layer_arg[i])
    return layer_arg

if __name__ == '__main__':
    main()
