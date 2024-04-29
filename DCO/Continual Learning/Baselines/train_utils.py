import trainer
import torch
from tensordict import tensorclass
import random
import math
import torch.nn.functional as F
import path_utils
from torchrl.data import ReplayBuffer, ListStorage, LazyTensorStorage, LazyMemmapStorage
import utils
import time

def calc_time(t=None, name=None):
    if not t:
        return time.time()
    print(f"Time for {name}: {(time.time() - t):.6f}")
    return time.time()

@tensorclass
class ReplayBufferData:
    images: torch.Tensor
    labels: torch.Tensor

class ReplayBufferCL():
    def __init__(self, n_tasks, max_size, batch_size):
        self.buffer = []
        salt = path_utils.generate_random_string(8)
        for t in range(n_tasks):
            path_utils.overwrite_directory(f"./tempdir/{salt}/buffer_{t}")
            rb = ReplayBuffer(storage=LazyMemmapStorage(max_size=max_size, scratch_dir=f"./tempdir/buffer_{t}"), batch_size=batch_size)
            self.buffer.append(rb)


def get_samples(rbcl, m_task, args):
    if not args.replay_buffer_batch_size:
        return None
    samples = []
    for task_i in range(m_task - 1):
        samples.append(rbcl.buffer[task_i].sample())
    return samples


def handle_new_hidden_layer_logic(mod_main, args, model_conf, added_layers_count, m_task):
    if not (added_layers_count < args.max_allowed_added_layers):
        return mod_main, added_layers_count
    mod_local = mod_main.module if isinstance(mod_main, torch.nn.DataParallel) else mod_main
    if args.freeze:
        mod_local.freeze_all_but_last(m_task)

    if args.added_layer_conf[1] == 0:
        args.added_layer_conf[1] = model_conf['s_layer']

    mod_local.add_hidden_layerV3(args.added_layer_conf[1], count=args.added_layer_conf[0], same=args.added_layer_conf[1] == model_conf['s_layer'], task=m_task)
    print("Current mod_main tasks: ", mod_main.tasks)
    print("Current mod_main output tasks: ", mod_main.tasks_output)

    # mod_local.add_hidden_layer(len(mod_local.layers) - 3, model_conf['s_layer'])
    # mod_local.add_hidden_layer(len(mod_local.layers) - 3, model_conf['s_layer'])
    mod_main = mod_main.to(args.device)
    print("New Hidden Layer added.")
    print("After adding hidden layer:")
    mod_local.print_grad_req_for_all_params()


    return mod_main, added_layers_count + 1


def window_average(x, start, end=None):
    return sum(x[start:end]) / len(x[start:end])


def is_list_increase(x, w):
    old_increase = round(window_average(x, -2 * w, -w), 6)
    new_increase = round(window_average(x, -w), 6)
    if new_increase > old_increase:
        # print("Old")
        # print(x[-2*w:-w])
        # print("new")
        # print(x[-w:None])
        # new_increase = round(window_average(x, -w), 6)
        # print("New increase: ", new_increase)
        # print("Old increase", old_increase)
        pass
    return new_increase > old_increase, old_increase, new_increase


def each_batch_test(args, save_object, mod_main, te_loaders, task_i, local_test_loss):
    # Run tests for each batch
    # cl_error_threshold
    window = 5
    list_increase_confirmed = False
    local_test_batch_loss = [[] for _ in range(args.num_tasks)]
    local_test_batch_error = [[] for _ in range(args.num_tasks)]
    for i in range(args.num_tasks):
        # t = calc_time()
        curr_error, test_loss = trainer.test(args, mod_main, te_loaders[i + 1], i + 1, None, task=i, record=False)
        # t = calc_time(t, "trainer.test")
        local_test_batch_error[i] = curr_error
        local_test_batch_loss[i] = test_loss
        if mod_main.tasks_output[i] == mod_main.tasks_output[-1]:
            local_test_loss[i].append(test_loss)
        if mod_main.tasks_output[i] == mod_main.tasks_output[-1] and len(local_test_loss[i]) > 10 and args.max_allowed_added_layers > 0:
            list_increase, old_increase, new_increase = is_list_increase(local_test_loss[i], window)
            if list_increase and i < task_i and save_object["test_batch_error"][-1][i] > args.cl_error_threshold:
                print(f"List increase in Task [{i}]")
                print(f"Local test loss[{i}]", local_test_loss[0][-11:])
                print("Old/New:", old_increase, new_increase)
                list_increase_confirmed = True
                # print("Local test loss", local_test_loss[i])
                local_test_loss[i] = []
        local_test_loss[i] = local_test_loss[i][-4*window:]
        # print("Local test loss:", local_test_loss)

    save_object["test_batch_loss"].append(local_test_batch_loss)
    save_object["test_batch_error"].append(local_test_batch_error)
    return list_increase_confirmed

def train_sgd_cl(args, mod_main, opt_main, data, target, task):
    main_loss = trainer.train(args, mod_main, opt_main, data, target, task)
    try:
        main_loss.backward()
        opt_main.step()
    except Exception as e:
        pass

def train_sgd_cl_replay(samples, args, mod_main, opt_main):
    if not args.replay_buffer_batch_size:
        return
    for task_i, sample in enumerate(samples):
        train_sgd_cl(args, mod_main, opt_main, sample.images, sample.labels, task_i)


def train_ewc_cl(args, mod_main, opt_main, data, target, mod_main_centers, Fs, task):
    """ For each task we save a seperate Fisher matrix and a set of optimal parameters. """
    ewc_loss = 0
    main_loss = trainer.train(args, mod_main, opt_main, data, target, task=task)
    for mod_main_center, F_grad in zip(mod_main_centers, Fs):
        for p1, p2, coe in zip(mod_main.parameters(), mod_main_center, F_grad):
            ewc_loss += 1 / 2 * args.ewc_lam * (coe * F.mse_loss(p1, p2, reduction='none')).sum()
    (main_loss + ewc_loss).backward()
    opt_main.step()

def train_ewc_cl_replay(samples, args, mod_main, opt_main, mod_main_centers, Fs):
    if not args.replay_buffer_batch_size:
        return
    for task_i, sample in enumerate(samples):
        train_ewc_cl(args, mod_main, opt_main, sample.images, sample.labels, mod_main_centers, Fs, task_i)



def train_dco_cl(args, mod_main, opt_main, data, target, m_task, mod_main_centers, cl_opt_main, mod_aes, opt_aes, task):
    main_loss = trainer.train(args, mod_main, opt_main, data, target, task=task)
    ae_loss = []
    for i in range(1, m_task):
        ae_loss += [
            trainer.ae_reg(args, mod_main, mod_main_centers[i], cl_opt_main, mod_aes[i], opt_aes[i],
                           data, target)]
    sum(ae_loss).backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(mod_main.parameters(), args.ae_grad_norm)
    # cur_iteration
    main_loss.backward()
    opt_main.step()
    return ae_loss, grad_norm

def train_dco_cl_replay(samples, args, mod_main, opt_main, data, target, m_task, mod_main_centers, cl_opt_main, mod_aes, opt_aes):
    if not args.replay_buffer_batch_size:
        return
    for task_i, sample in enumerate(samples):
        ae_loss, grad_norm = train_dco_cl(args, mod_main, opt_main, data, target, m_task, mod_main_centers, cl_opt_main, mod_aes, opt_aes, task=task_i)
    return ae_loss, grad_norm


def add_to_replay_buffer(rbcl, m_task, data, target, args):
    if args.replay_buffer_batch_size:
        replayBufferData = ReplayBufferData(
            images=data,
            labels=target,
            batch_size=[args.train_batch_size],
        )
        rbcl.buffer[m_task - 1].extend(replayBufferData)
    return rbcl
def get_random_replay_batch_ids(start, end, args):
    random_replay_batch_id = []

    if args.replay_buffer_batch_size:
        random_replay_batch_id = random.sample(range(start, end), math.ceil(
            args.replay_buffer_batch_size / args.train_batch_size) * 2)
    return random_replay_batch_id

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
            main_loss = trainer.train(args, mod_main, opt_main, data, target, task=m_task-1)
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

def train_task_1(args, mod_main, opt_main, m_task, visdom_obj, save_object, tmp_fisher_vars, big_omegas, rbcl, opt_main_scheduler, log_interval, tr_loaders, te_loaders, global_epoch, experiment_name, starting_point):
    print(f"Training task 1")

    cur_iteration = 0
    for epoch in range(1, args.lr_epochs + 1):
        random_replay_batch_ids = get_random_replay_batch_ids(1, len(tr_loaders[1]) - 1, args)
        for batch_idx, (data, target) in enumerate(tr_loaders[1], 1):
            if batch_idx in random_replay_batch_ids:
                add_to_replay_buffer(rbcl, 1, data, target, args)

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

        opt_main_scheduler.step()
        if epoch % log_interval == 0:
            errors = []
            for i in range(1, args.num_tasks + 1):
                cur_error, test_loss = trainer.test(args, mod_main, te_loaders[i], i, global_epoch + epoch, task=i - 1)
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

            current_point = utils.ravel_model_params(mod_main, False, 'cpu')
            l2_norm = (current_point - starting_point).norm().item()
            visdom_obj.line([l2_norm], [global_epoch + epoch], update='append', opts={'title': 'L2 Norm'},
                            win='l2_norm', name='T', env=f'{experiment_name}')
            visdom_obj.line([sum(errors) / args.num_tasks], [global_epoch + epoch], update='append',
                            opts={'title': 'Average Error'}, win='avg_error', name='T', env=f'{experiment_name}')
            save_object["errors"] += [errors]
    return tmp_fisher_vars, big_omegas, global_epoch
