import trainer
import torch
from tensordict import tensorclass
import random
import math
import torch.nn.functional as F
import path_utils
from torchrl.data import ReplayBuffer, ListStorage, LazyTensorStorage, LazyMemmapStorage


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
    mod_local.print_grad_req_for_all_params()
    if args.freeze:
        mod_local.freeze_all_but_last(m_task)
    mod_local.print_grad_req_for_all_params()

    if args.added_layer_conf[1] == 0:
        args.added_layer_conf[1] = model_conf['s_layer']

    mod_local.add_hidden_layerV3(args.added_layer_conf[1], count=args.added_layer_conf[0], same=args.added_layer_conf[1] == model_conf['s_layer'], task=m_task)
    print("Current mod_main tasks: ", mod_main.tasks)
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
    old_increase = window_average(x, -2 * w, -w)
    new_increase = window_average(x, -w)
    return new_increase > old_increase


def each_batch_test(args, save_object, mod_main, te_loaders, task_i, local_test_loss):
    # Run tests for each batch
    for i in range(args.num_tasks):
        curr_error, test_loss = trainer.test(args, mod_main, te_loaders[task_i + 1], task_i + 1, None, i, False)
        save_object["test_batch_loss"][i].append(test_loss)
        save_object["test_batch_error"][i].append(curr_error)
        local_test_loss[i].append(test_loss)
        mod_main.tasks_output[i]
        if mod_main.tasks_output[i] == mod_main.tasks_output[task_i] and len(local_test_loss[i]) > 10:
            list_increase = is_list_increase(local_test_loss[i], 5)
            if list_increase and i < task_i:
                print(f"List increase in Task [{i}]")
                print("Local test loss", local_test_loss[i])
                local_test_loss[i] = []
                return True
    return False

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

