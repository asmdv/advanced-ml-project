import trainer
import torch
from tensordict import tensorclass
import random
import math
@tensorclass
class ReplayBufferData:
    images: torch.Tensor
    labels: torch.Tensor

def get_samples(rbcl, m_task):
    samples = []
    for task_i in range(m_task - 1):
        samples.append(rbcl.buffer[task_i].sample())
    return samples

def train_sgd_cl(args, mod_main, opt_main, data, target):
    main_loss = trainer.train(args, mod_main, opt_main, data, target)
    main_loss.backward()
    opt_main.step()

def train_sgd_cl_replay(samples, args, mod_main, opt_main):
    if not args.replay_buffer_batch_size:
        return
    for task_i, sample in enumerate(samples):
        train_sgd_cl(args, mod_main, opt_main, sample.images, sample.labels)


def train_ewc_cl(args, mod_main, opt_main, data, target, mod_main_centers, Fs):
    """ For each task we save a seperate Fisher matrix and a set of optimal parameters. """
    ewc_loss = 0
    main_loss = trainer.train(args, mod_main, opt_main, data, target)
    for mod_main_center, F_grad in zip(mod_main_centers, Fs):
        for p1, p2, coe in zip(mod_main.parameters(), mod_main_center, F_grad):
            ewc_loss += 1 / 2 * args.ewc_lam * (coe * F.mse_loss(p1, p2, reduction='none')).sum()
    (main_loss + ewc_loss).backward()
    opt_main.step()

def train_ewc_cl_replay(samples, args, mod_main, opt_main, mod_main_centers, Fs):
    if not args.replay_buffer_batch_size:
        return
    for task_i, sample in enumerate(samples):
        train_ewc_cl(args, mod_main, opt_main, sample.images, sample.labels, mod_main_centers, Fs)



def train_dco_cl(args, mod_main, opt_main, data, target, m_task, mod_main_centers, cl_opt_main, mod_aes, opt_aes):
    main_loss = trainer.train(args, mod_main, opt_main, data, target)
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
        ae_loss, grad_norm = train_dco_cl(args, mod_main, opt_main, data, target, m_task, mod_main_centers, cl_opt_main, mod_aes, opt_aes)
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



