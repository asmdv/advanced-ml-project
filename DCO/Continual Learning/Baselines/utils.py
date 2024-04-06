import os
import os.path
import shutil
import torch
import visdom
import quadprog
import numpy as np

from torch.autograd import Variable
from torch.utils.data import DataLoader, Subset
from torch.utils.data.dataloader import default_collate
from torch.nn import init
from torchvision import datasets, transforms
# from qpth.qp import QPFunction

# ------- Visdom ------------
def get_visdom(args):
    vis = visdom.Visdom()
    if args.rank == 0:
        for env in vis.get_env_list():
            vis.delete_env(env)
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(args).items()):
        message += '{:>25}: {:<30}<br>'.format(str(k), str(v))
    message += '----------------- End -------------------'
    vis.text(message, win='information', env='gpu:%d'%args.rank)
    print('[Visdom] ===> Activated')
    return vis

# ------- checkpoints ------------
def save_checkpoint(model, model_dir, epoch, precision, best=True):
    path = os.path.join(model_dir, model.name)
    path_best = os.path.join(model_dir, '{}-best'.format(model.name))

    # save the checkpoint.
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    torch.save({
        'state': model.state_dict(),
        'epoch': epoch,
        'precision': precision,
    }, path)

    # override the best model if it's the best.
    if best:
        shutil.copy(path, path_best)
        print('=> updated the best model of {name} at {path}'.format(
            name=model.name, path=path_best
        ))

    # notify that we successfully saved the checkpoint.
    print('=> saved the model {name} to {path}'.format(
        name=model.name, path=path
    ))

def load_checkpoint(model, model_dir, best=True):
    path = os.path.join(model_dir, model.name)
    path_best = os.path.join(model_dir, '{}-best'.format(model.name))

    # load the checkpoint.
    checkpoint = torch.load(path_best if best else path)
    print('=> loaded checkpoint of {name} from {path}'.format(
        name=model.name, path=(path_best if best else path)
    ))

    # load parameters and return the checkpoint's epoch and precision.
    model.load_state_dict(checkpoint['state'])
    epoch = checkpoint['epoch']
    precision = checkpoint['precision']
    return epoch, precision

def validate(model, dataset, test_size=256, batch_size=32,
             cuda=False, verbose=True):
    mode = model.training
    model.train(mode=False)
    data_loader = get_data_loader(dataset, batch_size, cuda=cuda)
    total_tested = 0
    total_correct = 0
    for x, y in data_loader:
        # break on test size.
        if total_tested >= test_size:
            break
        # test the model.
        x = x.view(batch_size, -1)
        x = Variable(x).cuda() if cuda else Variable(x)
        y = Variable(y).cuda() if cuda else Variable(y)
        scores = model(x)
        _, predicted = scores.max(1)
        # update statistics.
        total_correct += int((predicted == y).sum())
        total_tested += len(x)
    model.train(mode=mode)
    precision = total_correct / total_tested
    if verbose:
        print('=> precision: {:.3f}'.format(precision))
    return precision

# ------------ manipute layers ------------
def xavier_initialize(model):
    modules = [
        m for n, m in model.named_modules() if
        'conv' in n or 'linear' in n
    ]

    parameters = [
        p for
        m in modules for
        p in m.parameters() if
        p.dim() >= 2
    ]

    for p in parameters:
        init.xavier_normal(p)

def gaussian_intiailize(model, std=.1):
    for p in model.parameters():
        init.normal(p, std=std)

def calculate_grad(ae, grad, wd):
    x = grad.view(ae.size(1), -1)
    grad = ((ae.mm(ae.t())).mm(ae) - ae).mm(x)
    # grad = grad.mm(x.t()) + wd* ae
    return grad, x

def layer_init(model, layer_names = []):
    for name, param in model.named_parameters():
        if name in layer_names:
            torch.nn.init.xavier_normal_(param)
            print('Re-initilize layer [%s]'%name)

def layer_zerograd(model, layer_names = []):
    for name, param in model.named_parameters():
        if name in layer_names:
            param.grad.zero_()

# options
def save_options(opt):
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        comment = ''
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'

    # save to the disk
    file_name = os.path.join('./', 'opt-{}-{}-{}.txt'.format(opt.cl_dataset, opt.cl_method, opt.rank))
    with open(file_name, 'wt') as opt_file:
        opt_file.write(message)
        opt_file.write('\n')

def ravel_list_params(model, is_grad, device):
    '''squash model parameters or gradients into a flat tensor (https://github.com/ucla-labx/distbelief)'''
    numel = 0
    for parameter in model:
        numel += parameter.data.numel()
    flat_tensor = torch.zeros(numel).to(device)
    current_index = 0
    for parameter in model:
        if is_grad:
            numel = parameter.grad.data.numel()
            flat_tensor[current_index:current_index+numel].copy_(parameter.grad.data.view(-1))
        else:
            numel = parameter.data.numel()
            flat_tensor[current_index:current_index+numel].copy_(parameter.data.view(-1))
        current_index += numel 
    return flat_tensor

def ravel_model_params(model, is_grad, device):
    '''squash model parameters or gradients into a flat tensor (https://github.com/ucla-labx/distbelief)'''
    numel = 0
    for parameter in model.parameters():
        numel += parameter.data.numel()
    flat_tensor = torch.zeros(numel).to(device)
    current_index = 0
    for parameter in model.parameters():
        if is_grad:
            numel = parameter.grad.data.numel()
            flat_tensor[current_index:current_index+numel].copy_(parameter.grad.data.view(-1))
        else:
            numel = parameter.data.numel()
            flat_tensor[current_index:current_index+numel].copy_(parameter.data.view(-1))
        current_index += numel 
    return flat_tensor

def assign_model_params(flat_tensor, model, is_grad):
    '''copy model parameters into the flat tensor'''
    current_index = 0 
    for parameter in model.parameters():
        if is_grad:
            size = parameter.grad.data.size()
            numel = parameter.grad.data.numel()
            parameter.grad.data.copy_(flat_tensor[current_index:current_index+numel].view(size))
        else:
            size = parameter.data.size()
            numel = parameter.data.numel()
            parameter.data.copy_(flat_tensor[current_index:current_index+numel].view(size))          
        current_index += numel

# TODO: check these out
def project2cone2(gradient, memories, margin=0.5, eps=1e-3):
    """
        Solves the GEM dual QP described in the paper given a proposed
        gradient "gradient", and a memory of task gradients "memories".
        Overwrites "gradient" with the final projected update.

        input:  gradient, p-vector
        input:  memories, (t * p)-vector
        output: x, p-vector
    """
    memories_np = memories.cpu().double().numpy()
    gradient_np = gradient.cpu().contiguous().view(-1).double().numpy()
    t = memories_np.shape[0]
    P = np.dot(memories_np, memories_np.transpose())
    P = 0.5 * (P + P.transpose()) + np.eye(t) * eps
    q = np.dot(memories_np, gradient_np) * -1
    G = np.eye(t)
    h = np.zeros(t) + margin
    v = quadprog.solve_qp(P, q, G, h)[0]
    x = np.dot(v, memories_np) + gradient_np
    return torch.FloatTensor(x)

def project2cone1(gradient, memories, margin=0.5, eps=1e-3):
    # TODO: the package has critical bugs
    """
        Solves the GEM dual QP described in the paper given a proposed
        gradient "gradient", and a memory of task gradients "memories".
        Overwrites "gradient" with the final projected update.

        input:  gradient, p-vector
        input:  memories, (t * p)-vector
        output: x, p-vector
    """
    memories_np = -memories.double()
    gradient_np = gradient.double()
    t = memories_np.size(0)
    P = memories_np.mm(memories_np.t())
    P = 0.5 * (P + P.t()) + torch.eye(t, device=gradient_np.device) * eps
    q = memories_np.mm(gradient_np.view(-1,1)).view(-1)
    G = -torch.eye(t, device=gradient_np.device,dtype=torch.double)
    h = torch.zeros(t, device=gradient_np.device,dtype=torch.double)
    e = torch.Tensor()
    # print(P.size(), q.size(), G.size(), h.size())
    # print(P.type(), q.type(), G.type(), h.type())
    v = QPFunction(maxIter=200)(P, q, G, h, e, e).view(-1, 1) + margin
    # print(v.size(), q.size())
    # print(0.5* v.t().mm(P).mm(v) + q.view(1, -1).mm(v))
    # print(memories_np.shape, v.shape, gradient_np.shape)

    return gradient_np + memories_np.t().mm(v).view(-1)