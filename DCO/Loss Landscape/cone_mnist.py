import datetime
import pdb
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as Data
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms

import utils
import random
import itertools
import numpy as np
import matplotlib.pyplot as plt

# -------------- \\ Class and Function \\ -------------------
def get_data_loader(batch_size, desired_labels = [0, 1]):
    ''' define training and testing data loader of MNIST'''
    train_data = datasets.MNIST('./Datasets/mnist', train=True, download=True,
                               transform = transforms.Compose([
                               transforms.Resize(8),
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                            ]))
    test_data = datasets.MNIST('./Datasets/mnist', train=False,
                               transform = transforms.Compose([
                               transforms.Resize(8),
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                            ]))
    train_labels = train_data.targets
    test_labels = test_data.targets
    train_index = [i for i in range(len(train_labels)) if train_labels[i].item() in desired_labels]
    test_index = [i for i in range(len(test_labels)) if test_labels[i].item() in desired_labels]
    train_data = Data.Subset(train_data, train_index)
    test_data = Data.Subset(test_data, test_index)

    # kwargs = {'num_workers': 4, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, test_loader

class FCNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(64, 30, bias=False), nn.ReLU(),
            nn.Linear(30, 30, bias=False), nn.ReLU(),
            nn.Linear(30,  2, bias=False))
    def forward(self,x):
        output = self.net(x.view(-1, 64))
        return output

# ------------ \\ step 1: define model and train loader \\ -----------------
fontdict = {'fontsize':20,'fontweight':22}
titledict = {'fontsize':22,'fontweight':24}


device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
    print('CUDA is available! Using GPU.')
else:
    print('CUDA is not available. Using CPU.')

torch.manual_seed(24)
model = FCNet().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.001)
train_loader1, test_loader1 = get_data_loader(128, [0, 1])

try:
    model.load_state_dict(torch.load('./illustrative_example/model_state.pt'))
except:
    start_time = datetime.datetime.now().replace(microsecond=0)
    # scheduler = StepLR(optimizer, step_size=1, gamma=0.95)
    for epoch in range(1, 80 + 1):
        for batch_idx, (data, target) in enumerate(train_loader1):
            optimizer.zero_grad()
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
        # scheduler.step()
        with torch.no_grad():
            test_loss = 0
            correct = 0
            for data, target in test_loader1:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
            test_loss /= len(test_loader1.dataset)
            print('[{:2d}]Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(epoch,
                    test_loss, correct, len(test_loader1.dataset), 100. * correct / len(test_loader1.dataset)))
    optimizer.zero_grad()
    torch.save(model.state_dict(), './illustrative_example/model_state.pt')
model_copy = utils.ravel_model_params(model, is_grad=False, device=device)

# ------------ \\ step 2: train classifier for 50 epochs \\ -----------------
m_params = []
start_time = datetime.datetime.now().replace(microsecond=0)
for epoch in range(1, 50 + 1):
    if epoch % 10 == 0:
        print('%d, '%epoch, end = '')
    for batch_idx, (data, target) in enumerate(train_loader1):
        optimizer.zero_grad()
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        m_params += [utils.ravel_model_params(model, is_grad=True, device=device)]
        optimizer.step()
print('trianining finished !!')

optimizer.zero_grad()
m_params = torch.stack(m_params)
n_params = m_params # - m_params.mean(dim=0, keepdim=True) # we do not normalize the gradients
eval, evec = torch.linalg.eig(n_params.t().mm(n_params))

# ------------ \\ step 3: generate plots \\ -----------------
# [ Observation 1: landscapes: fix alpha and change beta ]
all_train_loader1, all_test_loader1 = get_data_loader(12800, [0, 1])
for data, target in all_train_loader1:
    data, target = data.to(device), target.to(device)
print('The size of whole training dataset is %d'%len(data))
plt.figure()
plt.xlabel(r'$\beta$', fontdict=fontdict)
plt.ylabel(r'$f(0, \beta, v_i)$',fontdict=fontdict)
ks = [0, 10, 50, 100, 200, 500]  # + [i for i in range(100, 201, 100)]
labels = ['1st eigenvector', '11th eigenvector', '51st eigenvector', '101st eigenvector', '201st eigenvector', '501st eigenvector']
for i, k in enumerate(ks):
    cur_k = []
    for prop in range(-10, 11, 1):
        cur_prop = prop / 10
        utils.assign_model_params(model_copy + cur_prop* evec[:, k], model, False)
        output = model(data)
        cur_k += [F.cross_entropy(output, target).item()]
    plt.plot(cur_k, label=labels[i])
ax = plt.gca()
ax.tick_params(axis = 'x', which = 'major', labelsize = 12)
ax.tick_params(axis = 'y', which = 'major', labelsize = 12)
plt.grid(True)
plt.legend(prop={'size': 15})
plt.xticks(list(range(21)), ['' for prop in range(-10, 11, 1)])
plt.xticks([0, 5, 10, 15, 20], [-1.0, -0.05, 0.0, 0.05, 1.0])
plt.tight_layout()  
plt.ylim(0.02, 0.1)
plt.savefig('./illustrative_example/landscape.jpg')
plt.ylim(0.02, 0.04)
plt.tight_layout()  
plt.savefig('./illustrative_example/landscape_zoom.jpg')
plt.close()

# [ Observation 3: landscape: change both alpha and beta ]
'''
    cur_m: alpha; cur_prop: beta
    m_params.mean(dim=0): average gradients
    evec_[:, k]         : (k+1)-th eigendirection
'''
for i, k in enumerate(ks):
    plt.figure()
    plt.xlabel(r'$\beta$', fontdict=fontdict)
    plt.ylabel(r'$f(\alpha, \beta, v_i)$',fontdict=fontdict)
    for m in range(0, 5, 1):
        cur_m = 5* m  # const $5$ is just step size for alpha
        cur_k = []    # index of eigenvector
        x_labels = [] # the x-axis labels
        for prop in range(-10, 11, 1):
            if i == 0: # the step range of beta is smaller for 1st eigendirection
                cur_prop = prop / (8)
            else:
                cur_prop = prop / (1)
            x_labels += [cur_prop]
            utils.assign_model_params(model_copy - cur_m* m_params.mean(dim=0) + cur_prop* evec[:, k], model, False)
            output = model(data)
            cur_k += [F.cross_entropy(output, target).item()]
        plt.plot(cur_k, label=r'$\alpha$ = %.1f'%cur_m)
    ax = plt.gca()
    ax.tick_params(axis = 'x', which = 'major', labelsize = 12)
    ax.tick_params(axis = 'y', which = 'major', labelsize = 12)
    plt.grid(True)
    plt.legend(prop={'size': 15})
    indices = [0, 5, 10, 15, 20] # we only pick up several beta's on x-axis
    if i == 0:
        plt.xticks([0, 5, 10, 15, 20], [-0.3125, -0.15625, 0.0, 0.15625, 0.3125])
        plt.xticks(indices, [x_labels[i] for i in indices])
    else:
        plt.xticks([0, 5, 10, 15, 20], [-2.5, -1.25, 0.0, 1.25, 2.5])
        plt.xticks(indices, [x_labels[i] for i in indices])
    plt.title(labels[i], fontdict = titledict)
    plt.ylim(0.0008, 0.2)
    plt.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig('./illustrative_example/eigenvector-%d.jpg'%(k+1))
    plt.ylim(0.0008, 0.05)
    plt.title(labels[i], fontdict = titledict)
    plt.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig('./illustrative_example/eigenvector-%d_zoom.jpg'%(k+1))
    plt.close()

# [ Observation 2: subspaces spanned by eigenvectors ]
size = model_copy.numel()
plt.figure()
cs = ['b', 'g', 'r', 'c', 'm']
utils.assign_model_params(model_copy, model, False)
data_test, target_test = None, None
for data_test, target_test in all_train_loader1:
    data_test, target_test = data_test.to(device), target_test.to(device)
print('The size of whole training dataset is %d'%len(data_test))
output = model(data_test)
center_loss = F.cross_entropy(output, target_test)
'''
    lam: sigma^2
'''
for l, lam in enumerate([0.1, 0.2, 0.5, 1.0, 10]):
    loss_array1 = []
    torch.manual_seed(32)
    for i in range(3000):
        noise = torch.randn(size, 1).to(device)
        noise = noise / math.sqrt(size)
        cur_loss1 = []
        for a in range(0, 400, 50):
            step1 = evec[:, a:a+50].mm(evec[:, a:a+50].t().mm(noise)) # $V V^T \delta$
            utils.assign_model_params(model_copy + math.sqrt(lam)* step1.view(-1), model, False)
            output = model(data_test)
            loss1 = F.cross_entropy(output, target_test).item()
            cur_loss1 += [loss1]
        loss_array1 += [cur_loss1]
    loss_array1 = np.average(np.array(loss_array1), axis = 0)
    plt.plot(loss_array1, c=cs[l], marker='o', label=r'$\sigma^2$=%.2f'%lam)
plt.plot([center_loss for _ in range(len(loss_array1))], c='k', ls='dashed')
ax = plt.gca()
ax.tick_params(axis = 'x', which = 'major', labelsize = 12)
ax.tick_params(axis = 'y', which = 'major', labelsize = 12)
plt.grid(True)
plt.legend(prop={'size': 15})
plt.xlabel(r'$s$', fontdict=fontdict)
plt.ylabel(r'$h(\sigma, V_s)$',fontdict=fontdict)
plt.xticks(list(range(len(loss_array1))), [50*i for i in range(1, len(loss_array1) + 1)])
plt.tight_layout()
plt.savefig('./illustrative_example/subspace_sigma.jpg')
plt.ylim(0.02104, 0.0212)
plt.tight_layout()  
plt.savefig('./illustrative_example/subspace_sigma_zoom.jpg')
plt.close()