from functools import reduce
import itertools
import pdb
import math
import copy
import torch
from torch import nn
from torch.nn import functional as F
from torch import autograd
from torch.autograd import Variable
import utils

class CommonModel(torch.nn.Module):
    def __init__(self, mlp, topk):
        super().__init__()
        self.layers_E = nn.ModuleList([])
        self.layers_D = nn.ModuleList([])
        name = 'M'
        is_bias = False
        for l, layer in enumerate(mlp.layers):
            if not (isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d)):
                continue
            m_layer = layer.weight.view(layer.weight.size(0), -1)
            r_dim = m_layer.size(0)
            c_dim = m_layer.size(1)
            l_dim = topk
            E  = [torch.nn.Conv2d(             1, l_dim, (r_dim,     1), groups=    1, bias=is_bias)]
            E += [torch.nn.Conv2d(         l_dim, l_dim, (    1, c_dim), groups=l_dim, bias=is_bias)]
            D  = [torch.nn.ConvTranspose2d(l_dim, l_dim, (    1, c_dim), groups=l_dim, bias=is_bias)]
            D += [torch.nn.ConvTranspose2d(l_dim,     1, (r_dim,     1), groups=    1, bias=is_bias)]
            E = torch.nn.Sequential(*E)
            D = torch.nn.Sequential(*D)
            E[0].weight = D[1].weight
            E[1].weight = D[0].weight
            self.layers_E.append(E)
            self.layers_D.append(D)
            for i, p in enumerate(E):
                self.register_parameter('E%s-%d-%d'%(name, l, i), p.weight)
            for i, p in enumerate(D):
                self.register_parameter('D%s-%d-%d'%(name, l, i), p.weight)

class TinyModel(torch.nn.Module):
    def __init__(self, mlp, mod_ae, topk):
        super().__init__()
        self.ae_params = []
        for i, (n, p) in enumerate(mod_ae.named_parameters(), 1):
            self.ae_params += [p.data.detach().view(p.data.size(0), -1)]
        self.layers_E = nn.ModuleList([])
        self.layers_D = nn.ModuleList([])
        name = 'M'
        is_bias = False
        for l, layer in enumerate(mlp.layers):
            if not (isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d)):
                continue
            m_layer = layer.weight.view(layer.weight.size(0), -1)
            r_dim = m_layer.size(0)
            c_dim = m_layer.size(1)
            l_dim = topk
            E  = [torch.nn.Conv2d(             1, l_dim, (r_dim,     1), groups=    1, bias=is_bias)]
            E += [torch.nn.Conv2d(         l_dim, l_dim, (    1, c_dim), groups=l_dim, bias=is_bias)]
            D  = [torch.nn.ConvTranspose2d(l_dim, l_dim, (    1, c_dim), groups=l_dim, bias=is_bias)]
            D += [torch.nn.ConvTranspose2d(l_dim,     1, (r_dim,     1), groups=    1, bias=is_bias)]
            E = torch.nn.Sequential(*E)
            D = torch.nn.Sequential(*D)
            E[0].weight = D[1].weight
            E[1].weight = D[0].weight
            self.layers_E.append(E)
            self.layers_D.append(D)
            for i, p in enumerate(E):
                self.register_parameter('E%s-%d-%d'%(name, l, i), p.weight)
            for i, p in enumerate(D):
                self.register_parameter('D%s-%d-%d'%(name, l, i), p.weight)

    def train(self):
        mid_out = 0
        datas = []
        for i in range(len(self.ae_params) // 2):
            left   = self.ae_params[2* i+0].detach().unsqueeze(2)
            righ   = self.ae_params[2* i+1].detach().unsqueeze(1)
            data   = left.bmm(righ).unsqueeze(1)
            datas += [data]
            mid_out += self.layers_E[i](data)
        loss = 0
        for i in range(len(self.ae_params) // 2):
            out    = self.layers_D[i](mid_out)
            loss  += F.mse_loss(out, datas[i].detach(), reduction='sum')
        return loss

class MLP(nn.Module):
    def __init__(self, args, input_size, output_size, hidden_size=400, hidden_layer_num=2, is_bias=False):
        super().__init__() 
        """ Fully-connected neural network

        References:
          https://github.com/kuc2477/pytorch-ewc

        Warning:
          there is a critical modification towards the original implementation 
          should never do thing like [nn.Linear* 10] as these linear layers are going to be identical
        """
        hidden_layers = [[nn.Linear(hidden_size, hidden_size, bias=is_bias), nn.ReLU()] for _ in range(hidden_layer_num)]
        hidden_layers = itertools.chain.from_iterable(hidden_layers)
        self.layers = nn.ModuleList([
            nn.Linear(input_size, hidden_size, bias=is_bias), nn.ReLU(),
            *hidden_layers,
            nn.Linear(hidden_size, output_size, bias=is_bias)
        ])

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return reduce(lambda x, l: l(x), self.layers, x)

    def pull2point(self, point, pull_strength=0.1):
        assert pull_strength ** 2 < 1
        for p1, p2 in zip(self.parameters(), point):
            diff = p1 - p2
            p1.data.add_(-pull_strength, diff.data)

class Conv(nn.Module):
    def __init__(self, args, is_bias = False):
        super().__init__()
        """ Convolutional neural network

        References:
          RWALK paper -- Page 19
        """
        conv_net = []
        conv_net += [nn.Conv2d( 3, 32, 3, bias=is_bias), nn.ReLU(True)]
        conv_net += [nn.Conv2d(32, 32, 3, bias=is_bias), nn.ReLU(True)]
        conv_net += [nn.MaxPool2d(kernel_size = (2, 2)), nn.Dropout(0.5)]

        conv_net += [nn.Conv2d(32, 64, 3, bias=is_bias), nn.ReLU(True)]
        conv_net += [nn.Conv2d(64, 64, 3, bias=is_bias), nn.ReLU(True)]
        conv_net += [nn.MaxPool2d(kernel_size = (2, 2)), nn.Dropout(0.5)]
        
        conv_net += [View()]
        conv_net += [nn.Linear(1600, 100, bias=is_bias)]
        self.layers = nn.ModuleList(conv_net)

    def forward(self, x):
        return reduce(lambda x, l: l(x), self.layers, x)

    def pull2point(self, point, pull_strength=0.1):
        assert pull_strength ** 2 < 1
        for p1, p2 in zip(self.parameters(), point):
            diff = p1 - p2
            p1.data.add_(-pull_strength, diff.data)


class InvAuto(nn.Module):
    """ A linear autoencoder used for finding the prohibited directions of a given model's parameters
        Args:
            args: parsed arguments defined in options.py
            mlp : a pointer to the model being analysed
            topk: an integer defining the number of top directions of the optimizer's trajectory we would like to identify
            is_invauto: a flag indicating whether the encoder and decoder are transposed of each other
            is_svd: (TODO left to be a future work)
            is_bias: a flag indicating whether the biases of the autoencoder are turned on
    """
    def __init__(self, args, mlp, topk, is_invauto=True, is_svd=False, is_bias=False, prev_invauto = None):
        super().__init__()
        """ 
            initialize autoencoder: 
                for details please refer to section 4.2 of our paper.
            comments:
                (1) notice that (U^TMV)[i,i] = u^T_i M v_i (M has 2 dimensions),
                    which is similar to convolving M by using u_i and v_i as the filters
                (2) encoder and decoder share the same set of parameters
        """
        self.args = args
        self.saved_mlps = {'grad1':{}, 'grad2':{}}
        self.saved_mlp_structure = []
        self.mlp_counter = 0
        for layer in mlp.layers:
            if not (isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d)):
                continue
            self.saved_mlp_structure.append(layer)
        self.layers_E = {}
        self.layers_D = {}
        self.prev_invauto = prev_invauto
        if prev_invauto is not None:
            self.scalars = []
            for i in range(len(prev_invauto)):
                dim = args.ae_topk // (i+1)
                scalars = torch.randn(dim, dim)
                torch.nn.init.orthogonal_(scalars)
                identity = torch.ones(dim)
                identity = torch.diag(identity)
                # self.scalars += [torch.nn.Parameter(scalars+identity)]
                self.scalars += [torch.nn.Parameter(identity)]
                # self.scalars += [torch.nn.Parameter(scalars)]
            self.scalars = torch.nn.ParameterList(self.scalars)
        for name in args.ae_what:
            print('It has', name)
            self.layers_E[name] = nn.ModuleList([])
            self.layers_D[name] = nn.ModuleList([])
            for l, layer in enumerate(mlp.layers):
                if not (isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d)):
                    continue
                m_layer = layer.weight.view(layer.weight.size(0), -1)
                r_dim = m_layer.size(0)
                c_dim = m_layer.size(1)
                l_dim = topk
                E  = [torch.nn.Conv2d(             1, l_dim, (r_dim,     1), groups=    1, bias=is_bias)]
                E += [torch.nn.Conv2d(         l_dim, l_dim, (    1, c_dim), groups=l_dim, bias=is_bias)]
                D  = [torch.nn.ConvTranspose2d(l_dim, l_dim, (    1, c_dim), groups=l_dim, bias=is_bias)]
                D += [torch.nn.ConvTranspose2d(l_dim,     1, (r_dim,     1), groups=    1, bias=is_bias)]
                E = torch.nn.Sequential(*E)
                D = torch.nn.Sequential(*D)
                E[0].weight = D[1].weight
                E[1].weight = D[0].weight
                self.layers_E[name].append(E)
                self.layers_D[name].append(D)
                for i, p in enumerate(E):
                    self.register_parameter('E%s-%d-%d'%(name, l, i), p.weight)
                for i, p in enumerate(D):
                    self.register_parameter('D%s-%d-%d'%(name, l, i), p.weight)
        # for name, param in self.named_parameters():
        #     print(name, param.size())
        for l in range(len(self.saved_mlp_structure)):
            self.saved_mlps['grad1'][l] = []
            self.saved_mlps['grad2'][l] = []
        self.mlp_num_params = sum([l_mlp.weight.numel() for l_mlp in self.saved_mlp_structure])

    def param_normalize(self):
        counter = 0
        layer_sqaure = 0
        for n, p in self.named_parameters():
            if not 'scalars' in n:
                counter += 1
                if counter % 2 == 1:
                    ls1 = (p.view(p.size(0), -1)**2).sum(dim=1)
                if counter % 2 == 0:
                    ls2 = (p.view(p.size(0), -1)**2).sum(dim=1)
                    layer_sqaure += ls1* ls2
        print(layer_sqaure.shape, layer_sqaure.max(), layer_sqaure.min())
        for n, p in self.named_parameters():
            if not 'scalars' in n:
                p.data = p.data / torch.sqrt(layer_sqaure.detach().sqrt()).view(-1,1,1,1)

    def ae_re_grad(self, postfix=''):
        """ find the top prohibited directions of the optimzer's trajectory (method 1).
            train linear autoencoder by forcing it to reconstructing the gradients
            (the model parameters are updated **periodically** while sampling the gradients)
        """
        my_device = self._my_device()
        batch_size = 16
        sum_l2_losses, sum_l2_losses_prev, sum_prop_losses = 0, 0, 0
        for cur in range(0, self.mlp_counter, batch_size):
            if self.prev_invauto is not None:
                prev_mid_out = [0 for i in range(len(self.prev_invauto))]
            inps, outs = [], []
            l2_loss, l2_loss_prev, l2_norm = 0, 0, 0

            # Encoder
            for l, _ in enumerate(self.saved_mlp_structure):
                l_mlp1 = self.saved_mlps['grad1'][l][cur: cur+batch_size].unsqueeze(1)          # channel = 1
                l_grd1 = self.saved_mlps['grad2'][l][cur: cur+batch_size].unsqueeze(1) - l_mlp1 # channel = 1
                inp    = torch.cat((l_mlp1, l_grd1), dim = 0).to(my_device)
                inps  += [inp]

            if self.prev_invauto is not None:
                for l, l_mlp in enumerate(self.saved_mlp_structure):
                    for i in range(len(self.prev_invauto)):
                        prev_mid_out[i] += self.prev_invauto[i].layers_E['M'][l](inps[l])
                for i in range(len(self.prev_invauto)):
                    shape = prev_mid_out[i].shape
                    prev_mid_out[i] = prev_mid_out[i].view(-1, len(self.scalars[i])).mm(self.scalars[i].mm(self.scalars[i].t()))
                    prev_mid_out[i] = prev_mid_out[i].view(shape)

            # Decoder
            mid_out = 0
            for l, l_mlp in enumerate(self.saved_mlp_structure):
                mid_out += self.layers_E['M'][l](inps[l])
            for l, l_mlp in enumerate(self.saved_mlp_structure):
                out      = self.layers_D['M'][l](mid_out)
                if self.prev_invauto is not None:
                    for i in range(len(self.prev_invauto)):
                        out += self.prev_invauto[i].layers_D['M'][l](prev_mid_out[i])
                l2_loss      += F.mse_loss(    out, inps[l], reduction='none').sum(dim=-1).sum(dim=-1)
                l2_norm      += F.mse_loss(inps[l], torch.zeros_like(inps[l]), reduction='none').sum(dim=-1).sum(dim=-1)
                sum_l2_losses        += self.args.ae_re_lam* l2_loss.sum()
                sum_prop_losses      += self.args.ae_re_lam* l2_norm.sum()
        # print(sum_l2_losses, sum_prop_losses)
        return sum_l2_losses, sum_l2_losses / (2* self.mlp_counter) 

    def ae_re_grad_diff_u(self, postfix=''):
        """ find the top prohibited directions of the optimzer's trajectory (method 2).
            train linear autoencoder by forcing it to reconstructing the differences between two consequent gradients
            (the model parameters are **fixed** while sampling the consequent gradients)
        """
        my_device = self._my_device()
        batch_size = 16
        sum_l2_losses, sum_prop_losses = 0, 0
        for cur in range(0, self.mlp_counter, batch_size):
            mid_out = 0
            inps, outs = [], []
            l2_loss, l2_norm = 0, 0
            # Encoder
            for l, _ in enumerate(self.saved_mlp_structure):
                l_mlp1 = self.saved_mlps['grad1'][l][cur: cur+batch_size].unsqueeze(1)          # channel = 1
                l_grd1 = self.saved_mlps['grad2'][l][cur: cur+batch_size].unsqueeze(1) - l_mlp1 # channel = 1
                inp = (l_grd1 - l_mlp1).to(my_device)
                out = inp
                inps += [inp]
                outs += [out]
                mid_out += self.layers_E['F'][l](inp)
            # Decoder
            for l, l_mlp in enumerate(self.saved_mlp_structure):
                out = self.layers_D['F'][l](mid_out)
                l2_loss += F.mse_loss(out, inps[l], reduction='none').sum(dim=-1).sum(dim=-1)
                l2_norm += inps[l].norm(dim=-1).norm(dim=-1)
            sum_l2_losses   += self.args.ae_re_lam* l2_loss.sum()
            sum_prop_losses += (torch.sqrt(l2_loss)/l2_norm).sum()
        return sum_prop_losses/self.mlp_counter, sum_l2_losses/self.mlp_counter        

    def ae_re_grad_diff_f(self, postfix=''):
        """ find the top prohibited directions of the optimzer's trajectory (method 3).
            train linear autoencoder by forcing it to reconstructing the differences between two consequent gradients
            (the model parameters are **updated** when sampling the consequent gradients)
        """
        my_device = self._my_device()
        batch_size = 16
        sum_l2_losses, sum_prop_losses = 0, 0
        for cur in range(1, self.mlp_counter - batch_size, batch_size):
            mid_out = 0
            inps, outs = [], []
            l2_loss, l2_norm = 0, 0
            # encoder
            for l, _ in enumerate(self.saved_mlp_structure):
                l_mlp2 = self.saved_mlps['grad2'][l][cur-1: cur+batch_size-1].unsqueeze(1)
                l_grd2 = self.saved_mlps['grad2'][l][cur  : cur+batch_size  ].unsqueeze(1)
                inp = l_mlp2.to(my_device)
                out = ((l_mlp2 - l_grd2)/ self.args.main_online_lr).to(my_device)
                inps += [inp]
                outs += [out]
                mid_out += self.layers_E['H'][l](inp)
            # decoder
            for l, l_mlp in enumerate(self.saved_mlp_structure):
                out = self.layers_D['H'][l](mid_out)
                l2_loss += F.mse_loss(out, inps[l], reduction='none').sum(dim=-1).sum(dim=-1)
                l2_norm += inps[l].norm(dim=-1).norm(dim=-1)
            sum_l2_losses   += self.args.ae_re_lam* l2_loss.sum()
            sum_prop_losses += (torch.sqrt(l2_loss)/l2_norm).sum()
        return sum_prop_losses/(self.mlp_counter-batch_size), sum_l2_losses/(self.mlp_counter-batch_size)

    def ae_learn_offline(self, postfix=''):
        """ find the top prohibited directions by choosing and combining following methods: 
            (1) 'F': ae_re_grad --> reconstruct the gradients while updating the model parameters periodically
            (2) 'H': ae_re_grad_diff_u --> reconstruct the differences between two consequent gradients while fixing the model parameters
            (3) 'M': ae_re_grad_diff_f --> reconstruct the differences between two consequent gradients while updating the model parameters
        """
        l1, l2 = 0, 0
        if 'F' in self.args.ae_what:
            a, b = self.ae_re_grad_diff_u(postfix)
            l1 = l1 + a
            l2 = l2 + b
        if 'H' in self.args.ae_what:
            a, b = self.ae_re_grad_diff_f(postfix)
            l1 = l1 + a
            l2 = l2 + b
        if 'M' in self.args.ae_what:
            a, b = self.ae_re_grad(postfix)
            l1 = l1 + a
            l2 = l2 + b

        self.saved_mlps = {'grad1':{}, 'grad2':{}}
        for l in range(len(self.saved_mlp_structure)):
            self.saved_mlps['grad1'][l] = []
            self.saved_mlps['grad2'][l] = []
        return l1, l2

    def ae_snap_mlp(self, one_two):
        """ save current gradient """
        for l, l_mlp in enumerate(self.saved_mlp_structure):
            if one_two == 1:
                self.saved_mlps['grad1'][l].append(l_mlp.weight.grad.view(l_mlp.weight.grad.size(0), -1).data.to('cpu'))
            else:
                self.saved_mlps['grad2'][l].append(l_mlp.weight.grad.view(l_mlp.weight.grad.size(0), -1).data.to('cpu'))

    def ae_save_mlps(self, postfix=''):
        """ normalize the gradients and save the normalized ones  """
        for l, _ in enumerate(self.saved_mlp_structure):
            self.saved_mlps['grad1'][l] = torch.stack(self.saved_mlps['grad1'][l])
            self.saved_mlps['grad2'][l] = torch.stack(self.saved_mlps['grad2'][l])

        saved_mlps_var = 0
        for l1, l2 in zip(self.saved_mlps['grad1'].values(), self.saved_mlps['grad2'].values()):
            saved_mlps_var += l1.norm().item()** 2
            saved_mlps_var += (l2-l1).norm().item()** 2
        saved_mlps_var = math.sqrt(saved_mlps_var / (2* self.mlp_counter))
        for l, _ in enumerate(self.saved_mlp_structure):
            self.saved_mlps['grad1'][l] = self.saved_mlps['grad1'][l]/ saved_mlps_var
            self.saved_mlps['grad2'][l] = self.saved_mlps['grad2'][l]/ saved_mlps_var
        if self.args.is_mlps_saved:
            state = {'saved_mlps': self.saved_mlps}
            torch.save(state, 'cl_mlp_state_dict-%d-%s.pth.tar'%(self.args.rank, postfix))

    def ae_encode(self, mlp, mlp_center):
        """ encoding operation of the encoder """
        for name in self.args.ae_what:
            mid_out = 0
            if self.prev_invauto is not None:
                prev_mid_out = [0 for i in range(len(self.prev_invauto))]
            for l, l_mlp in enumerate(self.saved_mlp_structure):
                inp = l_mlp.weight.view(l_mlp.weight.size(0), -1) - mlp_center[l].view(l_mlp.weight.size(0), -1)
                inp = inp.view(1, 1, inp.size(0), inp.size(1))
                mid_out      += self.layers_E[name][l](inp)
                if self.prev_invauto is not None:
                    for i in range(len(self.prev_invauto)):
                        prev_mid_out[i] += self.prev_invauto[i].layers_E[name][l](inp)
            mid_out = mid_out.squeeze()
            if self.prev_invauto is not None:
                for i in range(len(self.prev_invauto)):
                    prev_mid_out[i] = prev_mid_out[i].view(-1, len(self.scalars[i])).mm(self.scalars[i]).squeeze()
                all_encodings = torch.cat(prev_mid_out + [mid_out])
            else:
                all_encodings = mid_out
        # all_encodings = mid_out
        return all_encodings

    def ae_grad_encode(self, mlp, mlp_center):
        """ encoding operation of the encoder """
        for name in self.args.ae_what:
            mid_out = 0
            if self.prev_invauto is not None:
                prev_mid_out = [0 for i in range(len(self.prev_invauto))]
            for l, l_mlp in enumerate(self.saved_mlp_structure):
                inp = l_mlp.weight.grad.data.view(l_mlp.weight.size(0), -1)
                inp = inp.view(1, 1, inp.size(0), inp.size(1))
                mid_out      += self.layers_E[name][l](inp)
                if self.prev_invauto is not None:
                    for i in range(len(self.prev_invauto)):
                        prev_mid_out[i] += self.prev_invauto[i].layers_E[name][l](inp)
            mid_out = mid_out.squeeze()
            if self.prev_invauto is not None:
                for i in range(len(self.prev_invauto)):
                    prev_mid_out[i] = prev_mid_out[i].view(-1, len(self.scalars[i])).mm(self.scalars[i]).squeeze()
                all_encodings = torch.cat(prev_mid_out + [mid_out])
            else:
                all_encodings = mid_out
        return all_encodings
 
    def _my_device(self):
        return next(self.parameters()).device

class View(nn.Module):
    """ similar to torch.flatten() but this is compatible with PyTorch <= 1.0 """
    def __init__(self):
        super(View, self).__init__()
    def forward(self, x):
        return x.view(x.size(0), -1)