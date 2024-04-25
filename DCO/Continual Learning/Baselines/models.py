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

class MLP(nn.Module):
    def __init__(self, args, input_size, output_size, hidden_size=400, hidden_layer_num=2, is_bias=False, num_tasks=5):
        super().__init__()
        """ Fully-connected neural network

        References:
          https://github.com/kuc2477/pytorch-ewc

        Warning:
          there is a critical modification towards the original implementation 
          should never do thing like [nn.Linear* 10] as these linear layers are going to be identical
        """
        # hidden_layers = [[nn.Linear(hidden_size, hidden_size, bias=is_bias), nn.ReLU()] for _ in range(hidden_layer_num)]
        # hidden_layers = itertools.chain.from_iterable(hidden_layers)
        # self.layers = nn.ModuleList([
        #     nn.Linear(input_size, hidden_size, bias=is_bias), nn.ReLU(),
        #     *hidden_layers,
        #     nn.Linear(hidden_size, output_size, bias=is_bias)
        # ])

        self.input_layer = nn.Linear(input_size, hidden_size, bias=is_bias)
        self.input_activation = nn.ReLU()

        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size, bias=is_bias) for _ in range(hidden_layer_num)
        ])
        self.hidden_activations = nn.ModuleList([
            nn.ReLU() for _ in range(hidden_layer_num)
        ])

        self.output_layers = nn.ModuleList([
            nn.Linear(hidden_size, output_size, bias=is_bias),
            nn.Linear(hidden_size, output_size, bias=is_bias),
            nn.Linear(hidden_size, output_size, bias=is_bias),
            nn.Linear(hidden_size, output_size, bias=is_bias),
            nn.Linear(hidden_size, output_size, bias=is_bias)
        ])

        self.tasks = [hidden_layer_num - 1] * num_tasks
        self.tasks_output = [0] * num_tasks

    def forward(self, x, task):
        # print(f"Forward task {task}")
        x = x.view(x.size(0), -1)
        x = self.input_activation(self.input_layer(x))
        # print(f"Layers executed for Task {task}: ", end="")
        for layer_i, (hidden_layer, hidden_activation) in enumerate(zip(self.hidden_layers, self.hidden_activations)):
            if task != None and layer_i == self.tasks[task] + 1:
                break
            # print(f"{layer_i} ", end="")
            x = hidden_activation(hidden_layer(x))
        # print()
        return self.output_layers[self.tasks_output[task]](x)

    # def forward(self, x):
    #     x = x.view(x.size(0), -1)
    #     return reduce(lambda x, l: l(x), self.layers, x)

    def pull2point(self, point, pull_strength=0.1):
        assert pull_strength ** 2 < 1
        for p1, p2 in zip(self.parameters(), point):
            diff = p1 - p2
            p1.data.add_(-pull_strength, diff.data)

    def print_layers(self):
        # Print all layers in the model
        print("Layers in the model:")
        for name, layer in self.named_modules():
            if name != "":
                print(f"{name}: {layer}")

    def print_grad_req_for_all_params(self):
        print("Printing all params")
        for name, param in self.named_parameters():
            print(name, param.requires_grad)
    def freeze_all_but_last(model, task):
        t = tuple(model.named_parameters())
        for name, param in t:
            if "output" in name:
                if int(name.split(".")[1]) == task-1:
                    param.requires_grad = False
            else:
                param.requires_grad = False
        print(f"All layers except the outputs frozen successfully.")

    def add_hidden_layerV3(self, new_layer_size, task, count=1, same=False):
        # if layer_index < 0 or layer_index >= len(self.layers):
        #     raise ValueError("Invalid layer_index")
        if count < 1:
            raise ValueError("Count should be at least 1")

        prev_out_features = self.hidden_layers[-1].out_features

        for i in range(task, len(self.tasks)):
            self.tasks[i] += 1

        if count == 1:
            # output_layer = nn.Linear(prev_out_features, self.output_layers[-1].out_features, bias=False)
            # self.output_layers.append(output_layer)
            for i in range(task, len(self.tasks)):
                self.tasks_output[i] += 1
            pass

        elif count == 2:
            new_layer = nn.Linear(prev_out_features, new_layer_size, bias=False)
            self.hidden_layers.append(new_layer)
            self.hidden_activations.append(nn.ReLU())
            for i in range(task, len(self.tasks)):
                self.tasks[i] += 1

            # output_layer = nn.Linear(new_layer_size, self.output_layers[-1].out_features, bias=False)
            # self.output_layers.append(output_layer)
            for i in range(task, len(self.tasks)):
                self.tasks_output[i] += 1


        elif count > 2:
            before_layer = nn.Linear(prev_out_features, new_layer_size, bias=False)
            self.hidden_layers.append(before_layer)
            self.hidden_activations.append(nn.ReLU())
            for i in range(task, len(self.tasks)):
                self.tasks[i] += 1

            for i in range(count-2):
                new_layer = nn.Linear(new_layer_size, new_layer_size, bias=False)
                self.hidden_layers.append(new_layer)
                self.hidden_activations.append(nn.ReLU())
                for i in range(task, len(self.tasks)):
                    self.tasks[i] += 1

            # output_layer = nn.Linear(new_layer_size, self.output_layers[-1].out_features, bias=False)
            # self.output_layers.append(output_layer)
            for i in range(task, len(self.tasks)):
                self.tasks_output[i] += 1

        else:
            raise Exception("Invalid count format")

        # if not same:
        #     if count == 1:
        #         raise Exception(f"Cannot add only 1 layer with different hidden size: {new_layer_size}. Try at least 2.")
        #     new_layer_before = nn.Linear(prev_out_features, new_layer_size, bias=False)
        #     new_layer_after = nn.Linear(new_layer_size, next_in_features, bias=False)
        #     new_layer = nn.Linear(new_layer_size, new_layer_size, bias=False)
        #
        #     self.hidden_layers.insert(len(self.hidden_layers), new_layer_before)
        #     self.hidden_activations.insert(len(self.hidden_activations), nn.ReLU())
        #
        #     for i in range(count-2):
        #         self.hidden_layers.insert(len(self.hidden_layers), new_layer)
        #         self.hidden_activations.insert(len(self.hidden_activations), nn.ReLU())
        #
        #     self.hidden_layers.insert(len(self.hidden_layers), new_layer_after)
        #     self.hidden_activations.insert(len(self.hidden_activations), nn.ReLU())
        # else:
        #     for i in range(count-1):
        #         new_layer = nn.Linear(new_layer_size, new_layer_size, bias=False)
        #         self.hidden_layers.insert(len(self.hidden_layers), new_layer)
        #         self.hidden_activations.insert(len(self.hidden_activations), nn.ReLU())
        #     self.output_layer.append(nn.Linear)

    def add_hidden_layerV2(self, layer_index, new_layer_size, count=1, same=False):
        if layer_index < 0 or layer_index >= len(self.layers):
            raise ValueError("Invalid layer_index")
        if count < 1:
            raise ValueError("Count should be at least 1")

        prev_out_features = self.layers[layer_index].out_features
        next_in_features = self.layers[layer_index + 2].in_features
        print(f"New Hidden Layer is the same as others: {same}")
        if not same:
            new_layer_before = nn.Linear(prev_out_features, new_layer_size, bias=False)
            new_layer_after = nn.Linear(new_layer_size, next_in_features, bias=False)
            new_layer = nn.Linear(new_layer_size, new_layer_size, bias=False)

            self.layers.insert(layer_index + 2, new_layer_before)
            self.layers.insert(layer_index + 3, nn.ReLU())

            for i in range(count):
                self.layers.insert(layer_index + 4 + 2*i, new_layer)
                self.layers.insert(layer_index + 5 + 2*i, nn.ReLU())

            self.layers.insert(layer_index + 4 + 2*count, new_layer_after)
            self.layers.insert(layer_index + 5 + 2*count, nn.ReLU())
        else:
            for i in range(count):
                new_layer = nn.Linear(new_layer_size, new_layer_size, bias=False)
                self.layers.insert(layer_index + 2 + 2*i, new_layer)
                self.layers.insert(layer_index + 3 + 2*i, nn.ReLU())

        # print("Layers after insert: ", self.layers)
        # print("self.layers[layer_index + 3].in_features before", self.layers[layer_index + 3].in_features)
        # self.layers[layer_index + 4].in_features = new_layer_size
        # print("Layers after in featues change insert: ", self.layers)

        # if next_in_features != new_layer_size:
        #     print("next_in_features != new_layer_size")
        #     self.layers[layer_index + 4].in_features = new_layer_size
        # print("Layers final: ", self.layers)



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
    def __init__(self, args, mlp, topk, is_invauto=True, is_svd=False, is_bias=False):
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
        self.topk = args.ae_topk // len(args.ae_what)
        self.saved_mlps = {'grad1':{}, 'grad2':{}}
        self.saved_mlp_structure = []
        self.mlp_counter = 0
        for layer in mlp.layers:
            if not (isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d)):
                continue
            self.saved_mlp_structure.append(layer)
        self.layers_E = {}
        self.layers_D = {}
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
                l_dim = self.topk
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

    def ae_re_grad(self, postfix=''):
        """ find the top prohibited directions of the optimzer's trajectory (method 1).
            train linear autoencoder by forcing it to reconstructing the gradients
            (the model parameters are updated **periodically** while sampling the gradients)
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
                inp = torch.cat((l_mlp1, l_grd1), dim = 0).to(my_device)
                out = inp
                inps += [inp]
                outs += [out]
                mid_out += self.layers_E['M'][l](inp)
            # Decoder
            for l, l_mlp in enumerate(self.saved_mlp_structure):
                out = self.layers_D['M'][l](mid_out)
                l2_loss += F.mse_loss(out, inps[l], reduction='none').sum(dim=-1).sum(dim=-1)
                l2_norm += inps[l].norm(dim=-1).norm(dim=-1)
            sum_l2_losses   += self.args.ae_re_lam* l2_loss.sum()
            sum_prop_losses += (torch.sqrt(l2_loss)/l2_norm).sum()
        return sum_prop_losses/(2* self.mlp_counter), sum_l2_losses/(2* self.mlp_counter) 

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
        all_encodings = 0
        for name in self.args.ae_what:
            mid_out = 0
            for l, l_mlp in enumerate(self.saved_mlp_structure):
                inp = l_mlp.weight.view(l_mlp.weight.size(0), -1) - mlp_center[l].view(l_mlp.weight.size(0), -1)
                inp = inp.view(1, 1, inp.size(0), inp.size(1))
                out = self.layers_E[name][l](inp)
                mid_out = mid_out + self.layers_E[name][l](inp)
            all_encodings += torch.abs(mid_out)
        all_encodings = all_encodings / len(self.args.ae_what)
        return all_encodings
        
    def _my_device(self):
        return next(self.parameters()).device

class View(nn.Module):
    """ similar to torch.flatten() but this is compatible with PyTorch <= 1.0 """
    def __init__(self):
        super(View, self).__init__()
    def forward(self, x):
        return x.view(x.size(0), -1)