import torch
import torch.nn.functional as F
import pdb

def train(args, mod_main, opt_main, data, target, is_grad_acc = False):
    '''
        is_grad_acc:    by default we clean up the gradients of classifer
                        but we hope to accumulate the gradients for our method
        masked outputs: the multi-head setup is similar to the codes provided in SI GitHub
                        permuted_mnist: do not need it!
                        split_mnist:    5 tasks with two labels per task
                        split_cifar100: 10 tasks with 10 labels per task
                        Notice that the reason of iterating over a mini-batch data point by data point is that
                        for replay method the target labels are a mixture of all previous tasks so we can't just mask the outputs for a specific task
        TODO: check the influence of dividing the loss by target.size(0) for split_cifar100
    '''
    mod_main.train()
    if not is_grad_acc:
        opt_main.zero_grad()
    if args.cl_dataset == 'permuted_mnist':
        data, target = data.to(args.device), target.to(args.device)
        output = mod_main(data)
        main_loss = F.cross_entropy(output, target)
    elif args.cl_dataset == 'split_mnist' or args.cl_dataset == 'split_cifar10':
        main_loss = 0
        data, target = data.to(args.device), target.to(args.device)
        task_target = target % 2
        output = mod_main(data)
        for i in range(target.size(0)):
            output_id = 2* (target[i].item()//2)
            main_loss += F.cross_entropy(output[i:i+1, output_id:output_id+2], task_target[i].view(-1), reduction='sum')
        main_loss = main_loss / target.size(0)
    elif args.cl_dataset == 'split_cifar100':
        main_loss = 0
        data, target = data.to(args.device), target.to(args.device)
        task_target = target % 10
        output = mod_main(data)
        if 'gem' in args.cl_method:
            for i in range(target.size(0)):
                output_id = 10* (target[i].item()//10)
                main_loss += F.cross_entropy(output[i:i+1, output_id:output_id+10], task_target[i].view(-1), reduction='sum')
        else:
            output_id = 10* (target[0].item()//10)
            main_loss += F.cross_entropy(output[:, output_id:output_id+10], task_target, reduction='sum')
        main_loss = main_loss / target.size(0)
    elif args.cl_dataset == 'from_fashion_mnist' or args.cl_dataset == 'to_fashion_mnist':
        data, target = data.to(args.device), target.to(args.device)
        output = mod_main(data)
        main_loss = F.cross_entropy(output, target)
    return main_loss

def test(args, model, test_loader, epoch, prefix=''):
    '''
        Be careful about the multi-head setup for both loss and error
    '''
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            if args.cl_dataset == 'permuted_mnist':
                data, target = data.to(args.device), target.to(args.device)
                output = model(data)
                test_loss += F.cross_entropy(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
            elif args.cl_dataset == 'split_mnist' or args.cl_dataset == 'split_cifar10':
                data, target = data.to(args.device), target.to(args.device)
                task_target = target % 2
                output = model(data)
                for i in range(target.size(0)):
                    output_id = 2* (target[i].item()//2)
                    test_loss += F.cross_entropy(output[i:i+1, output_id:output_id+2], task_target[i].view(-1), reduction='sum').item()
                    pred = output[i:i+1, output_id:output_id+2].argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                    correct += pred.eq(task_target[i].view_as(pred)).sum().item()
            elif args.cl_dataset == 'split_cifar100':
                data, target = data.to(args.device), target.to(args.device)
                # print(target)
                task_target = target % 10
                output = model(data)
                if 'gem' in args.cl_method:
                    for i in range(target.size(0)):
                        output_id = 10* (target[i].item()//10)
                        test_loss += F.cross_entropy(output[i:i+1, output_id:output_id+10], task_target[i].view(-1), reduction='sum').item()
                        pred = output[i:i+1, output_id:output_id+10].argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                        correct += pred.eq(task_target[i].view_as(pred)).sum().item()
                else:
                    output_id = 10* (target[0].item() // 10)
                    test_loss += F.cross_entropy(output[:, output_id:output_id+10], task_target, reduction='sum').item()
                    pred = output[:, output_id:output_id+10].argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                    correct += pred.eq(task_target.view_as(pred)).sum().item()
            elif args.cl_dataset == 'from_fashion_mnist' or args.cl_dataset == 'to_fashion_mnist':
                data, target = data.to(args.device), target.to(args.device)
                output = model(data)
                test_loss += F.cross_entropy(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    res='[{:2d}] Average TEST Loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
                 epoch, test_loss, correct, len(test_loader.dataset),100. * correct / len(test_loader.dataset))
    print(prefix + res)
    return 100. * (1-correct / len(test_loader.dataset))

def ae_reg(args, mod_main, mod_main_center, opt_main, mod_ae, opt_ae, data, target):
    '''
        ae_loss: encoding loss
    '''
    opt_ae.zero_grad()
    ae_mid_out = mod_ae.ae_encode(mod_main, mod_main_center)
    if args.ae_mean:
        ae_loss = F.mse_loss(ae_mid_out, torch.zeros_like(ae_mid_out), reduction='mean')* args.ae_cl_lam
        # ae_loss = F.l1_loss(ae_mid_out, torch.zeros_like(ae_mid_out), reduction='mean')* args.ae_cl_lam
    else:
        ae_loss = F.mse_loss(ae_mid_out, torch.zeros_like(ae_mid_out), reduction='sum')* args.ae_cl_lam
        # ae_loss = F.l1_loss(ae_mid_out, torch.zeros_like(ae_mid_out), reduction='sum')* args.ae_cl_lam
    opt_ae.zero_grad()
    return ae_loss

def ae_test(args, mod_main, mod_ae, te_loader, epoch, prefix=''):
    with torch.no_grad():
        mod_ae.mlp_low_rank(mod_main)
        test(args, mod_main, te_loader, epoch, prefix)
        mod_ae.mlp_full_rank(mod_main)