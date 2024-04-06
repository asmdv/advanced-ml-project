import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.dataloader import default_collate

AVAILABLE_DATASETS = {
    'permuted_mnist': datasets.MNIST,
    'split_mnist': datasets.MNIST,
    # 'split_cifar10': datasets.CIFAR10,
    'split_cifar100': datasets.CIFAR100,
    'mnist': datasets.MNIST,
    'fashion_mnist': datasets.FashionMNIST,
}

DATASET_CONFIGS = {
    'permuted_mnist': {'size': 28, 'channels': 1, 'classes': 10},
    'split_mnist': {'size': 28, 'channels': 1, 'classes': 10},
    # 'split_cifar10': {'size': 32, 'channels': 3, 'classes': 10},
    'split_cifar100': {'size': 32, 'channels': 3, 'classes': 100},
    'mnist': {'size': 28, 'channels': 1, 'classes': 10},
    'fashion_mnist': {'size': 28, 'channels': 1, 'classes': 10},
}

AVAILABLE_TRANSFORMS = {
    'permuted_mnist': [ # https://discuss.pytorch.org/t/normalization-in-the-mnist-example/457
        transforms.Resize(DATASET_CONFIGS['permuted_mnist']['size']),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ],
    'split_mnist': [ # https://discuss.pytorch.org/t/normalization-in-the-mnist-example/457
        transforms.Resize(DATASET_CONFIGS['split_mnist']['size']),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ],
    # 'split_cifar10': [ # https://gist.github.com/weiaicunzai/e623931921efefd4c331622c344d8151
    #     transforms.Resize(DATASET_CONFIGS['split_cifar10']['size']),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # ],
    'split_cifar100': [ # https://gist.github.com/weiaicunzai/e623931921efefd4c331622c344d8151
        transforms.Resize(DATASET_CONFIGS['split_cifar100']['size']),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ],
    'mnist': [
        transforms.Resize(DATASET_CONFIGS['mnist']['size']),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ],
    'fashion_mnist': [
        transforms.Resize(DATASET_CONFIGS['fashion_mnist']['size']),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ],
}

MODEL_CONFIGS = {
    'permuted_mnist': {'n_layer': 2, 's_layer': 256, 'in_drop': .0, 'out_drop': .0},
    'split_mnist': {'n_layer': 2, 's_layer': 100, 'in_drop': .0, 'out_drop': .0},
    'fashion_mnist': {'n_layer': 2, 's_layer': 256, 'in_drop': .0, 'out_drop': .0},
}

class map_dataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    def __getitem__(self,index):
        data, target = self.dataset[index]
        return data, target
    def __len__(self):
        return len(self.dataset)

def get_dataset(name, m_task = 0, train=True, download=True):
    dataset = AVAILABLE_DATASETS[name]
    transform = AVAILABLE_TRANSFORMS[name]
    if name == 'permuted_mnist':
        np.random.seed(m_task)
        length = DATASET_CONFIGS[name]['size']**2
        channel = DATASET_CONFIGS[name]['channels']
        m_permutation = np.random.permutation(length* channel)
        trans_permutation = transforms.Lambda(lambda x: _permutate_image_pixels(x, m_permutation))
        transform = transforms.Compose(transform + [trans_permutation])
    else:
        transform = transforms.Compose(transform)
    return dataset('./datasets/{name}'.format(name=name), train=train,
                   download=download, transform=transform)

def get_data_loader(dataset, batch_size, cuda=False, collate_fn=None):
    return DataLoader(
        dataset, batch_size=batch_size,
        shuffle=True, collate_fn=(collate_fn or default_collate),
        **({'num_workers': 4, 'pin_memory': True} if cuda else {})
    )

# TODO: Temporary
def get_label_data_loader(dataset, batch_size, cuda=False, collate_fn=None, labels = []):
    ''' define training and testing data loader '''
    if len(labels) == 0:
        raise ValueError('labels are empty')    
    # TODO: FIXME data targets
    # indices = [i for i in range(len(dataset.targets))
    #                 if dataset.targets[i].item() in labels]
    try:
        indices = [i for i in range(len(dataset.train_labels))
                        if dataset.train_labels[i] in labels]
    except:
        indices = [i for i in range(len(dataset.test_labels))
                        if dataset.test_labels[i] in labels]
    my_dataset = Subset(dataset, indices)
    return get_data_loader(my_dataset, batch_size, cuda=cuda, collate_fn=collate_fn)

def _permutate_image_pixels(image, m_permutation, dataset = 'permuted_mnist'):
    size = image.size()
    image = image.view(-1)[m_permutation]
    return image.view(size)