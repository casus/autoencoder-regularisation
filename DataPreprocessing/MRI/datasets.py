import torch
import torch.utils.data

import torchvision
import torchvision.transforms as transforms

#create fast data loader
class InMemDataLoader(object):
    __initialized = False
    def __init__(self, tensors, batch_size=1, shuffle=False, sampler=None,
                 batch_sampler=None, drop_last=False):
        """A torch dataloader that fetches data from memory."""
        self.dataset = tensors#[torch.tensor(tensor) for tensor in tensors]
        #dataset = torch.utils.data.TensorDataset(*tensors)
        #self.dataset = tensors
        self.batch_size = batch_size
        self.drop_last = drop_last
        
        if batch_sampler is not None:
            if batch_size > 1 or shuffle or sampler is not None or drop_last:
                raise ValueError('batch_sampler option is mutually exclusive '
                                 'with batch_size, shuffle, sampler, and '
                                 'drop_last')
            self.batch_size = None
            self.drop_last = None

        if sampler is not None and shuffle:
            raise ValueError('sampler option is mutually exclusive with '
                             'shuffle')
            
        if batch_sampler is None:
            if sampler is None:
                if shuffle:
                    sampler = torch.utils.data.RandomSampler(self.dataset)
                else:
                    sampler = torch.utils.data.SequentialSampler(self.dataset)
            batch_sampler = torch.utils.data.BatchSampler(sampler, batch_size, drop_last)

        self.sampler = sampler
        self.batch_sampler = batch_sampler
        self.__initialized = True
    
    def __setattr__(self, attr, val):
        if self.__initialized and attr in ('batch_size', 'sampler', 'drop_last'):
            raise ValueError('{} attribute should not be set after {} is '
                             'initialized'.format(attr, self.__class__.__name__))

        super(InMemDataLoader, self).__setattr__(attr, val)

    def __iter__(self):
        for batch_indices in self.batch_sampler:
            yield self.dataset[batch_indices]

    def __len__(self):
        return len(self.batch_sampler)
    
    def to(self, device):
        self.dataset.tensors = tuple(t.to(device) for t in self.dataset.tensors)
        return self


def getMNIST(batch_size = 32, drop_last=False):   
    mnist_data = torchvision.datasets.MNIST(download=False, root = 'data/mnist', transform = 
                                                                                 transforms.Compose([transforms.Resize(32),
                                                                                 transforms.ToTensor(), 
                                                                                 transforms.Lambda(lambda x: x.repeat(3, 1, 1))
                                                                                 ]))
    mnist_test_data = torchvision.datasets.MNIST(download=False, root = 'data/mnist', train=False, transform = 
                                                                                 transforms.Compose([transforms.Resize(32),
                                                                                 transforms.ToTensor(), 
                                                                                 transforms.Lambda(lambda x: x.repeat(3, 1, 1))
                                                                                 ]))


    train_loader = torch.utils.data.DataLoader(mnist_data,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=16,
                                              drop_last=drop_last)
    
    test_loader = torch.utils.data.DataLoader(mnist_test_data,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=16,
                                              drop_last=drop_last)

    return train_loader, test_loader


def getFashionMNIST(batch_size = 32, drop_last=False):
    fashionmnist_data = torchvision.datasets.FashionMNIST(download=True, root = 'data/fashionmnist', transform = 
                                                                                 transforms.Compose([transforms.Resize(32),
                                                                                 transforms.ToTensor(), 
                                                                                 transforms.Lambda(lambda x: x.repeat(3, 1, 1))
                                                                                 ]))

    fashionmnist_data_test = torchvision.datasets.FashionMNIST(download=True, root = 'data/fashionmnist', train=False, transform = 
                                                                                 transforms.Compose([transforms.Resize(32),
                                                                                 transforms.ToTensor(), 
                                                                                 transforms.Lambda(lambda x: x.repeat(3, 1, 1))
                                                                                 ]))

    train_loader = torch.utils.data.DataLoader(fashionmnist_data,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=16,
                                              drop_last=drop_last)

    test_loader = torch.utils.data.DataLoader(fashionmnist_data_test,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=16,
                                              drop_last=drop_last)

    return train_loader, test_loader


def getCifar10(batch_size = 32, drop_last=False):
    cifar10_data = torchvision.datasets.CIFAR10(download=True, root = 'data/cifar10', transform = transforms.Compose([
            transforms.ToTensor()]))

    cifar10_data_test = torchvision.datasets.CIFAR10(download=True, root = 'data/cifar10', train=False, transform = transforms.Compose([
            transforms.ToTensor()]))

    train_loader = torch.utils.data.DataLoader(cifar10_data,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=16,
                                              drop_last=drop_last)


    test_loader = torch.utils.data.DataLoader(cifar10_data_test,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=16,
                                              drop_last=drop_last)

    return train_loader, test_loader


def getDataset(dataset = "MNIST", batch_size = 32, drop_last=False):
    if(dataset == "MNIST"):
        train_loader, test_loader = getMNIST(batch_size, drop_last)
        noChannels,dx, dy = train_loader.dataset.__getitem__(1)[0].shape
    elif(dataset == "FashionMNIST"):
        train_loader, test_loader = getFashionMNIST(batch_size, drop_last)
        noChannels, dx, dy = train_loader.dataset.__getitem__(1)[0].shape
    elif(dataset == "Cifar10"):
        train_loader, test_loader = getCifar10(batch_size, drop_last)
        noChannels, dx, dy = train_loader.dataset.__getitem__(1)[0].shape
    else:
        return None, None, None, None, None    
        
    return train_loader, test_loader, noChannels, dx, dy