from torchvision.datasets import CIFAR10
from torchvision import transforms
import torch
from utilities.utils_datasets import CustomDataset


def load_binary_cifar10(low=1, high=7):
    transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465),
                                  (0.2023, 0.1994, 0.2010)),])
    train_set = CIFAR10(root="cifar10", download=True, train=True, transform=transform)
    train_set.data = torch.tensor(train_set.data)
    train_set.targets = torch.tensor(train_set.targets)
    train_labels = (train_set.targets == low) | (train_set.targets == high)
    train_set.data = train_set.data[train_labels]
    train_set.targets = (train_set.targets[train_labels] == low).to(int)
    
    assert train_set.data.shape[0] == train_set.targets.shape[0]

    test_set = CIFAR10(root="cifar10", download=True, train=False, transform=transform)
    test_set.data = torch.tensor(test_set.data)
    test_set.targets = torch.tensor(test_set.targets)
    test_labels = (test_set.targets == low) | (test_set.targets == high)
    test_set.data = test_set.data[test_labels]
    test_set.targets = (test_set.targets[test_labels] == low).to(int)

    assert test_set.data.shape[0] == test_set.targets.shape[0]

    train_dataset = CustomDataset(data=train_set.data, targets=train_set.targets, transform=transform, real_targets=False, is_an_image=True)
    test_dataset = CustomDataset(data=test_set.data, targets=test_set.targets, transform=transform, real_targets=False, is_an_image=True)

    collate_fn = None
    return train_dataset, test_dataset, collate_fn

def load_cifar10():
    transform = transforms.Compose(
        [transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                (0.2023, 0.1994, 0.2010)),])
    train_set = CIFAR10(root="cifar10", download=False, train=True, transform=transform)
    train_set.data = torch.tensor(train_set.data)
    train_set.targets = torch.tensor(train_set.targets)

    test_set = CIFAR10(root="cifar10", download=False, train=False, transform=transform)
    test_set.data = torch.tensor(test_set.data)
    test_set.targets = torch.tensor(test_set.targets)
    
    train_dataset = CustomDataset(data=train_set.data, targets=train_set.targets, transform=transform, real_targets=False, is_an_image=True)
    test_dataset = CustomDataset(data=test_set.data, targets=test_set.targets, transform=transform, real_targets=False, is_an_image=True)

    collate_fn = None
    return train_dataset, test_dataset, collate_fn

if __name__ == "__main__":
    load_binary_cifar10()
