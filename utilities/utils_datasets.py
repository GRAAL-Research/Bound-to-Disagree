import torch
from PIL import Image
from torchvision import transforms
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, targets, indices=None, transform=transforms.ToTensor(), real_targets=False, is_an_image=True):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if indices is not None:
            if isinstance(data, list):
                self.data = []
                if isinstance(indices, torch.Tensor):
                    for i in range(len(indices)):
                        if indices[i]:
                            self.data.append(data[i])
                else:
                    for i in indices:
                        self.data.append(data[i])
            else:
                self.data = data[indices]
            self.targets = targets[indices]
        else:
            self.data = data
            self.targets = targets

        self.transform = transform
        self.real_targets = real_targets
        self.is_an_image = is_an_image

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        if self.real_targets:
            target = self.targets[idx]
        else:
            target = int(self.targets[idx])

        if self.is_an_image:
            # doing this so that it is consistent with all other datasets
            # to return a PIL Image
            img = Image.fromarray(img.numpy()) #, mode="L")

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def clone_dataset(self, indices):
        return CustomDataset(data=self.data,
                            targets=self.targets,
                            indices=indices,
                            transform=self.transform,
                            real_targets=self.real_targets,
                            is_an_image=self.is_an_image)


def split_prior_train_validation_dataset(dataset : CustomDataset, prior_size : float, validation_size : float):
    if prior_size == 0.0:
        train_data, val_data = torch.utils.data.random_split(dataset, [1-validation_size, validation_size])
        train_set = dataset.clone_dataset(train_data.indices)
        validation_set = dataset.clone_dataset(val_data.indices)

        assert len(train_set) + len(validation_set) == len(dataset)
        return None, train_set, validation_set
    
    splits = [prior_size, 1-prior_size - validation_size, validation_size]
    prior_data, train_data, val_data = torch.utils.data.random_split(dataset, splits)
    prior_set = dataset.clone_dataset(prior_data.indices)
    train_set = dataset.clone_dataset(train_data.indices)
    validation_set = dataset.clone_dataset(val_data.indices)
    
    assert len(prior_set) + len(train_set) + len(validation_set) == len(dataset)

    return prior_set, train_set, validation_set

def split_train_validation_dataset(dataset : CustomDataset, validation_size : float):
    train_data, val_data = torch.utils.data.random_split(dataset, [1-validation_size, validation_size])
    train_set = dataset.clone_dataset(train_data.indices)
    validation_set = dataset.clone_dataset(val_data.indices)

    assert len(train_set) + len(validation_set) == len(dataset)
    return train_set, validation_set


def get_data_augmentation_transform(dataset, method="p2l"):
    if dataset == "mnist":
        return transforms.Compose([transforms.RandomAffine(
                                    degrees=20,
                                    translate=(0.1, 0.1),
                                    scale=(0.9, 1.1)),
                                    transforms.ToTensor()])
    elif dataset == "cifar10" and method == "p2l":
        return transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAutocontrast(0.5),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                        (0.2023, 0.1994, 0.2010))
            ]
            )
    elif dataset == "cifar10" and method == "baseline":
        return transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                        (0.2023, 0.1994, 0.2010))
            ]
            )
    elif dataset == "amazon":
        return None
    else:
        raise NotImplementedError(f"Data augmentation for {dataset} is not implemented yet.")
    
def get_dataset_callbacks(dataset):
    if dataset == "cifar10":
        return [LearningRateMonitor(logging_interval="step")]
    elif dataset == "mnist":
        return [EarlyStopping(monitor="validation_error", patience=10)]
    elif dataset == "amazon":
        return [EarlyStopping(monitor="validation_error", patience=2)]
    else:
        raise NotImplementedError(f"The callbacks for {dataset} are not implemented yet.")
    
