import torch
from math import ceil

from torchvision.transforms import transforms
from utils.jpeg import JPEGCompression
from torchvision.datasets.stl10 import STL10
from torchvision.datasets.oxford_iiit_pet import OxfordIIITPet


class DatasetWrapper():
    def __init__(self, train_dataset, val_dataset,
                 batch_size, num_classes, workers, input_shape):
        self.val_dataset = val_dataset
        self.train_dataset = train_dataset
        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                        shuffle=True, num_workers=workers)
        self.val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                                      shuffle=True, num_workers=workers)

        self.train_loader_len = ceil(len(train_dataset) / batch_size)
        self.val_loader_len = ceil(len(val_dataset) / batch_size)
        self.train_samples = len(train_dataset)
        self.val_samples = len(val_dataset)
        self.num_classes = num_classes
        self.input_shape = (3,) + input_shape


def get_transforms(split='train', input_size=(96, 96), jpeg_quality=None):
    if split == 'train':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomRotation(degrees=(-20, 20)),
            transforms.RandomResizedCrop(input_size, antialias=True),
            transforms.RandomHorizontalFlip(),
        ])
    elif split == 'test':
        transform = transforms.Compose([
            transforms.Resize(input_size),
            JPEGCompression(jpeg_quality),
            transforms.ToTensor()
        ])
    return transform


def get_dataset(dataset_config, batch_size, jpeg_quality=None, workers=8):
    dataset_name = dataset_config['name']

    num_classes = 0
    if dataset_name == "stl10":
        input_shape = (96, 96)
        num_classes = 10
        train_dataset = STL10(root="./data/stl10",download=True, split="train",
                              transform=get_transforms(split='train', input_size=input_shape),)
        val_dataset = STL10(root="./data/stl10", download=True, split="test",
                            transform=get_transforms(split='test', jpeg_quality=jpeg_quality, input_size=input_shape))


    elif dataset_name == "oxford_pets":
        input_shape = (224, 224)
        num_classes = 37
        train_dataset = OxfordIIITPet(root="./data/pets", download=True, split="trainval",
                                      transform=get_transforms(split='train', input_size=input_shape))
        val_dataset = OxfordIIITPet(root="./data/pets", download=True, split="test",
                                    transform=get_transforms(split='test', input_size=input_shape, jpeg_quality=jpeg_quality))


    ############################ Loaders ########################################
    d = DatasetWrapper(train_dataset, val_dataset,
                       batch_size, num_classes,
                       workers, input_shape)

    return d
