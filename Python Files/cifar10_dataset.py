import os
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

class CIFAR10ImageNet(Dataset):
    def __init__(self, root, train=True, download=False):
        self.dataset = CIFAR10(root=root, train=train, download=download)
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, target = self.dataset[idx]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        return self.transform(img), target
