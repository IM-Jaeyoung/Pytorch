import torch
from torchvision import transforms, datasets
import torchvision
import numpy as np
import matplotlib.pyplot as plt

data_transform = transforms.Compose([
	transforms.Resize((256, 1024)),
	transforms.ToTensor()])

kitti_dataset= datasets.ImageFolder(root='/home/jyim/hdd2/pytorch_KITTI', transform=data_transform)
dataset_loader = torch.utils.data.DataLoader(kitti_dataset, batch_size=4, shuffle=None, num_workers=1)

dataset_sizes = len(dataset_loader)

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0)  # pause a bit so that plots are updated


# Get a batch of training data
inputs, classes = next(iter(dataset_loader))
print(inputs.size())

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

imshow(out)