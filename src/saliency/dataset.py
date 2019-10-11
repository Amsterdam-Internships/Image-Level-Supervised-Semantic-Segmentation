"""
PiCANet dataset implementation in PyTorch

Code adapted from: https://github.com/Ugness/PiCANet-Implementation

"""
import os
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image


class CustomDataset(data.Dataset):
    def __init__(self, root_dir):
        self.image_list = sorted(os.listdir(root_dir))
        self.transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        self.root_dir = root_dir

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, item):
        img_name = '{}/{}'.format(self.root_dir, self.image_list[item])
        img = Image.open(img_name)
        width, height = img.size
        sample = img.convert('RGB')
        sample = self.transform(sample)
        return sample, (width, height), self.image_list[item]
