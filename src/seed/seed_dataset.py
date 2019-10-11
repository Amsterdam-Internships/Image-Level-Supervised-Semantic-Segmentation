"""
This file contains the dataset objects.
Different class objects exist for the Pascal VOC and Garbage Dataset.
Additionally, both of these contain a Train and Test dataset,
where the Train version augments the data and Test does not.
Crop size has been set to 224x224, matching the input size of VGG.
"""

import os
import os.path as osp
import random
from PIL import Image
import torch
from torch.utils.data.dataset import Dataset

import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from pascal_voc_utils import pascal_get_weights, pascal_get_means, pascal_label_to_name, pascal_labels_from_annotation
from garbage_utils import garbage_labels_from_annotation, image_to_pascal_size


class SeedTrainDataset(Dataset):

    def __init__(self, image_folder, annotation_folder, gt_seg_folder, label_file):
        """
        Reads a text file with image IDs and binary labels separated by spaces.
        Not all images have labels.
        Images with no mask are filtered.
        Outputs list of image file locations and labels.
        """
        self.image_names = []
        self.labels = []
        self.normalize = pascal_get_means()

        with open(label_file, "r") as text_file:
            image_ids = text_file.read().splitlines()

        for image_name in os.listdir(image_folder):
            image_id = str(image_name.split('.')[0])
            if image_id in image_ids:
                annotation_file = image_id + '.xml'
                annotation_file = osp.join(annotation_folder, annotation_file)
                gt_seg_file = osp.join(gt_seg_folder, image_id + '.png')
                if osp.isfile(annotation_file) and osp.isfile(gt_seg_file):
                    self.labels.append(pascal_labels_from_annotation(annotation_file))
                    self.image_names.append(osp.join(image_folder, image_name))

    def __len__(self):
        return len(self.image_names)

    def transforms(self, image):
        """
        Performs data augmentation.
        """

        # Random Horizontal Flip on mask and input
        if random.random() > 0.5:
            image = TF.hflip(image)

        width, height = image.size
        if width < 224 or height < 224:
            image = TF.resize(image, 224)

        # Random Crop of images
        i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(224, 224))
        image = TF.crop(image, i, j, h, w)

        # Images to tensor
        image = TF.to_tensor(image)

        # Normalize image colors
        image = self.normalize(image)

        return image

    def __getitem__(self, idx):
        """
        Loads the images, performs data augmentation and returns coupled data.
        """

        # Load images
        image = Image.open(self.image_names[idx])

        dims = image.size

        # Apply transformations
        image = self.transforms(image)

        # Load image labels
        labels = torch.FloatTensor(self.labels[idx])

        return image, labels, self.image_names[idx], dims


class SeedTestDataset(Dataset):

    def __init__(self, image_folder, annotation_folder, gt_seg_folder, label_file):
        """
        Reads a text file with image IDs and binary labels separated by spaces.
        Not all images have labels.
        Images with no mask are filtered.
        Outputs list of image file locations and labels.
        """
        self.image_names = []
        self.labels = []
        self.normalize = pascal_get_means()

        with open(label_file, "r") as text_file:
            image_ids = text_file.read().splitlines()

        for image_name in os.listdir(image_folder):
            image_id = str(image_name.split('.')[0])
            if image_id in image_ids:
                annotation_file = image_id + '.xml'
                annotation_file = osp.join(annotation_folder, annotation_file)
                gt_seg_file = osp.join(gt_seg_folder, image_id + '.png')
                if osp.isfile(annotation_file) and not osp.isfile(gt_seg_file):
                    self.labels.append(pascal_labels_from_annotation(annotation_file))
                    self.image_names.append(osp.join(image_folder, image_name))

    def __len__(self):
        return len(self.image_names)

    def transforms(self, image):
        """
        Performs data augmentation.
        """

        image = TF.resize(image, (224, 224))

        # Images to tensor
        image = TF.to_tensor(image)

        # Normalize image colors
        image = self.normalize(image)

        return image

    def __getitem__(self, idx):
        """
        Loads the images, performs data augmentation and returns coupled data.
        """

        # Load images
        image = Image.open(self.image_names[idx])

        dims = image.size

        # Apply transformations
        image = self.transforms(image)

        # Load image labels
        labels = torch.FloatTensor(self.labels[idx])

        return image, labels, self.image_names[idx], dims


class GarbageTrainDataset(Dataset):

    def __init__(self, image_folder, annotation_folder, label_file):

        self.image_names = []
        self.labels = []
        self.normalize = pascal_get_means()

        with open(label_file, "r") as text_file:
            image_ids = text_file.read().splitlines()

        for image_name in os.listdir(image_folder):
            image_id = str(image_name.split('.')[0])
            if image_id in image_ids:
                annotation_file = image_id + '.xml'
                annotation_file = osp.join(annotation_folder, annotation_file)
                if osp.isfile(annotation_file):
                    self.labels.append(garbage_labels_from_annotation(annotation_file))
                    self.image_names.append(osp.join(image_folder, image_name))

    def __len__(self):
        return len(self.image_names)

    def transforms(self, image):
        """
        Performs data augmentation.
        """
        # Resize to Pascal shape first
        if max(image.size) != 500:
            image = image_to_pascal_size(image)

        # Random Horizontal Flip on mask and input
        if random.random() > 0.5:
            image = TF.hflip(image)

        width, height = image.size
        if width < 224 or height < 224:
            image = TF.resize(image, 224)

        # Random Crop of images
        i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(224, 224))
        image = TF.crop(image, i, j, h, w)

        # Images to tensor
        image = TF.to_tensor(image)

        # Normalize image colors
        image = self.normalize(image)

        return image

    def __getitem__(self, idx):
        """
        Loads the images, performs data augmentation and returns coupled data.
        """

        # Load images
        image = Image.open(self.image_names[idx])

        dims = image.size

        # Apply transformations
        image = self.transforms(image)

        # Load image labels
        labels = torch.FloatTensor(self.labels[idx])

        return image, labels, self.image_names[idx], dims


class GarbageTestDataset(Dataset):

    def __init__(self, image_folder, annotation_folder, label_file):

        self.image_names = []
        self.labels = []
        self.normalize = pascal_get_means()

        with open(label_file, "r") as text_file:
            image_ids = text_file.read().splitlines()

        for image_name in os.listdir(image_folder):
            image_id = str(image_name.split('.')[0])
            if image_id in image_ids:
                annotation_file = image_id + '.xml'
                annotation_file = osp.join(annotation_folder, annotation_file)
                if osp.isfile(annotation_file):
                    self.labels.append(garbage_labels_from_annotation(annotation_file))
                    self.image_names.append(osp.join(image_folder, image_name))

    def __len__(self):
        return len(self.image_names)

    def transforms(self, image):
        """
        Performs data augmentation.
        """

        # Resize to Pascal shape first
        if max(image.size) != 500:
            image = image_to_pascal_size(image)

        # Resize to VGG input size
        image = TF.resize(image, (224, 224))

        # Images to tensor
        image = TF.to_tensor(image)

        # Normalize image colors
        image = self.normalize(image)

        return image

    def __getitem__(self, idx):
        """
        Loads the images, performs data augmentation and returns coupled data.
        """

        # Load images
        image = Image.open(self.image_names[idx])

        dims = image.size

        # Apply transformations
        image = self.transforms(image)

        # Load image labels
        labels = torch.FloatTensor(self.labels[idx])

        return image, labels, self.image_names[idx], dims
