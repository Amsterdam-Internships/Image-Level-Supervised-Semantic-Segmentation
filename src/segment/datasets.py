"""
Creates a segmentation dataset.
In the training dataset, images can be cropped, mirrored and scaled.
Partial supervision is possible by setting a ground truth percentage

Code adapted from: https://github.com/speedinghzl/Pytorch-Deeplab
"""

import os
import os.path as osp
import random

import cv2
import numpy as np
from torch.utils import data


class VOCDataSet(data.Dataset):
    def __init__(self, root, list_path, image_folder, gt_segment_folder, label_folder, max_iters=None, crop_size=(321, 321), mean=(128, 128, 128),
                 scale=True, mirror=True, ignore_label=255, gt_percentage=0):
        self.root = root
        self.list_path = list_path
        self.image_folder = image_folder
        self.gt_segment_folder = gt_segment_folder
        self.label_folder = label_folder
        self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.gt_percentage = gt_percentage
        self.mean = mean
        self.is_mirror = mirror
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if max_iters is not None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []

        random.shuffle(self.img_ids)

        print("# names in file: " + str(len(self.img_ids)))
        print("opening images from: " + str(osp.join(self.root, self.image_folder)))
        print("opening labels from: " + str(osp.join(self.root, self.gt_segment_folder)))

        for idx, name in enumerate(self.img_ids):
            img_file = osp.join(self.root, self.image_folder, "%s.jpg" % name)

            # Load labels
            if gt_percentage == 1:
                # Fully Supervised
                label_file = osp.join(self.root, self.gt_segment_folder, "%s.png" % name)
            elif gt_percentage == 0:
                # Weakly Supervised
                label_file = osp.join(self.label_folder, "%s.png" % name)
            else:
                # Semi-Supervised
                if osp.isfile(osp.join(self.root, "SegmentationClass/%s.png" % name)):
                    label_file = osp.join(self.root, self.gt_segment_folder, "%s.png" % name)
                else:
                    label_file = osp.join(self.label_folder, "%s.png" % name)

            # Check if file exists
            if osp.isfile(label_file):
                self.files.append({
                    "img": img_file,
                    "label": label_file,
                    "name": name
                })
        assert len(self.files) > 0, "Dataset is empty, check whether correct folder is selected"

    def __len__(self):
        return len(self.files)

    @staticmethod
    def generate_scale_label(image, label):
        f_scale = 0.5 + random.randint(0, 11) / 10.0
        image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_NEAREST)
        return image, label

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)
        size = image.shape
        name = datafiles["name"]
        if self.scale:
            image, label = self.generate_scale_label(image, label)
        image = np.asarray(image, np.float32)
        image -= self.mean
        img_h, img_w = label.shape
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            img_pad = cv2.copyMakeBorder(image, 0, pad_h, 0,
                                         pad_w, cv2.BORDER_CONSTANT,
                                         value=(0.0, 0.0, 0.0))
            label_pad = cv2.copyMakeBorder(label, 0, pad_h, 0,
                                           pad_w, cv2.BORDER_CONSTANT,
                                           value=(self.ignore_label,))
        else:
            img_pad, label_pad = image, label

        img_h, img_w = label_pad.shape
        h_off = random.randint(0, img_h - self.crop_h)
        w_off = random.randint(0, img_w - self.crop_w)

        image = np.asarray(img_pad[h_off: h_off + self.crop_h, w_off: w_off + self.crop_w], np.float32)
        label = np.asarray(label_pad[h_off: h_off + self.crop_h, w_off: w_off + self.crop_w], np.float32)

        image = image.transpose((2, 0, 1))  # BGR to RGB
        if self.is_mirror:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]

        return image.copy(), label.copy(), np.array(size), name


class VOCDataTestSet(data.Dataset):
    def __init__(self, root, list_path, crop_size=(505, 505), mean=(128, 128, 128)):
        self.root = root
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.mean = mean
        if list_path is not None:
            self.img_ids = [i_id.strip() for i_id in open(self.list_path)]
        else:
            self.img_ids = [i_id.split('.')[0]
                            for i_id in os.listdir(osp.join(self.root, "video"))
                            if osp.isfile(osp.join(self.root, "video/%s.jpg" % i_id))]

        self.files = []
        for name in self.img_ids:
            img_file = osp.join(self.root, "video/%s.jpg" % name)
            self.files.append({
                "img": img_file,
                "name": name
            })
        assert len(self.files) > 0, "Dataset is empty, check whether correct folder is selected"

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        size = image.shape
        name = datafiles["name"]
        image = np.asarray(image, np.float32)
        image -= self.mean

        img_h, img_w, _ = image.shape
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            image = cv2.copyMakeBorder(image, 0, pad_h, 0,
                                       pad_w, cv2.BORDER_CONSTANT,
                                       value=(0.0, 0.0, 0.0))
        image = image.transpose((2, 0, 1))
        return image.copy(), np.array(size), name
