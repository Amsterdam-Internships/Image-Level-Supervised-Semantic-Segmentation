"""
Custom util functions on the Pascal VOC 2012 dataset.
"""
from xml.etree import ElementTree

import torch
import torchvision.transforms as transforms
import numpy as np


def pascal_labels_from_annotation(annotation_file):
    """
    Returns class indices of an image given the annotation file.

    :param annotation_file:
    :return: list of label ids per annotation file
    """

    classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
               "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
               "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    class_ids = [0] * 20

    root = ElementTree.parse(annotation_file).getroot()
    for image_object in root.findall('object'):
        value = image_object.find('name').text
        if value in classes:
            class_ids[classes.index(value)] = 1

    return class_ids


def pascal_label_to_name(labels, background=False):
    """
    Sets labels to class names. Works in classifier (background=False) and segmenter (background=True)
    """

    if background:
        classes = ["__background__", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
                   "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
                   "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    else:
        classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
                   "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
                   "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

    # Single element
    if type(labels) is int:
        names = classes[labels]
    # List (single image)
    elif type(labels) is list:

        # Binary list
        if all(i == 1 or i == 0 for i in labels) and (len(labels) == 20 or len(labels) == 21):
            indices = np.where(labels)
            names = []
            for item in indices[0].tolist():
                names.append(classes[item])

        # List with indices
        else:
            names = [classes[label] for label in labels]
    # 2-D list
    elif type(labels) is np.ndarray:
        names = []
        for row in labels:
            indices_row = np.where(row)
            names_row = []
            for item in indices_row[0].tolist():
                names_row.append(classes[item])
            names.append(names_row)

    # Binary Tensor
    elif type(labels) is torch.Tensor:

        if len(labels.size()) == 2:
            names = []
            for row in labels:
                current_names = []
                indices = (row != 0).nonzero()
                for index in indices:
                    current_names.append(classes[index])
                names.append(current_names)

    else:
        names = "Could not determine class input: " + str(type(labels))

    return names


def pascal_get_weights():
    """
    Gets class weights, related to occurrence in training set
    """
    counts = [327, 268, 395, 260, 365, 213, 590, 539, 566, 151, 269, 632, 237, 265, 1994, 269, 171, 257, 273, 290]
    weights = [(5717-x)/x for x in counts]
    weights = torch.FloatTensor(weights)

    return weights


def pascal_get_means():
    """
    Gets mean pixel values, based on training set.
    Pixel means are based on ImageNet (like in Guided Saliency paper)
    """

    means = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    return means


def pascal_index_to_color(mask):
    """
    Converts a mask with pixel labels to a colored mask. Labels should always be in "background" mode
    :param mask: mask with class ids
    :return: mask with RGB colors, aligning with official Pascal class colors
    """
    colormap = [(0, 0, 0), (0.5, 0, 0), (0, 0.5, 0), (0.5, 0.5, 0), (0, 0, 0.5), (0.5, 0, 0.5), (0, 0.5, 0.5),
                (0.5, 0.5, 0.5), (0.25, 0, 0), (0.75, 0, 0), (0.25, 0.5, 0), (0.75, 0.5, 0), (0.25, 0, 0.5),
                (0.75, 0, 0.5), (0.25, 0.5, 0.5), (0.75, 0.5, 0.5), (0, 0.25, 0), (0.5, 0.25, 0), (0, 0.75, 0),
                (0.5, 0.75, 0), (0, 0.25, 0.5)]

    colored_mask = np.zeros((3, mask.shape[0], mask.shape[1]))

    for i, row in enumerate(mask):
        for j, label in enumerate(row):
            color = colormap[int(label)]

            colored_mask[0, i, j] = color[0]
            colored_mask[1, i, j] = color[1]
            colored_mask[2, i, j] = color[2]

    colored_mask = np.uint8(colored_mask * 255)
    colored_mask = np.transpose(colored_mask, (1, 2, 0))

    return colored_mask


