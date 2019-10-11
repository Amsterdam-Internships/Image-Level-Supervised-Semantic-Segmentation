"""

Functions that deal with labels

"""
import torch
import numpy as np


def binary_to_index(binary_labels):
    """
    Takes a tensor with binary labels, returns a nested list with indices
    """

    index_labels = []

    binary_labels = binary_labels.tolist()

    # Account for batch size of 1
    if not type(binary_labels[0]) is list:
        binary_labels = [binary_labels]

    for image in binary_labels:

        labels = []

        for i, bin_class in enumerate(image):
            if bin_class == 1:
                labels.append(i)

        index_labels.append(labels)

    return index_labels


def output_to_binary(out):
    """
    Takes a tensor with network output values.
    Applies sigmoid and thresholds to return binary labels
    Returns numpy array of labels
    """

    probabilities = torch.sigmoid(out).cpu().detach().numpy()

    # If no class over 50% confident, choose max class
    for i in range(probabilities.shape[0]):
        image_prediction = probabilities[i]
        if not np.any(image_prediction > 0.5):
            probabilities[i, np.argmax(image_prediction)] = 0.51

    binary = (probabilities > 0.5).astype(int)

    return binary
