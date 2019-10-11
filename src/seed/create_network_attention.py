"""
This python file contains code that does the following:
- load a trained classifier
- produce the network attention using GradCAM
- save attention masks in tmp folder

"""

import os
import os.path as osp

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.models as models
from tqdm import tqdm
import numpy as np
from PIL import Image

from config import Config
from seed.seed_dataset import SeedTestDataset, GarbageTestDataset
from util.label_utils import binary_to_index, output_to_binary
from seed.grad_cam import GradCam


def produce_masks(config, net, dataset):
    """
    This function iterates over all images in the dataset and produces the network attention.
    The network attention is either produced per predicted label from the classifier,
    or per ground truth label from the image dataset

    :param config: config file
    :param net: classifier network
    :param dataset: image dataset
    :return:
    """

    text_file = open(osp.join(config.locations['root_folder'], config.general_config['model_name'] + ".txt"), "a")
    print("-----", file=text_file)
    print('Creating Network Attention..', file=text_file)

    if config.general_config['device'] == 'cpu':
        use_cuda = False
    else:
        use_cuda = True

    # Set to evaluation mode
    net.eval()

    # Init Grad-CAM model
    grad_cam = GradCam(model=net,
                       target_layer_names=[config.classifier_config['target_layer']],
                       use_cuda=use_cuda,
                       normalize=config.guide_config['normalize_gradcam'])

    for x, y, ids, (w, h) in tqdm(dataset):

        x = x.to(config.general_config['device'])

        # Add dimension for batch size
        x = x[None]

        labels = binary_to_index(y)

        labels = labels[0]  # Remove batch dimension
        image_id = ids.split('/')[-1][:-4]

        for target_index in labels:
            retrieve_attention(config, grad_cam, x, image_id, h, w, target_index,
                               osp.join(config.locations['root_folder'],
                                        config.locations['attention_storage']),
                               osp.join(config.locations['root_folder'],
                                        config.locations['unbin_attention_storage']))


def retrieve_attention(config, grad_cam, x, image_id, h, w, target_index, img_storage, float_storage):
    """
    This function applies GradCAM to the image to receive network attention on the input space.
    The attention is turned into a binary mask by applying a threshold and then stored.

    :param config: config file
    :param grad_cam: gradCAM model
    :param x: input image
    :param image_id: image file name
    :param h: image height
    :param w: image width
    :param target_index: class labels (either predicted or ground truth)
    :param img_storage: folder for saving binarized attention
    :param float_storage: folder for saving unbinarized attention
    :return:
    """
    attention_mask = grad_cam(x, target_index)
    attention_mask = attention_mask[None][None]  # Add batch size and channel dimension
    attention_mask = nn.functional.interpolate(attention_mask, size=(h, w))  # To original image dimensions

    # Save raw attention data in tmp folder
    np_attention_mask = attention_mask.squeeze().detach().cpu().numpy()
    np.save(osp.join(float_storage, image_id + '_' + str(target_index)), np_attention_mask.astype(np.float32))

    # If attention has not been normalized yet, normalize for binarization
    if not config.guide_config['normalize_gradcam']:
        attention_mask = attention_mask - torch.min(attention_mask)
        attention_mask = attention_mask / torch.max(attention_mask)

    bin_attention_mask = torch.where(attention_mask > config.guide_config['attention_bin_threshold'],
                                     torch.ones_like(attention_mask),
                                     torch.zeros_like(attention_mask))

    # Save network attention in tmp folder
    attention_save_name = image_id + '_' + str(target_index) + '.png'
    bin_attention_mask = bin_attention_mask.squeeze().detach().cpu().numpy() * 255

    img = Image.fromarray(bin_attention_mask.astype(np.uint8), mode='L')
    img.save(osp.join(img_storage, attention_save_name))


def load_model(config):
    """
    Loads the classifier

    :param config: config file
    :return: classifier network
    """
    # Load network
    vgg = models.vgg16_bn()
    vgg.classifier[6] = nn.Linear(4096, config.dataset_config['num_classes'], bias=True)
    net = vgg.to(config.general_config['device'])

    return net


def load_datasets(config):
    """
    Loads the datasets.
    Both datasets will have 'Test' augmentation, meaning no flipping/cropping is present.
    This will cause the network attention the be calculated for the original image

    :param config: config file
    :return: train and validation datasets
    """

    if config.dataset_config['name'] == 'PascalVOC':
        train_data = SeedTestDataset(osp.join(config.locations['data_root_folder'],
                                              config.locations['image_folder']),
                                     osp.join(config.locations['root_folder'],
                                              config.locations['train_labels']))

        val_data = SeedTestDataset(osp.join(config.locations['data_root_folder'],
                                            config.locations['image_folder']),
                                   osp.join(config.locations['root_folder'],
                                            config.locations['val_labels']))
    else:
        train_data = GarbageTestDataset(osp.join(config.locations['data_root_folder'],
                                                 config.locations['image_folder']),
                                        osp.join(config.locations['root_folder'],
                                                 config.locations['train_labels']))

        val_data = GarbageTestDataset(osp.join(config.locations['data_root_folder'],
                                               config.locations['image_folder']),
                                      osp.join(config.locations['root_folder'],
                                               config.locations['val_labels']))

    return train_data, val_data


def main(config):

    cudnn.enabled = True

    net = load_model(config)

    net.load_state_dict(torch.load(osp.join(
        config.locations['root_folder'],
        config.locations['model_storage'],
        config.general_config['model_name']
        + '_classification'
        + '.pt'),
        map_location=lambda storage, loc: storage))

    cudnn.benchmark = True

    train_data, val_data = load_datasets(config)

    produce_masks(config, net, train_data)
    produce_masks(config, net, val_data)


if __name__ == '__main__':
    test_config = Config()

    test_config.general_config['model_name'] = 'resfized_images_VGG16'

    main(test_config)
