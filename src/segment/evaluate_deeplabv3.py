"""
This file will take a trained DeepLab network, run segmentation on images and produce:
- a mean intersection over union + confusion matrix
- matplotlib images comparing the ground truth and predictions (using Pascal color scheme)
- images for submitting to Pascal evaluation server (pixel value = class label)

Code adapted from: https://github.com/speedinghzl/Pytorch-Deeplab
"""

import os
import os.path as osp
import time

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils import data
from torchvision import models
from tqdm import tqdm

from config import Config
from segment.datasets import VOCDataSet, VOCDataTestSet
from util.pytorch_utils import Interpolate
from util.apply_dense_crf import apply_dense_crf
from segment.metric import ConfusionMatrix

# ImageNet mean colors
IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)


def get_iou(confusion_matrix, text_file, save_path=None):
    average_jaccard, j_list, M = confusion_matrix.jaccard()
    print('meanIOU: ' + str(average_jaccard) + '\n', file=text_file)
    if save_path:
        with open(save_path, 'w') as file:
            file.write('meanIOU: ' + str(average_jaccard) + '\n')
            file.write(str(j_list) + '\n')
            file.write(str(M) + '\n')


# Produce output for pascal submission server (pixel value = class label)
def produce_pascal_output(config, pred, name):
    img = Image.fromarray(pred.astype(np.uint8), mode='L')
    img.save(osp.join(config.locations['root_folder'],
                      config.locations['pascal_submission_storage'],
                      str(name) + '.png'))


def show_all(config, gt, pred, name):
    import matplotlib
    if "DISPLAY" not in os.environ:
        matplotlib.use('Agg')  # Allows saving images without showing (on server)
    import matplotlib.pyplot as plt
    from matplotlib import colors

    dpi = 96

    # Remove ignore labels (which messes up display)
    gt[gt == int(config.dataset_config['ignore_label'])] = 0

    classes = np.array(('background',  # bg always index 0
                        'aeroplane', 'bicycle', 'bird', 'boat',
                        'bottle', 'bus', 'car', 'cat', 'chair',
                        'cow', 'diningtable', 'dog', 'horse',
                        'motorbike', 'person', 'pottedplant',
                        'sheep', 'sofa', 'train', 'tvmonitor'))
    colormap = [(0, 0, 0), (0.5, 0, 0), (0, 0.5, 0), (0.5, 0.5, 0), (0, 0, 0.5), (0.5, 0, 0.5), (0, 0.5, 0.5),
                (0.5, 0.5, 0.5), (0.25, 0, 0), (0.75, 0, 0), (0.25, 0.5, 0), (0.75, 0.5, 0), (0.25, 0, 0.5),
                (0.75, 0, 0.5), (0.25, 0.5, 0.5), (0.75, 0.5, 0.5), (0, 0.25, 0), (0.5, 0.25, 0), (0, 0.75, 0),
                (0.5, 0.75, 0), (0, 0.25, 0.5)]
    cmap = colors.ListedColormap(colormap)
    bounds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    # Image to numpy
    original_image = Image.open(osp.join(config.locations['data_root_folder'],
                                         config.locations['image_folder'],
                                         name + '.jpg'))
    original_image = np.asarray(original_image)

    fig = plt.figure(figsize=(500 / dpi, 281 / dpi * 2), dpi=dpi, frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 0.5])
    ax2 = plt.Axes(fig, [0., 0.5, 1., 0.5])
    ax.set_axis_off()
    ax2.set_axis_off()
    fig.add_axes(ax)
    fig.add_axes(ax2)
    ax.imshow(original_image)
    ax.imshow(pred, cmap=cmap, norm=norm, alpha=0.7)
    ax2.imshow(original_image)
    plt.savefig(osp.join(config.locations['root_folder'],
                         config.locations['segmentation_storage'],
                         str(name) + '.png'))
    plt.close(fig)


def get_crf(config, output, name, gt_prob=0.9):

    # Get involved classes and unary energy from labels
    involved_classes, unary = np.unique(output, return_inverse=True)

    # CRF inference not possible on just background
    if len(involved_classes) == 1:
        return output

    # Image to numpy
    original_image = Image.open(osp.join(config.locations['data_root_folder'],
                                         config.locations['image_folder'],
                                         name[0] + '.jpg'))
    original_image = np.array(original_image)

    output = apply_dense_crf(original_image,
                             unary,
                             involved_classes,
                             gt_prob=gt_prob,
                             stage='deeplab')

    return output


def main(config):
    """Create the model and start the evaluation process."""

    start_time = time.time()

    # Needs pre-trained, as auxiliary loss features are somehow not included in empty object.
    model = models.segmentation.deeplabv3_resnet101(pretrained=True)
    model.classifier[4] = nn.Conv2d(256, config.dataset_config['num_classes_bg'], kernel_size=(1, 1), stride=(1, 1))
    model.aux_classifier[4] = nn.Conv2d(256, config.dataset_config['num_classes_bg'], kernel_size=(1, 1), stride=(1, 1))

    # Logging
    text_file = open(osp.join(config.locations['root_folder'], config.general_config['model_name'] + ".txt"), "a")
    print("-----", file=text_file)
    print('Evaluating DeepLab..', file=text_file)

    # Replace network weights with weights from train_deeplab
    saved_state_dict = torch.load(osp.join(
                                  config.locations['root_folder'],
                                  config.locations['deeplab_snapshots'],
                                  config.general_config['model_name']
                                  + '_segmentation' + '.pth'))

    model.load_state_dict(saved_state_dict)
    model.eval()
    model.to(config.general_config['device'])

    if config.deeplab_config['gt_available']:
        test_loader = data.DataLoader(
            VOCDataSet(config.locations['data_root_folder'],
                       osp.join(config.locations['root_folder'], config.locations['val_labels']),
                       config.locations['image_folder'],
                       config.locations['gt_segment_folder'],
                       osp.join(config.locations['root_folder'], config.locations['guide_storage']),
                       crop_size=(505, 505),
                       mean=IMG_MEAN,
                       scale=False,
                       mirror=False,
                       gt_percentage=1),
            batch_size=1,
            shuffle=False,
            pin_memory=True)

    else:
        test_loader = data.DataLoader(
            VOCDataTestSet(config.locations['data_root_folder'],
                           None,
                           crop_size=(505, 505),
                           mean=IMG_MEAN),
            batch_size=1,
            shuffle=False,
            pin_memory=True)

    interpolate = Interpolate(size=(505, 505), mode='bilinear', align_corners=True)
    confusion_matrix = ConfusionMatrix(config.dataset_config['num_classes_bg'])

    with torch.no_grad():
        for index, batch in enumerate(tqdm(test_loader)):
            if index % 100 == 0:
                print('%d processed' % index, file=text_file)

            if config.deeplab_config['gt_available']:
                image, label, size, name = batch
            else:
                image, size, name = batch

            size = size[0].numpy()
            output = model(Variable(image).to(config.general_config['device']))['out']  # N x C x H x W
            output = interpolate(output)
            output = output.cpu().data[0].numpy()  # Remove batch dimension (batch size is always one.)

            output = output[:, :size[0], :size[1]]

            if config.deeplab_config['gt_available']:
                gt = np.asarray(label[0].numpy()[:size[0], :size[1]], dtype=np.int)

            if config.deeplab_config['crf_postprocessing']:
                output = np.asarray(np.argmax(output, axis=0), dtype=np.int)
                output = get_crf(config, output, name, gt_prob=config.deeplab_config['crf_label_confidence'])
            else:
                output = np.asarray(np.argmax(output, axis=0), dtype=np.int)

            # Produce greyscale images with label as color
            if config.dataset_config['name'] == 'PascalVOC':
                produce_pascal_output(config, output, name[0])

            if config.deeplab_config['gt_available']:
                # Save / Display predictions
                if config.deeplab_config['save_predictions']:
                    show_all(config, gt.copy(), output, name[0])

                item_matrix = confusion_matrix.generateM([gt.flatten(), output.flatten()])
                confusion_matrix.addM(item_matrix)

    if config.deeplab_config['gt_available']:
        # Calculate IoU from confusion matrix
        get_iou(confusion_matrix,
                text_file,
                save_path=config.general_config['model_name'] + 'miou.txt')

    # Time operations
    training_time = time.time() - start_time
    minutes, seconds = divmod(int(training_time), 60)
    hours, minutes = divmod(minutes, 60)
    print('Time to complete evaluating deeplab:  {:d}:{:02d}:{:02d}'.format(hours, minutes, seconds), file=text_file)

    # Close logging file
    text_file.close()


if __name__ == '__main__':
    from config import Config

    test_config = Config()
    test_config.general_config['debug'] = True
    test_config.deeplab_config['save_predictions'] = True
    test_config.deeplab_config['gt_available'] = True
    test_config.deeplab_config['crf_postprocessing'] = True
    main(test_config)
