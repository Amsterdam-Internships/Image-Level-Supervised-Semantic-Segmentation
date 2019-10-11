"""
This file will train a DeepLab v3 architecture.
Training can be done based on ground truth labels, or on guide labels.
Initialization can be done from a model pre-trained on COCO (torchvision model zoo).

Code adapted from: https://github.com/speedinghzl/Pytorch-Deeplab
"""

import os
import os.path as osp
import timeit

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils import data
from tensorboardX import SummaryWriter
from torchvision import models
from tqdm import tqdm

from segment.datasets import VOCDataSet
from util.pytorch_utils import Interpolate

start = timeit.default_timer()

# ImageNet mean colors
IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)


def loss_calc(prediction, label, config):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    label = Variable(label.long()).to(config.general_config['device'])
    criterion = torch.nn.CrossEntropyLoss(ignore_index=config.dataset_config['ignore_label']).to(config.general_config['device'])

    return criterion(prediction, label)


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def get_1x_lr_params_NOscale(model):
    """
    This generator returns all the parameters of the net except for
    the last classification layer. Note that for each batchnorm layer,
    requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
    any batchnorm parameter
    """
    b = []

    b.append(model.conv1)
    b.append(model.bn1)
    b.append(model.layer1)
    b.append(model.layer2)
    b.append(model.layer3)
    b.append(model.layer4)

    for i in range(len(b)):
        for j in b[i].modules():
            for k in j.parameters():
                if k.requires_grad:
                    yield k


def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the last layer of the net,
    which does the classification of pixel into classes
    """
    b = []
    b.append(model.classifier.parameters())

    for j in range(len(b)):
        for i in b[j]:
            yield i


def adjust_learning_rate(optimizer, i_iter, config):
    """Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 160th epochs"""
    lr = lr_poly(config.deeplab_config['learning_rate'],
                 i_iter,
                 config.deeplab_config['max_training_steps'],
                 config.deeplab_config['power'])
    optimizer.param_groups[0]['lr'] = lr
    optimizer.param_groups[1]['lr'] = lr * 10


def write_config_parameters(text_file, config):
    print("General config", file=text_file)
    print(config.general_config, file=text_file)
    print("DeepLab config", file=text_file)
    print(config.deeplab_config, file=text_file)


def main(config):
    """Create the model and start the training."""

    h, w = map(int, config.deeplab_config['input_size'].split(','))
    input_size = (h, w)

    cudnn.enabled = True

    # Logging
    writer = SummaryWriter(osp.join(config.locations['root_folder'],
                                    config.locations['runs_folder'],
                                    'deeplab',
                                    config.general_config['model_name']))
    text_file = open(osp.join(config.locations['root_folder'],
                              config.general_config['model_name'] + ".txt"), "a")
    print("-----", file=text_file)
    print('Training DeepLab..', file=text_file)
    write_config_parameters(text_file, config)

    # Create network
    model = models.segmentation.deeplabv3_resnet101(pretrained=True)
    model.classifier[4] = nn.Conv2d(256, config.dataset_config['num_classes_bg'], kernel_size=(1, 1), stride=(1, 1))
    model.aux_classifier[4] = nn.Conv2d(256, config.dataset_config['num_classes_bg'], kernel_size=(1, 1), stride=(1, 1))

    model.train()
    model.to(config.general_config['device'])

    cudnn.benchmark = True

    if not os.path.exists(osp.join(config.locations['root_folder'], config.locations['deeplab_snapshots'])):
        os.makedirs(osp.join(config.locations['root_folder'], config.locations['deeplab_snapshots']))

    train_loader = data.DataLoader(
        VOCDataSet(config.locations['data_root_folder'],
                   osp.join(config.locations['root_folder'], config.locations['train_labels']),
                   config.locations['image_folder'],
                   config.locations['gt_segment_folder'],
                   osp.join(config.locations['root_folder'], config.locations['guide_storage']),
                   max_iters=config.deeplab_config['max_training_steps'] * config.deeplab_config['batch_size'],
                   crop_size=input_size,
                   scale=config.deeplab_config['random_scale'],
                   mirror=config.deeplab_config['random_mirror'],
                   gt_percentage=config.deeplab_config['gt_percentage'],
                   mean=IMG_MEAN),
        batch_size=config.deeplab_config['batch_size'],
        shuffle=True,
        num_workers=5,
        pin_memory=True)

    optimizer = optim.SGD([{'params': get_1x_lr_params_NOscale(model.backbone), 'lr': config.deeplab_config['learning_rate']},
                           {'params': get_10x_lr_params(model), 'lr': 10 * config.deeplab_config['learning_rate']}],
                          lr=config.deeplab_config['learning_rate'],
                          momentum=config.deeplab_config['momentum'],
                          weight_decay=config.deeplab_config['weight_decay'])

    optimizer.zero_grad()

    interpolate = Interpolate(size=input_size, mode='bilinear', align_corners=True)

    for i_iter, batch in enumerate(tqdm(train_loader)):
        images, labels, _, _ = batch
        assert images.shape[0] == labels.shape[0] and images.shape[2:] == labels.shape[1:]
        images = Variable(images).to(config.general_config['device'])

        optimizer.zero_grad()
        adjust_learning_rate(optimizer, i_iter, config)
        prediction = interpolate(model(images)['out'])
        loss = loss_calc(prediction, labels, config)
        loss.backward()
        optimizer.step()

        # Log process
        print('iter = ', i_iter, 'of', config.deeplab_config['max_training_steps'], 'completed, loss = ',
              loss.data.cpu().numpy(), file=text_file)
        writer.add_scalar('loss/train', loss, global_step=i_iter)

        if i_iter >= config.deeplab_config['max_training_steps'] - 1:
            print('save model ...', file=text_file)
            torch.save(model.state_dict(),
                       osp.join(config.locations['root_folder'],
                                config.locations['deeplab_snapshots'],
                                config.general_config['model_name']
                                + '_segmentation'
                                + '.pth'))
            break

        if i_iter % config.deeplab_config['model_save_freq'] == 0 and i_iter != 0:
            print('taking snapshot ...', file=text_file)
            torch.save(model.state_dict(),
                       osp.join(config.locations['root_folder'],
                                config.locations['deeplab_snapshots'],
                                config.general_config['model_name']
                                + '_segmentation'
                                + '.pth'))

    end = timeit.default_timer()
    training_time = end - start
    minutes, seconds = divmod(int(training_time), 60)
    hours, minutes = divmod(minutes, 60)
    print('Time to complete training:  {:d}:{:02d}:{:02d}'.format(hours, minutes, seconds), file=text_file)


if __name__ == '__main__':
    main()
