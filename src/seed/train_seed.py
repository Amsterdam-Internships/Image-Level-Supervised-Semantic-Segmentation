"""
This file trains a multi-label classifier.
This classifier is later used for generating network attention.
Training progress can be monitored in Tensorboard and images with classification predictions can be displayed.
"""

import os
import os.path as osp
import time

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torchvision.models as models
from tensorboardX import SummaryWriter
from sklearn.metrics import hamming_loss, precision_score, recall_score

from config import Config
from seed.seed_dataset import SeedTrainDataset, GarbageTrainDataset, GarbageTestDataset
from pascal_voc_utils import pascal_get_weights


def get_metrics(out, y):
    """
    Determines performance metrics.
    Uses scipy, so conversion from tensors to numpy arrays is needed.
    Sigmoid included in BCELoss, but not in output.
    :param out: output of the final convolutional layer of the neural network
    :param y: ground truth labels
    :return: hamming loss, micro/macro precision and recall
    """

    outnp = torch.sigmoid(out).cpu().detach().numpy()
    outnp = (outnp > 0.5).astype(int)

    ynp = y.cpu().detach().numpy()

    return (hamming_loss(outnp, ynp),
            precision_score(outnp, ynp, average='micro'),
            recall_score(outnp, ynp, average='micro'),
            precision_score(outnp, ynp, average='macro'),
            recall_score(outnp, ynp, average='macro'))


def train(config, train_loader, val_loader, net, optimizer, criterion):
    """
    Performs training of the model.

    :param config: config file
    :param train_loader: training data loader
    :param val_loader: validation data loader
    :param net: neural network for classification
    :param optimizer: optimization method (SGD/Adam)
    :param criterion: loss function for backpropagation
    """

    # Init performance logging on Tensorboard
    writer_location = osp.join(config.locations['root_folder'],
                               config.locations['runs_folder'],
                               'seed',
                               config.general_config['model_name'])
    writer = SummaryWriter(writer_location)

    net.train()
    step = 0

    # File for tracking progress
    text_file = open(osp.join(config.locations['root_folder'],
                              config.general_config['model_name'] + ".txt"), "a")
    print('-----', file=text_file)
    print('Training Classifier..', file=text_file)
    write_config_parameters(text_file, config)

    start_time = time.time()

    while step < config.classifier_config['max_training_steps']:
        for batch_idx, (x, y, ids, _) in enumerate(train_loader):

            if batch_idx % 1000 == 0:
                print("Step: " + str(step), file=text_file)

            x = x.to(config.general_config['device'])
            y = y.to(config.general_config['device'])

            optimizer.zero_grad()

            # Make predictions
            out = net(x)

            # Determine loss
            loss = criterion(out, y)
            writer.add_scalar('loss/train', loss, global_step=step)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Save model
            if step % config.classifier_config['model_save_freq'] == 0 and step != 0:
                print("Saving Weights: " + str(step), file=text_file)

                torch.save(net.state_dict(), osp.join(config.locations['root_folder'],
                                                      config.locations['model_storage'],
                                                      config.general_config['model_name']
                                                      + '_classification'
                                                      + '.pt'))

            # Validate model
            if step % config.classifier_config['eval_freq'] == 0 and step != 0:
                validate(config, val_loader, net, criterion, writer, step, text_file)
                # Set network back to training mode
                net.train()

            step += 1
            if step > config.classifier_config['max_training_steps']:
                break

            if config.general_config['debug']:
                break

    # Time operations
    training_time = time.time() - start_time
    minutes, seconds = divmod(int(training_time), 60)
    hours, minutes = divmod(minutes, 60)
    print('Time to complete training:  {:d}:{:02d}:{:02d}'.format(hours, minutes, seconds), file=text_file)

    text_file.close()


def validate(config, val_loader, net, criterion, writer, step, text_file):
    """
    Reports performance of the network on the validation set.
    The entire validation set is considered per call.

    :param config: config file
    :param val_loader: validation data loader
    :param net: neural network for classification
    :param criterion: loss function for backpropagation
    :param writer: SummaryWriter for tensorboard
    :param step: current training step
    :param text_file: logging file
    :return:
    """

    net.eval()

    # intialize evaluation metrics
    mean_loss = 0
    mean_ham = 0
    mean_mic_p = 0
    mean_mic_r = 0
    mean_mac_p = 0
    mean_mac_r = 0

    # Pass all samples through network, but do not edit network
    with torch.no_grad():
        for batch_idx, (x, y, ids, _) in enumerate(val_loader):

            x = x.to(config.general_config['device'])
            y = y.to(config.general_config['device'])

            # Make predictions
            out = net(x)

            loss = criterion(out, y).detach().cpu().numpy()
            ham, mic_p, mic_r, mac_p, mac_r = get_metrics(out, y)

            # Update means
            mean_loss = (batch_idx * mean_loss + loss) / (batch_idx + 1)
            mean_ham = (batch_idx * mean_ham + ham) / (batch_idx + 1)
            mean_mic_p = (batch_idx * mean_mic_p + mic_p) / (batch_idx + 1)
            mean_mic_r = (batch_idx * mean_mic_r + mic_r) / (batch_idx + 1)
            mean_mac_p = (batch_idx * mean_mac_p + mac_p) / (batch_idx + 1)
            mean_mac_r = (batch_idx * mean_mac_r + mac_r) / (batch_idx + 1)

            if config.general_config['debug']:
                if batch_idx > 5:
                    break

        # Log values
        writer.add_scalar('loss/val', mean_loss, global_step=step)
        writer.add_scalar('metrics/hamming_loss', mean_ham, global_step=step)
        writer.add_scalar('metrics/micro_precision', mean_mic_p, global_step=step)
        writer.add_scalar('metrics/micro_recall', mean_mic_r, global_step=step)
        writer.add_scalar('metrics/macro_precision', mean_mac_p, global_step=step)
        writer.add_scalar('metrics/macro_recall', mean_mac_r, global_step=step)

    print("Validate: " + str(step), file=text_file)


def write_config_parameters(text_file, config):
    """
    Writes the configuration parameters in the logging file

    :param text_file: logging text file
    :param config: config file
    :return:
    """

    print("General config", file=text_file)
    print(config.general_config, file=text_file)
    print("Network config", file=text_file)
    print(config.classifier_config, file=text_file)


def load_model(config):
    """
    Loads the neural network architecture.
    Output nodes of last layer are changed to match the number of classes in the dataset.

    :param config: config file
    :return: neural network for classification
    """

    if config.classifier_config['pretrained']:
        vgg = models.vgg16_bn(pretrained=True)

        vgg.classifier[6] = nn.Linear(4096, config.dataset_config['num_classes'], bias=True)
        net = vgg.to(config.general_config['device'])

    else:
        vgg = models.vgg16_bn()
        vgg.classifier[6] = nn.Linear(4096, config.dataset_config['num_classes'], bias=True)
        net = vgg.to(config.general_config['device'])

    return net


def load_dataset(config):
    """
    Loads the dataset objects.
    The training and validation image sets are disjoint, but undergo the same augmentations.

    :param config: config file
    :return: train and validation datasets
    """

    train_labels_path = osp.join(config.locations['root_folder'],
                                 config.locations['train_labels'])
    val_labels_path = osp.join(config.locations['root_folder'],
                               config.locations['val_labels'])

    if config.dataset_config['name'] == 'PascalVOC':
        train_dataset = SeedTrainDataset(osp.join(config.locations['data_root_folder'],
                                                  config.locations['image_folder']),
                                         osp.join(config.locations['data_root_folder'],
                                                  config.locations['annotation_folder']),
                                         osp.join(config.locations['data_root_folder'],
                                                  config.locations['gt_segment_folder']),
                                         train_labels_path)
        print('{} images in training dataset'.format(len(train_dataset)))

        val_dataset = SeedTrainDataset(osp.join(config.locations['data_root_folder'],
                                                config.locations['image_folder']),
                                       osp.join(config.locations['data_root_folder'],
                                                config.locations['annotation_folder']),
                                       osp.join(config.locations['data_root_folder'],
                                                config.locations['gt_segment_folder']),
                                       val_labels_path)
        print('{} images in validation dataset'.format(len(val_dataset)))
    else:
        train_dataset = GarbageTrainDataset(osp.join(config.locations['data_root_folder'],
                                                     config.locations['image_folder']),
                                            osp.join(config.locations['data_root_folder'],
                                                     config.locations['annotation_folder']),
                                            train_labels_path)
        print('{} images in training dataset'.format(len(train_dataset)))

        val_dataset = GarbageTestDataset(osp.join(config.locations['data_root_folder'],
                                                  config.locations['image_folder']),
                                         osp.join(config.locations['data_root_folder'],
                                                  config.locations['annotation_folder']),
                                         val_labels_path)
        print('{} images in validation dataset'.format(len(val_dataset)))

    return train_dataset, val_dataset


def load_loss(config):
    """
    Loads the loss function

    :param config: config file
    :return: loss function for backpropagation
    """

    if config.classifier_config['weighted_loss']:
        weights = pascal_get_weights()
        weights = weights.to(config.general_config['device'])
        criterion = nn.BCEWithLogitsLoss(pos_weight=weights)
    else:
        criterion = nn.BCEWithLogitsLoss()

    return criterion


def load_optimizer(config, net):
    """
    Loads the optimizer function and assigns what parameters to optimize

    :param config: config file
    :param net: neural network architecture
    :return: optimization method
    """
    optimizer = torch.optim.Adam(net.parameters(), lr=config.classifier_config['learning_rate'],
                                 weight_decay=config.classifier_config['weight_decay'])
    return optimizer


def load_network_weights(config, net):
    """
    Loads network weights of a pretrained model (e.g. on ImageNet).

    :param config: config file
    :param net: neural network architecture
    :return: neural network with weights
    """

    net.load_state_dict(torch.load(osp.join(
        config.locations['root_folder'],
        config.locations['model_storage'],
        config.general_config['weights_file_name']),
        map_location=lambda storage, loc: storage))

    return net


def main(config):

    cudnn.enabled = True

    net = load_model(config)

    cudnn.benchmark = True

    train_dataset, val_dataset = load_dataset(config)

    # Make iterable batches
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.classifier_config['batch_size'],
        shuffle=True)

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config.classifier_config['batch_size'],
        shuffle=True)

    criterion = load_loss(config)
    optimizer = load_optimizer(config, net)

    if config.classifier_config['weights_from_file']:
        load_network_weights(config, net)

    else:
        # Run the training operation
        train(config, train_loader, val_loader, net, optimizer, criterion)


if __name__ == '__main__':
    test_config = Config()

    test_config.dataset_config['num_classes'] = 3
    test_config.dataset_config['name'] = 'Garbage'
    test_config.locations['data_root_folder'] = '/home/arend/Documents/Thesis/ImageData/garbage-object-detection/'
    test_config.locations['image_folder'] = 'images'
    test_config.locations['train_labels'] = 'g_train.txt'
    test_config.locations['val_labels'] = 'g_val.txt'
    test_config.general_config['model_name'] = 'garbage_dataset'
    test_config.classifier_config['weighted_loss'] = False

    main(test_config)
