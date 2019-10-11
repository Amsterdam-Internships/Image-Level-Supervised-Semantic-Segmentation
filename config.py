"""
Contains all parameters for the code
"""
import torch


class Config(object):

    # Contains general configuration settings
    general_config = {
        'model_name': 'city_garbage_detection',  # Models and logs will be stored using this name
        'debug': False,  # If "True", prevents saving of models/data and allows for visualizations
        'random_seed': 1234
    }

    # Contains parameters of folder and file locations.
    # Change the project root folder and the dataset root folder.
    locations = {
        'data_root_folder': '/home/arend/Documents/Thesis/ImageData/garbage-object-detection',
        'image_folder': 'images',
        'annotation_folder': 'annotations/xmls',
        'gt_segment_folder': 'segments',
        'train_labels': 'g_train.txt',
        'val_labels': 'g_val.txt',
        'test_labels': 'test.txt',
        'root_folder': '/home/arend/Documents/Thesis/GuidedSaliencyWithPRM/',
        'runs_folder': 'runs',
        'model_storage': 'data',
        'deeplab_snapshots': 'snapshots',
        'guide_storage': 'tmp/guide',
        'saliency_storage': 'tmp/sal',
        'attention_storage': 'tmp/attention',
        'unbin_attention_storage': 'tmp/unbin_attention',
        'segmentation_storage': 'tmp/predicted_images',
        'pascal_submission_storage': 'tmp/pascal_results'
    }

    # Contains all hyperparameters related to the classifier
    classifier_config = {
        'pretrained': True,  # Pretrained on ImageNet, from PyTorch model zoo
        'weights_from_file': False,  # Load weights from a file
        'weights_file_name': 'city_garbage.pt',  # Name of file containing network weights
        'batch_size': 15,
        'learning_rate': 1e-5,
        'max_training_steps': 8000,
        'eval_freq': 100,
        'model_save_freq': 20,
        'weight_decay': 0,
        'weighted_loss': False,  # Weighted loss function based on Pascal class occurences
        'target_layer': '42'  # Final network layer name, used for attention generation in Grad-CAM
    }

    # Contains all hyperparameters related to guide generation
    guide_config = {
        'normalize_gradcam': True,
        'saliency_bin_threshold': 0.5,
        'attention_bin_threshold': 0.4,
    }

    # Contains all hyperparameters related to the segmenter
    deeplab_config = {
        'batch_size': 4,
        'learning_rate': 2.5e-4,
        'max_training_steps': 2000,  # DeepLab default is 20000, but for training efficiency, experiments on 2000
        'model_save_freq': 1000,
        'weight_decay': 0.0005,
        'input_size': '321,321',
        'momentum': 0.9,
        'power': 0.9,
        'random_mirror': True,
        'random_scale': True,
        'gt_percentage': 0,  # Percentage ground truth in training deeplab, for semi-supervised learning experiments
        'save_predictions': True,  # Store predicted masks
        'crf_postprocessing': True,
        'crf_label_confidence': 0.9,
        'gt_available': False  # if ground truth segments are not available, no mIoU will be calculated
    }

    # Contains all hyperparameters related to the dataset
    dataset_config = {
        'name': 'CityGarbage',  # Special functionality for "PascalVOC"
        'num_classes': 3,  # Number of classes excluding background class(es)
        'num_classes_bg': 4,  # Number of classes including background class(es)
        'ignore_label': 255  # Pixel value to be ignored in ground truth segmentations
    }

    def __init__(self):
        # Determine whether to run on GPU or CPU
        if torch.cuda.is_available():
            self.general_config['device'] = 'cuda:0'
        else:
            self.general_config['device'] = 'cpu'