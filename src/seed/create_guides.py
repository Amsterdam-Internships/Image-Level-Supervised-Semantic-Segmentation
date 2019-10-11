"""
Functions in this file take the saliency mask and the network attention masks.
By combining these masks following one of several fusion protocols, a guide segmentation is produced.

Code adapted from: https://github.com/coallaoh/GuidedLabelling
"""
import os
import os.path as osp

import numpy as np
import skimage.morphology as skimorph
from PIL import Image
from tqdm import tqdm

from pascal_voc_utils import pascal_index_to_color
from apply_dense_crf import apply_dense_crf
from config import Config


def combine_masks(saliency, network_attention, original_attention, classes, name, config):
    """
    Mask combination. First small blobs are removed in attention maps and saliency.
    Then, based on strategy, saliency and attention blobs are merged.
    In g0, the saliency is assigned one of the predicted class labels.
    In g1, ...
    In g2, overlaps are compared. Saliency blobs with no overlapping attention are ignored.
    Saliency blobs that overlap with the attention of a single class are assigned that class.
    Saliency blobs that overlap with multiple attention blobs are divided through dense CRF inference.
    Attention blobs that do not overlap with saliency blobs are kept as-is.

    :param saliency: 2d binary tensor
    :param network_attention: list of 2d binary tensors
    :param original_attention: numpy array with raw network attention values (background first)
    :param classes: list of class labels
    :param name: image id
    :param config: config file
    :return: 2d int tensor with class labels as values per pixel
    """

    # Filter saliency blobs (remove blobs < 1% of image)
    saliency_blobs = filter_blobs(saliency, min_area=saliency.shape[0] * saliency.shape[1] * .01)

    # Convert classification labels to segmentation labels and add background as class
    classes = [x+1 for x in classes]
    # At least a single background pixel is assumed to be present in each image
    # Background class is labeled as "0".
    classes.append(0)

    # Filter network attention masks (excluding background, which has no attention)
    network_attention_dict = {}
    network_attention_overlap_dict = {}  # Binary for each attention blob
    for i in range(len(network_attention)):
        network_attention_dict[classes[i]] = filter_blobs(network_attention[i])
        network_attention_overlap_dict[classes[i]] = np.zeros(len(network_attention_dict[classes[i]]))

    merged_mask = np.ones(saliency.shape, dtype=np.uint8) * 255

    # Load RGB Image
    file_name = osp.join(config.locations['data_root_folder'], config.locations['image_folder'], name + '.jpg')
    original_image = Image.open(file_name)
    original_image = np.asarray(original_image, dtype=np.uint8)
    original_image = original_image.copy()

    # Set saliency blobs
    for saliency_blob in saliency_blobs:
        overlap = []
        for predicted_class in classes:
            # Ignore background class
            if predicted_class == 0:
                continue
            for idx, attention_blob in enumerate(network_attention_dict[predicted_class]):
                blob_overlap = compute_iou(saliency_blob, attention_blob)
                if blob_overlap > 0:
                    overlap.append([predicted_class, attention_blob])
                    # Mark network attention blob as overlapping.
                    network_attention_overlap_dict[predicted_class][idx] = 1

        # Check number of unique classes that overlap with saliency blob
        num_class_overlaps = len(np.unique(np.array([o[0] for o in overlap])))

        # If no attention overlaps with saliency, ignore saliency
        if num_class_overlaps == 0:
            merged_mask[saliency_blob.astype(bool)] = 255
        # If single attention class overlaps, assign all saliency to that class
        elif num_class_overlaps == 1:
            merged_mask[saliency_blob.astype(bool)] = overlap[0][0]
        # If attention of multiple classes overlap with saliency blob, apply crf inference
        elif num_class_overlaps >= 2:

            # Get attention of relevant classes
            current_class_indices = [0]  # Background attention always included
            current_blob_classes = [o[0] for o in overlap]
            current_class_indices += [i + 1 for i, x in enumerate(classes) if x in current_blob_classes]
            current_blob_attention = original_attention[:, :, current_class_indices]

            # Get inference
            involved_classes = np.unique(np.array([o[0] for o in overlap]))
            inference = apply_dense_crf(original_image,
                                        current_blob_attention[:, :, 1:],
                                        involved_classes,
                                        stage='guide')

            # Merge with results
            merged_mask[saliency_blob.astype(bool)] = inference[saliency_blob.astype(bool)]

    # Check for isolated network attention blobs. These are included in the final segmentation
    for predicted_class in classes:
        if predicted_class == 0:
            continue
        for idx, attention_blob in enumerate(network_attention_dict[predicted_class]):

            # If network attention blob does not overlap, assign it the
            if not network_attention_overlap_dict[predicted_class][idx]:
                merged_mask[attention_blob.astype(bool)] = predicted_class

    # Set ignore labels to background
    merged_mask[merged_mask == 255] = 0
    mask = merged_mask

    if config.general_config['debug']:
        display_image(mask, name)
    else:
        save_image(mask, name, config)


def compute_iou(a, b):
    """
    Computes intersection over union on two binary numpy arrays
    :param a, b: binary numpy array
    :return: float giving IoU of two arrays
    """

    overlap = a * b
    union = a + b

    return overlap.sum() / float(union.sum())


def filter_blobs(mask, min_area=5 ** 2):
    """
    Removes small and/or elongated blobs. Extracts per-blob binary images

    :param mask: binary 2d numpy array
    :param min_area: int
    :return: list of binary numpy arrays (one per blob)
    """

    blobs = []

    # Turns binary mask into 8-connected blobs with labels starting from 1.
    labels = skimorph.label(mask, connectivity=2)

    for n in np.unique(labels):

        # Ignore background
        if n == 0:
            continue

        # Select blob
        mask_indiv = (labels == n).astype('f')

        # Ignore small blobs and  blobs without surface (e.g. lines)
        if mask_indiv.sum() > min_area and bbox_area(mask_indiv) != 0:
            blobs.append(mask_indiv)

    if len(blobs) == 0:
        blobs.append(np.zeros_like(mask))

    return blobs


def fuse_blobs(blobs):
    """
    Fuses multiple blobs to a single one

    :param blobs: list of arrays all representing different blobs
    :return: single array containing all blobs
    """

    all_blobs = np.zeros_like(blobs[0])

    for blob in blobs:
        all_blobs = np.where(blob > 0, blob, all_blobs)

    return all_blobs


def bbox_area(mask):
    """
    Calculates the size of the bounding box.
    If this size is small, the blob is too stretched and removed later.
    :param mask: 2d array with a binary mask of a single blob
    :return: square surface of said blob
    """
    x_min = np.where(mask)[0].min()
    x_max = np.where(mask)[0].max()
    y_min = np.where(mask)[1].min()
    y_max = np.where(mask)[1].max()
    return (x_max - x_min) * (y_max - y_min)


def display_image(mask, name):
    """
    Displays an image, converts the pixel labels to more distinctive PASCAL colors.
    :param mask: 2d array with non-binary mask
    :param name: image id
    """

    # Utils -> to color
    mask = pascal_index_to_color(mask)

    # Plot image
    img = Image.fromarray(mask, mode='RGB')
    img.show(title=name)  # TODO: Does not display name correctly


def save_image(mask, name, config):
    """
    Saves image in greyscale, using labels as pixel values.
    :param mask: 2d array with non-binary mask
    :param name: image id
    :param config: config settings object
    """

    img = Image.fromarray(mask.astype(np.uint8), mode='L')
    name = name + '.png'
    img.save(osp.join(config.locations['root_folder'],
                      config.locations['guide_storage'],
                      str(name)))


def load_raw_attention(config, image_shape, file_names, file):
    """
    Loads all attention .npy files and creates background attention
    :param config: config settings object
    :param image_shape: shape of current image
    :param file_names: all file names in folder
    :param file: current file name
    :return: single numpy array with all attention including background (H x W x Classes)
    """

    original_attention = np.ones((image_shape[0], image_shape[1], 1))

    for mask_file in file_names:
        if file in mask_file:
            attention_location = osp.join(config.locations['root_folder'],
                                          config.locations['unbin_attention_storage'],
                                          mask_file)

            current_attention = np.load(attention_location)
            assert current_attention.shape == original_attention.shape[:2]

            original_attention[:,:,0] = original_attention[:,:,0] - current_attention

            current_attention = np.expand_dims(current_attention, axis=2)
            original_attention = np.concatenate((original_attention, current_attention), axis=2)

    return original_attention


def main(config):
    """
    Main loop, loads the images from the correct folders and calls combine mask function for each image.
    Images are loaded from .png and converted to numpy. Fusion is computed on the numpy arrays.
    The fused image is stored as .png
    :param config: config settings object
    """

    text_file = open(osp.join(config.locations['root_folder'], config.general_config['model_name'] + ".txt"), "a")
    print("-----", file=text_file)
    print('Creating Guides..', file=text_file)

    file_names = []

    train_image_file = osp.join(config.locations['root_folder'], config.locations['train_labels'])
    val_image_file = osp.join(config.locations['root_folder'], config.locations['val_labels'])

    with open(train_image_file, 'r') as file:
        for image_id in file:
            file_names.append(image_id.rstrip())
    with open(val_image_file, 'r') as file:
        for image_id in file:
            file_names.append(image_id.rstrip())

    mask_files = os.listdir(osp.join(config.locations['root_folder'],
                                     config.locations['attention_storage']))

    raw_mask_files = os.listdir(osp.join(config.locations['root_folder'],
                                         config.locations['unbin_attention_storage']))

    count = 0
    for file in tqdm(file_names):

        # Load saliency masks
        saliency_location = osp.join(config.locations['root_folder'],
                                     config.locations['saliency_storage'],
                                     file + '.png')

        saliency_image = Image.open(saliency_location)
        saliency_image = np.asarray(saliency_image, dtype=np.uint8) / 255

        # Load raw attention, which is used in CRF
        original_attention = load_raw_attention(config, saliency_image.shape, raw_mask_files, file)

        # Load attention masks and class labels -> List per image
        current_image_masks = []
        current_image_predictions = []
        for mask_file in mask_files:
            if file in mask_file:

                attention_location = osp.join(config.locations['root_folder'],
                                              config.locations['attention_storage'],
                                              mask_file)

                attention_mask = Image.open(attention_location)
                attention_mask = np.asarray(attention_mask, dtype=np.uint8) / 255
                current_image_masks.append(attention_mask)

                # Get class label from file name
                label = mask_file.split('.')[0].split('_')[-1]
                current_image_predictions.append(int(label))

        combine_masks(saliency_image, current_image_masks, original_attention, current_image_predictions, file, config)

        count = count + 1

        if count % 100 == 0:
            print(str(count) + "/" + str(len(file_names)) + " done", file=text_file)

        if config.general_config['debug'] and count > 10:
            quit()


if __name__ == '__main__':
    from config import Config

    test_config = Config()
    test_config.general_config['debug'] = True
    test_config.guide_config['fusion'] = 'g2'
    main(test_config)
