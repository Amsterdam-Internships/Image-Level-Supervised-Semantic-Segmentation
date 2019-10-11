"""
Applies dense Conditional Random Fields
- Used in generating guides when 2 or more attention blobs overlap with single saliency blob
- Used for post-processing of segmentations

Code adapted from: https://github.com/speedinghzl/Pytorch-Deeplab
"""
import numpy as np
import pydensecrf.densecrf as dense_crf
from pydensecrf.utils import unary_from_labels


def apply_dense_crf(original_image, unary, involved_classes, max_iter=10, gt_prob=0.9, stage='guide'):
    """
    Applies dense conditional random fields

    :param original_image: numpy array containing RGB image (H x W x ColorChannels)
    :param unary: 3d numpy array containing network activations (H x W x Classes). Positive log probabilities
    :param involved_classes: list containing class ids of classes involved
    :param max_iter: maximum iterations of iterative CRF process
    :param gt_prob: certainty of the ground truth labels. Must be between 0 and 1
    :param stage: indicates stage in which CRF is called, which changes hyperparameters
    :return: 3d numpy array with class probabilities per pixel (H x W x Classes)
    """
    assert 0 < gt_prob < 1
    if stage == 'guide':
        assert (original_image.shape[:2] == unary.shape[:2])

    height, width = original_image.shape[:2]
    n_labels = len(involved_classes)

    # initialize CRF
    crf = dense_crf.DenseCRF2D(width, height, n_labels)

    if stage == 'deeplab':
        # set unary potentials based on labels
        unary = unary_from_labels(unary, n_labels, gt_prob=gt_prob, zero_unsure=False)
        crf.setUnaryEnergy(unary)
    else:
        # set unary potentials (to negative log probabilities, flatten and assure right format)
        crf.setUnaryEnergy(-unary.transpose((2, 0, 1)).reshape((n_labels, -1)).copy(order='C').astype('float32'))

    # Values taken as in previous research
    crf.addPairwiseGaussian(sxy=3, compat=3)
    if stage == 'deeplab':
        crf.addPairwiseBilateral(sxy=121, srgb=5, rgbim=original_image, compat=4)
    else:
        crf.addPairwiseBilateral(sxy=80, srgb=13, rgbim=original_image, compat=10)

    # run inference
    prediction = np.array(crf.inference(max_iter)).reshape((n_labels, height, width)).transpose((1, 2, 0))

    return involved_classes[prediction.argmax(2)]
