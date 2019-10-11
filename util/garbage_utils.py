from xml.etree import ElementTree
import torchvision.transforms.functional as TF


def garbage_labels_from_annotation(annotation_file):
    """
    Returns class indices of an image given the annotation file.

    :param annotation_file:
    :return: list of label ids per annotation file
    """

    classes = ["garbage_bag", "container_small", "cardboard"]
    class_ids = [0] * 3

    root = ElementTree.parse(annotation_file).getroot()
    for image_object in root.findall('object'):
        value = image_object.find('name').text
        if value in classes:
            class_ids[classes.index(value)] = 1

    return class_ids


def image_to_pascal_size(image):
    """
    Rescales an image to match Pascal shape

    :param image: PIL Image of random size
    :return: PIL Image in Pascal size ( 500 on largest axis )
    """
    dims = image.size
    # Transform image to Pascal size first
    max_val = max(dims)
    max_id = dims.index(max_val)
    scale = 500 / max_val
    min_out = round(min(dims) * scale)
    if max_id == 0:
        image = TF.resize(image, (min_out, 500))
    else:
        image = TF.resize(image, (500, min_out))
    return image
