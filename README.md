# Image-Level Supervised Semantic Segmentation with Network Attention and Saliency Priors
This repo contains the code for the master thesis of Arend van Dormalen for the MSc Artificial Intelligence at the University of Amsterdam. This master thesis was made in collaboration with the City of Amsterdam.

---
### Project Goal

In this project, image segmentations are produced from a set of image-level labeled images. This is done by training a segmentation network with pseudo-ground truth segmentations.


To obtain these pseudo-ground truths, an image classification network is trained on the dataset. 
By visualizing network attention of the ground truth classes on an image, we obtain class-specific regions.
These class-specific regions determine the region that is most discriminative for a class, but this region does not necessarily cover the full object extend.
To find these object edges, a saliency network is applied that predicts a binary foreground-background segmentation on an image. 
This foreground mask is than combined with the class-specific masks to produce a pseudo ground-truth segmentation.
Class overlaps are counted for each region in the foreground mask.
If no class regions overlap, the foreground region is deemed to be an object outside the classes of interest.
If a single class region overlaps, the entire foreground region is assigned that class label.
If more than one class region overlaps, a conditional random field is applied to divide the foreground region over the classes.

---

### Code Sections

The code is divided in numerous parts, each of which can be switched on or off in `main.py`.
Intermediate files are stored in a `tmp` folder.

* By setting `CREATE_SALIENCY` to True, a pre-trained saliency network (PiCANet) is applied on each image in the dataset and foreground-background segmentations will be stored.
* By setting `TRAIN_SEED` to True, a classification network (VGG) will be trained on the dataset.
* By setting `CREATE_ATTENTION` to True, a network attention method (Grad-CAM) is applied on the gradients of the classification network to obtain a class value per pixel. Both a binary image and the raw weights are stored per class per image.
* By setting `CREATE_GUIDE` to True, the class masks and foreground masks are combined to generate pseudo-ground truth segmentations. If multiple class regions overlap with a foreground region, the raw class weights are loaded as input in the conditional random field.
* By setting `TRAIN_DEEPLAB` to True, a segmentation network (DeepLab) is trained. A ground truth percentage can be set to determine if the network is trained with ground truth segmentations, pseudo-ground truth segmentations or a mixture of both.
If trained with ground truth segmentations only, the previous steps need not be executed.
* By setting `EVAL_DEEPLAB` to True, this segmentation network is used to segment images.

To determine performance, for both the classification and segmentation network, statistics are tracked in Tensorboard, which are stored in the `runs` folder.


---

### Storage

Storage locations can be edited in `config.py`.
Here the project root folder and the data root folder *must* be specified.
Additionally, `tmp` folders will be created with subfolders for the foreground masks, class attention masks, raw class attention values, pseudo-ground truth segmentations and predicted segmentations on the images in the test set.


---

### Results

---

### Installation instructions
