"""
PiCANet implementation in PyTorch

Code adapted from: https://github.com/Ugness/PiCANet-Implementation

"""
import os
import os.path as osp

import torch
from torch.nn.functional import interpolate
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
from tqdm import tqdm

from saliency.network import Unet
from saliency.dataset import CustomDataset
from config import Config


def main(config):

    text_file = open(osp.join(config.locations['root_folder'], config.general_config['model_name'] + ".txt"), "a")
    print("-----", file=text_file)
    print('Creating Saliency..', file=text_file)

    custom_dataset = CustomDataset(root_dir=osp.join(config.locations['data_root_folder'],
                                                     config.locations['image_folder']))
    data_loader = DataLoader(custom_dataset, batch_size=1, shuffle=False)

    device = torch.device(config.general_config['device'])

    state_dict = torch.load(osp.join(config.locations['root_folder'],
                                     config.locations['model_storage'],
                                     'picanet.ckpt'),
                            map_location=config.general_config['device'])
    model = Unet().to(device)
    model.load_state_dict(state_dict)

    model.eval()

    for i, batch in enumerate(tqdm(data_loader)):
        img, (width, height), img_name = batch

        # Output name in .png instead of .jpg
        save_name = img_name[0].split('.')[0] + '.png'

        img = img.to(device)
        with torch.no_grad():
            prediction, _ = model(img)
        prediction = prediction[5].data
        prediction.requires_grad_(False)

        # Normalization (for better performance on images with no foreground objects)
        prediction = prediction - prediction.min()
        prediction = prediction / prediction.max()

        # Interpolation to original image size
        prediction = interpolate(prediction, size=(height, width))

        # Binarization
        saliency_mask = torch.where(prediction > config.guide_config['saliency_bin_threshold'],
                                    torch.ones_like(prediction),
                                    torch.zeros_like(prediction))
        saliency_mask = saliency_mask.squeeze().detach().cpu().numpy() * 255
        img = Image.fromarray(saliency_mask.astype(np.uint8), mode='L')
        img.save(osp.join(config.locations['root_folder'],
                          config.locations['saliency_storage'],
                          save_name))

        if i % 100 == 0:
            print(str(i) + "/" + str(len(data_loader)) + " done", file=text_file)


if __name__ == '__main__':

    test_config = Config()

    test_config.guide_config['saliency_bin_threshold'] = 0.5
    test_config.locations['model_storage'] = '../../data'

    main(test_config)
