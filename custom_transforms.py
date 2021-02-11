import torch
from torchvision import transforms, utils
from skimage import io, transform
import adataset
from PIL import Image
import numpy as np

class Rescale(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self,data):
        image, mask = data['image'], data['mask']

        nheight, nwidth = self.output_size

        if image.shape[2] > 3:
            image = image[:,:,:3]

        img = transform.resize(image, (nheight, nwidth))

        if len(mask.shape) > 2:
            mask = mask[:,:,0]

        print(image.shape, mask.shape)        

        msk = transform.resize(mask, (nheight, nwidth))

        return {'image': img, 'mask': msk}


# class ToTensor(object):
#     def __call__(self, data):
#         image, mask = data['image'], data['mask']

#         image = image.transpose((2,0,1))
#         return {'image':torch.from_numpy(image), 
#                 'mask': torch.from_numpy(mask)}




