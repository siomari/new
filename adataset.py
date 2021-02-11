from utilities import data_preprocess
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
from skimage import io, transform
from sklearn.model_selection import train_test_split


def train_val_dataset(dataset, val_split = 0.20):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets={}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    return datasets

class CustomDataset(Dataset):
    def __init__(self, images_path, masks_path, transform = None):
        self.images, self.masks = data_preprocess(images_path, masks_path)
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self,idx):

        img_name = self.images[idx]
        mask_name = self.masks[idx]

        image = io.imread(img_name)
        mask = io.imread(mask_name)
        s = {'image': image, 'mask': mask}
        return s

