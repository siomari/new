from utilities import data_preprocess
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
from skimage import io
from sklearn.model_selection import train_test_split
from skimage import io, transform
from torchvision import transforms, utils



def train_val_dataset(dataset, val_split = None):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    train_data = Subset(dataset, train_idx)
    val_data = Subset(dataset, val_idx)
    return train_data, val_data


class CustomDataset(Dataset):
    def __init__(self, images_path, masks_path):
        self.images, self.masks = data_preprocess(images_path, masks_path)
    
    def __len__(self):
        return len(self.images)

    def transform(self, image, mask, output_size):

        # Resize
        nheight, nwidth = output_size

        if image.shape[2] > 3:
            image = image[:,:,:3]

        img = transform.resize(image, (nheight, nwidth))

        if len(mask.shape) > 2:
            mask = mask[:,:,0]

        msk = transform.resize(mask, (nheight, nwidth))

        img = transforms.functional.to_tensor(img)
        msk = transforms.functional.to_tensor(msk)

        return img, msk

    def __getitem__(self,idx):

        img_name = self.images[idx]
        mask_name = self.masks[idx]

        image = io.imread(img_name)
        mask = io.imread(mask_name)
        x, y = self.transform(image, mask, [1000,1000])
        return x, y

