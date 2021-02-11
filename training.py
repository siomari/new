import torch
from adataset import train_val_dataset, CustomDataset
from custom_transforms import  Rescale
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

device = ("cuda" if torch.cuda.is_available() else "cpu")

num_epochs = 10
learning_rate = 0.00001
batch_size = 32

transform = transforms.Compose([Rescale([1000,1000]),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


data = CustomDataset("fake/", "masks/", transform = transform)

# print(type(data))

train_set, val_set = train_val_dataset(data, val_split = 0.20)


train_loader = DataLoader(dataset = train_set, shuffle = True, batch_size = 32)
val_loader = DataLoader(dataset = val_set, shuffle = False, batch_size = 32)


# models
# cross entropy loss
# adam optim