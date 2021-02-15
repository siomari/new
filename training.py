import torch
from adataset import train_val_dataset, CustomDataset
from torch.utils.data import Dataset, DataLoader
import network
import torch.nn as nn

# if torch.cuda.is_available():
#     device = "cuda"
# else: device = "cpu"

# print(device)

learning_rate = 0.00001
batch_size = 64

data = CustomDataset("fake/", "masks/")

train_set, val_set = train_val_dataset(data, val_split = 0.20)

train_loader = DataLoader(dataset = train_set, shuffle = True, batch_size = batch_size)
val_loader = DataLoader(dataset = val_set, shuffle = False, batch_size = batch_size)

model1 = network.LSTM_CNN(3, 64, 2)
model2 = network.Encoder()

model = network.Decoder(model1, model2)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)


for epoch in range(1):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):

        image, mask = data

        optimizer.zero_grad()

        
        output = model(image,image)
        loss = criterion(output, mask)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

