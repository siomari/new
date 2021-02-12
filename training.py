import torch
from adataset import train_val_dataset, CustomDataset
from torch.utils.data import Dataset, DataLoader
import network
import torch.nn as nn

if torch.cuda.is_available():
    device = "cuda"
else: device = "cpu"

print(device)

# num_epochs = 10
learning_rate = 0.00001
batch_size = 32

data = CustomDataset("fake/", "masks/")

train_set, val_set = train_val_dataset(data, val_split = 0.20)

train_loader = DataLoader(dataset = train_set, shuffle = True, batch_size = 32)
val_loader = DataLoader(dataset = val_set, shuffle = False, batch_size = 32)


model1 = network.LSTM_CNN(64, 64, 2)
model2 = network.Encoder()

# model1.load_state_dict()
# model2.load_state_dict()

model = network.Decoder(model1, model2)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)


# detail = iter(train_loader)
# i, m = detail.next()
# print(i.shape)

# i = i.view(32,3000,1000)

# print(i.shape)

for epoch in range(1):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):

        image, mask = data

        optimizer.zero_grad()

        image = image.view(32,46875,64)
        
        output = model(image,image)
        loss = criterion(output, mask)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0


# models
# cross entropy loss
# adam optim