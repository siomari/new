import torch
from adataset import train_val_dataset, CustomDataset
from torch.utils.data import Dataset, DataLoader
import network
import torch.nn as nn
from skimage import io, data

learning_rate = 0.00001
batch_size = 16

data = CustomDataset("fake/", "masks/")

train_set, val_set = train_val_dataset(data, val_split = 0.20)

train_loader = DataLoader(dataset = train_set, shuffle = True, batch_size = batch_size)
val_loader = DataLoader(dataset = val_set, shuffle = False, batch_size = batch_size)

# model1 = network.LSTM_CNN()
model2 = network.Encoder()

model = network.Decoder(model2)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)


for epoch in range(10):
    for i, data in enumerate(train_loader):

        image, mask = data

        optimizer.zero_grad()
        
        output = model(image)
        loss = criterion(output, mask)
        loss.backward()
        optimizer.step()


        if i % 22 == 21:
            print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, loss.item()))

print("Finished Training")


model.eval()

with torch.no_grad():
    for t in val_loader:
        test_image, y_test = t
        y_pred = model(test_image)
        print("1")
        io.show(y_test.numpy())
        print("2")
        io.show(y_pred.numpy())
        break
