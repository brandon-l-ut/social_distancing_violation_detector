## Trainer file for Disnet. 

import json
import torch
import torch.nn as nn
import torch.optim as optim

from disnet import Disnet

epochs = 101
b_size = 1024
## Train test split
train_size = 6000
eval_size = 1481

if __name__ == '__main__':
    with open('data.json') as f:
        Data = json.load(f)
        
    Train = Data['train']
    Eval = Data['eval']

    train_x = torch.tensor(Train['X'])
    train_labels = torch.tensor(Train['Y'])
    train_dataset = torch.utils.data.TensorDataset(train_x, train_labels)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=train_size, shuffle=True)

    eval_x = torch.tensor(Eval['X'])
    eval_labels = torch.tensor(Eval['Y'])
    eval_dataset = torch.utils.data.TensorDataset(eval_x, eval_labels)
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=eval_size, shuffle=False)

    model = Disnet()

    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr = 0.001, betas=(0.9, 0.999), eps=1e-08)
    #optimizer = optim.SGD(model.parameters(), lr = 0.001, momentum = 0.9)

    for epoch in range(epochs):
        epoch_loss = 0
        for i, (x, labels) in enumerate(train_dataloader, 0):
            optimizer.zero_grad()
            outputs = model(x, False)
            loss = criterion(outputs, labels)
            loss.backward()
            epoch_loss+= loss.item()
            optimizer.step()

        def valid_loss(): 
            val_loss = 0
            with torch.no_grad():
                for inputs, labels in eval_dataloader:
                    outputs = model(inputs, False)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
            
            print("Val L1: {}".format(val_loss))

        if epoch % 10 == 0:
            valid_loss()
            print("Epoch {} L1 train loss: {}".format(epoch, epoch_loss))

    torch.save(model.state_dict(), "Disnet.pth")
