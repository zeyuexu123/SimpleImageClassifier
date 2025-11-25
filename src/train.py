import torch
import torch.nn as nn
import torch.optim as optim

from data import get_train_loader
from model import Net
from config import LR, MOMENTUM, EPOCHS, DEVICE, MODEL_PATH

def train():
    net = Net().to(DEVICE)
    trainloader = get_train_loader()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=LR, momentum=MOMENTUM)

    for epoch in range(EPOCHS):
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            loss = criterion(net(inputs), labels)
            loss.backward()
            optimizer.step()
            
        print(f"Epoch: {epoch+1}  Loss: {loss.item():.3f}")

    print('Finished Training')
    torch.save(net.state_dict(), MODEL_PATH)