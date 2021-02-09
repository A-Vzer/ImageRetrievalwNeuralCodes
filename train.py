from model import NeuralCodes
from data import execute, TinyImageNet
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
import pickle
import h5py
import numpy as np
from tqdm import tqdm

with h5py.File('tinyImgNet.h5', "r") as f:
    a_group_key = list(f.keys())[0]
    # Get the data
    train_x = torch.from_numpy(f['train_x'][()])
    train_y = torch.from_numpy(f['train_y'][()])
    test_x = torch.from_numpy(f['test_x'][()])
    test_y = torch.from_numpy(f['test_y'][()])
    val_x = torch.from_numpy(f['val_x'][()])
    val_y = torch.from_numpy(f['val_y'][()])
    ood_x = torch.from_numpy(f['ood_x'][()])
    ood_y = torch.from_numpy(f['ood_y'][()])


def train():
    print(f"Epoch: {epoch + 1}")
    model.train()
    print("Training...")
    accuracies = 0
    losses = 0
    for idx, (image, target) in enumerate(train_loader):
        optimizer.zero_grad()
        target = target.cuda()
        image = image.cuda()
        output = model(image)
        loss = criterion(output, target.long())
        losses += loss.detach().item()
        acc = (target == output.argmax(-1)).detach().cpu().numpy()
        accuracies += sum(acc) / len(acc)
        loss.backward()
        optimizer.step()
        if idx % log == 0 and idx != 0:
            print(f"Avg train loss: {losses / log}")
            print(f"Avg train accuracy: {accuracies / log}")
            losses = 0
            accuracies = 0


def evaluate():
    eval_losses = 0
    model.eval()
    print("Validating...")
    with torch.no_grad():
        for idx, (word, target) in enumerate(val_loader):
            output = model(word.cuda())
            loss = criterion(output, target.cuda().long())
            eval_losses += loss.detach().item()
    eval_losses = eval_losses / len(val_loader)
    print(f"Avg evaluation loss epoch {epoch + 1}: {eval_losses}")
    torch.save({'model': model.state_dict(), 'EvalLoss': eval_losses}, 'saves/model.pt')


def test_evaluate():
    with torch.no_grad():
        test_losses = 0
        for idx, (word, target) in enumerate(test_loader):
            output = model(word.cuda())
            loss = criterion(output, target.cuda().long())
            test_losses += loss.detach().item()
        test_losses = test_losses / len(test_loader)
        print(f"Avg test loss: {test_losses}")


train_data = TinyImageNet(x=train_x, y=train_y)
val_data = TinyImageNet(x=val_x, y=val_y)
test_data = TinyImageNet(x=test_x, y=test_y)
bs = 128
bs_eval = 128
train_loader = DataLoader(train_data, batch_size=bs, shuffle=True)
val_loader = DataLoader(val_data, batch_size=bs_eval, shuffle=False)
test_loader = DataLoader(test_data, batch_size=bs_eval, shuffle=False)
model = NeuralCodes(code_length=2048, classes=190).cuda()
optimizer = optim.Adam(model.parameters(), lr=0.0001, amsgrad=True)
criterion = nn.NLLLoss()
epochs = 25
losses = []
log = 100
torch.cuda.empty_cache()

if __name__ == '__main__':
    for epoch in range(epochs):
        train()
        evaluate()
    test_evaluate()
