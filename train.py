from data import FemDataset
from model.linear import StressModel
from model.nn import StressModel
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import torch
import pdb
import argparse

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, default='linear', help='Model to train (linear|non_linear)')
args = parser.parse_args()
# pdb.set_trace()

df_train = FemDataset('1000randoms.csv')
df_test = FemDataset('40semi-randoms.csv')

### Training the model ###

if args.model == 'linear':
    mymodel = StressModel(input_features=9, output_features=4)
elif args.model == 'non_linear':
    mymodel = StressModel(input_features=9, hidden_layer1=30, hidden_layer2=40, output_features=4, p=0.4)
else:
    raise ValueError('model not recognized')

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(mymodel.parameters(), lr=0.01, weight_decay=1e-2)
train_loader = DataLoader(df_train, batch_size=32, shuffle=True)
test_loader = DataLoader(df_test, batch_size=32, shuffle=False)

epochs = 50

for i in range(epochs):
    mymodel.train()
    train_losses = []
    for X_train, y_train in tqdm(train_loader, desc=f"epochs {i}"):
        y_pred = mymodel.forward(X_train)
        loss = criterion(y_pred, y_train)
        train_losses.append(torch.mean(loss).item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    mymodel.eval()
    val_losses = []
    for X_val, y_val in test_loader:
        with torch.no_grad():
            y_hat = mymodel.forward(X_val)
            y_val = y_val.to(dtype=torch.long)
            val_loss = torch.mean(criterion(y_hat, y_val))
            val_losses.append(val_loss)

    print(f"Training error : {np.mean(train_losses):.2f}")
    print(f"Validating error : {np.mean(val_losses):.2f}")
pdb.set_trace()
