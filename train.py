from data import FemDataset
from model import StressModel
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import torch
import pdb

df_train = FemDataset('5184doe.csv')
df_test = FemDataset('40semi-randoms.csv')

### Training the model ###

mymodel = StressModel(input_features=9, output_features=4)
criterion = torch.nn.MSELoss() 
optimizer = torch.optim.Adam(mymodel.parameters(), lr=0.1)
train_loader = DataLoader(df_train, batch_size=32, shuffle=True)
test_loader = DataLoader(df_test, batch_size=32, shuffle=False)

epochs = 10

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
