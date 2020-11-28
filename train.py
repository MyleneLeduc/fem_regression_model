from data import FemDataset
from model import StressModel
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import torch
import pdb

df_train = FemDataset('5184doe.csv')

X_train = df_train.X
y_train = df_train.y

mymodel = StressModel(input_features=9, hidden_layer1=25, hidden_layer2=30, output_features=4, p=0.4)

criterion = torch.nn.MSELoss() 
optimizer = torch.optim.Adam(mymodel.parameters(), lr=0.1, weight_decay=1e-2)

epochs = 100
losses = []

for i in range(epochs):
    mymodel.train()
    inputs = Variable(X_train)
    labels = Variable(y_train)
    outputs = mymodel(inputs)
    
    loss = criterion(outputs, labels)
    losses.append(loss)
    print(f'epoch: {i:2}  loss: {loss.item():10.8f}')
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

mymodel.eval()

### Validating and testing the model ###

df_test = FemDataset('40semi-randoms.csv')

X_test = df_test.X
y_test = df_test.y


inputs = Variable(X_test)
preds = mymodel(inputs)

erreur = criterion(preds, y_test)
print("Nonlinear regression : Erreur : {:.4f}".format(erreur))
pdb.set_trace()
