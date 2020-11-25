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

mymodel = StressModel(input_features=9, output_features=4)

criterion = torch.nn.MSELoss() 
optimizer = torch.optim.Adam(mymodel.parameters(), lr=0.1)

epochs = 100
losses = []

for i in range(epochs):
    inputs = Variable(X_train)
    labels = Variable(y_train)
    outputs = mymodel(inputs)
    
    loss = criterion(outputs, labels)
    losses.append(loss)
    print(f'epoch: {i:2}  loss: {loss.item():10.8f}')
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

### Validating and testing the model ###

df_test = FemDataset('40semi-randoms.csv')

X_test = df_test.X
y_test = df_test.y


inputs = Variable(X_test)
preds = mymodel(inputs)

erreur_relative = abs((y_test-preds)/y_test)
erreur = erreur_relative.detach().numpy()
erreur = np.mean(erreur)


pdb.set_trace()
