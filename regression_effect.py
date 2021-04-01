import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"
x1= np.random.uniform(0,0.3,10000)
x2 = np.random.uniform(0.9,1,1000)

x = np.concatenate((x1,x2))

y = np.sin(x)
ind = np.random.choice(range(len(x)),size=int(0.8*len(x)) ,replace=False)
train_x = x[ind]
train_y = y[ind]

ind_test = set(range(len(x))).difference(set(ind))
test_x = x[[i for i in ind_test]]
test_y = y[[i for i in ind_test]]

train_x = torch.as_tensor(train_x)
train_y = torch.as_tensor(train_y)
test_x = torch.as_tensor(test_x)
test_y = torch.as_tensor(test_y)


class net(nn.Module):
    def __init__(self):
        super(net,self).__init__()
        self.fc1 = nn.Linear(1,50)
        self.fc2 = nn.Linear(50,1)
        
    def forward(self,x):
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.fc2(x)
        x = nn.functional.sigmoid(x)
        
        return x
    

net1 = net()

optimizer = torch.optim.Adam(net1.parameters(),lr=0.0001)
loss = torch.nn.MSELoss()
for i in range(10000):
    ind = np.random.choice(range(len(train_x)),1000,False)
    x = train_x[ind]
    y = train_y[ind]
    x = torch.unsqueeze(x,dim=1)
    x = x.float()
    y = y.float()
    optimizer.zero_grad()
    out = net1(x)
    y = torch.unsqueeze(y, 1)
    loss1 = loss(out, y)
    loss1.backward()    
    optimizer.step()
    print(f"training loss: {loss1}")
    
plt.scatter(x.detach().numpy(),out.detach().numpy(),label='pred')
plt.scatter(x.detach().numpy(),y.detach().numpy(),label='truth')
    
