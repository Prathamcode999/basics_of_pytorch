import torch
import matplotlib.pyplot as plt
from sklearn import datasets
import torch.nn as nn
from sklearn import datasets
import numpy as np


# dataset creation
n_pts = 500
X,y = datasets.make_circles(n_samples=n_pts, random_state=123, noise=0.1, factor=0.2) 

def scatter_plot():
    plt.scatter(X[y==0,0], X[y==0, 1])
    plt.scatter(X[y==1,0], X[y==1, 1]) 
    plt.show()

#we are converting the data's into tensor for training
x_data = torch.tensor(X, dtype=torch.float32)
y_data = torch.tensor(y, dtype=torch.float32).reshape(500,1)


# model initialization
class Model(nn.Module):
    def __init__(self, input_size, H1, output_size): #h1 is the hidden layer
        super().__init__()
        self.linear = nn.Linear(input_size,H1)#first we connect the input to hidden layer
        self.linear2 = nn.Linear(H1,output_size)
    
    def forward(self, x):
        x = torch.sigmoid(self.linear(x)) #calc sigmoid from input to hidden layer
        x = torch.sigmoid(self.linear2(x)) #calc sigmoid from hidden to output layer
        return x
    
    def predict(self,x):
        predict = self.forward(x)
        if predict >= 0.5:
            return 1
        else:
            return 0


torch.manual_seed(2)
model1 = Model(2,4,1)
print(list(model1.parameters()))

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model1.parameters(), lr=0.1)  #adam is combination of adamgrad and RMSprop

epochs = 1000
losses = []

for i in range(epochs):
    y_pred = model1.forward(x_data)
    loss = criterion(y_pred, y_data)
    if i % 100 == 0:
        print(f" epoch: {i} , loss:{loss.item()}") #for printing every 100 epochs

    losses.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def plot_train():
    plt.plot(range(epochs), losses)
    plt.ylabel('Loss')
    plt.xlabel('epoch')
    plt.show()

plot_train()


def plot_decision_boundary(X,y):
    x_span = np.linspace(min(X[:,0]),max(X[:, 0]))
    y_span = np.linspace(min(X[:,1]), max(X[:,1]))
    xx, yy = np.meshgrid(x_span, y_span)
    grid = torch.Tensor(np.c_[xx.ravel(), yy.ravel()])
    pred_func = model1.forward(grid)
    z = pred_func.view(xx.shape).detach().numpy()
    plt.contourf(xx,yy,z)
    scatter_plot()
    plt.show()
    

plot_decision_boundary(X,y)
