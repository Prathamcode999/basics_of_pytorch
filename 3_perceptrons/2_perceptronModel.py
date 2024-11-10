import torch
import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np
import torch.nn as nn


n_pts = 100
centers = [[-0.5,0.2],[0.2,-0.5]]
X,y = datasets.make_blobs(n_samples=n_pts, random_state=123, centers=centers, cluster_std=0.4) 

def scatter_plot():
    plt.scatter(X[y==0,0], X[y==0, 1])
    plt.scatter(X[y==1,0], X[y==1, 1]) 
    plt.show()

#we are converting the data's into tensor for training
x_data = torch.tensor(X, dtype=torch.float32)
y_data = torch.tensor(y, dtype=torch.float32).reshape(100,1)


class Model(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size,output_size)
    
    def forward(self, x):
        predict = torch.sigmoid(self.linear(x)) #instead of normally creating linear, we are passing it through sigmoid function
        return predict
    
    def predict(self,x):
        predict = self.forward(x)
        if predict >= 0.5:
            return 1
        else:
            return 0

torch.manual_seed(1)
model1 = Model(2,1)
w,b = model1.parameters()
print (f"weight: {w} , bias: {b}")

w1,w2 = w.view(2).detach().numpy() #normally 2d tensor cannot be plotted so we divide the 2d tensors to w1 and w2 making it scalar
b1 = b[0].detach().numpy()

def plot_fit(title):
    plt.title = title
    x1 = np.array([-2.0,2.0])
    x2 = (w1*x1 + b1)/-w2
    plt.plot(x1,x2,'r')
    scatter_plot()
    
plot_fit('before training')

############################################################################
# training data 
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model1.parameters(), lr=0.02)


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
###############################################################################
w1,w2 = model1.linear.weight[0].detach().numpy() #linear is the nn.linear layer which we have used in model for taking i/o size so we called the linear layer
 #model1.linear.weight[0] accesses the first row of the weight matrix. 
 # Since this is a binary classification task (with 1 output neuron), there is only one row, 
 # which represents the weights for the two input features. Thus, model1.linear.weight[0] gives you the
 #  weights w1 and w2 for the two features.
b1 = model1.linear.bias[0].detach().numpy()
# The .bias attribute stores the bias term of the linear layer. For a single output neuron, this is a scalar.
# summary:-
# w1 and w2 are the weights for the two input features, converted to NumPy arrays for plotting.
# b1 is the bias term, also converted to a NumPy array.
# These are then used to plot the decision boundary after training.

plot_fit('after training')

#########################################
#testing
point1 = torch.tensor([1.0,-1.0])# we allot two random points for testing
point2= torch.tensor([-1.0,1.0])
def plotting_test():
    plt.plot(point1.numpy()[0], point1.numpy()[1], 'ro')
    plt.plot(point2.numpy()[0], point2.numpy()[1], 'ko')
    plot_fit("after testing")

plotting_test()

print("red point positive probability = {}".format(model1.forward(point1).item()))
print("black point positive probability = {}".format(model1.forward(point2).item())) # return the probabilty of the points

print("red point in class = {}".format(model1.predict(point1)))
print("black point in class = {}".format(model1.predict(point2)))# return in which class they belong, predict is initalized in model class