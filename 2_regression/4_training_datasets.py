import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
########################################
#creation of dataset


X = torch.randn(100,1) * 10 #100 rows , 1 columns randomly generated
y = X + torch.randn(100,1)*3
plt.plot(X.numpy(), y.numpy(), 'o') #the datapoints will be shown as o
plt.ylabel('y')
plt.xlabel('x')
plt.show()

#################################################################
class LR(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size,output_size)

    def forward(self,x):
        prediction = self.linear(x)
        return prediction

torch.manual_seed(1)
model = LR(1,1)
print(model)
########################################################

# [w,b] = model.parameters() 
# # print(w,b) #print to know the elements positions then
# w1 = w[0][0].item() #since its a matrix, can you find it with .type method. 
# b1 = b[0].item() #since its scalar, .item() converts from tensors to normal python numbers

#converting the above code into function:

[w,b] = model.parameters() 
def get_params():
    return (w[0][0].item(), b[0].item())

##############################################################

def plot_fit(title):
    plt.title = title 
    w1, b1 = get_params()
    x1 = np.array([-30, 30]) #we have defined the range by seeing our dataset in matplot
    y1 = w1*x1 + b1
    plt.plot(x1,y1, 'r')
    plt.scatter(X, y)
    plt.show()


print(plot_fit('inital model'))

####################################################################
# summary till now
# created a dataset -> created a model -> got the parameter of the model -> used the random parameter to scale it to x axis of orginal data and find the slope 

# now we will use the gradient descent to improve the accuracy 
####################################################################
#we will use gradient descent to reduce the loss function
#
# we find the derivative of the loss fuction to find the slope which will lead to least error then subtract the current weight from the gradient revcieved weight.
# to move towards the least error we must move slowly towards the slope so we multiply it by learning rate
#######################################################
#training and optimization

criterion = nn.MSELoss() # loads loss function in criterion
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01) # stochastic gradient descent, lr is learning rate, updates the model parameters

epochs = 75
losses = [] # store the loss value at each epochs
for i in range(epochs):
    y_pred = model.forward(X) # feeds the input data to get the predeicted value
    loss = criterion(y_pred, y) #calcualtes the loss between perdicted and original target values y
    print("epochs: ", i, "loss: ", loss.item())

    losses.append(loss.item()) #store the loss item in losses
    optimizer.zero_grad() # clears the privous gradient so that in total they all dont accumulate together but separately.
    loss.backward() #computes the gradient of the loss function wrt model parameter using back propogration
    optimizer.step() #updates the model parameters based on computed gradients and learning rate.

def plot_train():
    plt.plot(range(epochs), losses)
    plt.ylabel('Loss')
    plt.xlabel('epoch')
    plt.show()

plot_train()

plot_fit("trained model")