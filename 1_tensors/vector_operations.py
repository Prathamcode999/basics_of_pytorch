import torch
import matplotlib.pyplot as plt

# t1 = torch.tensor([1,2,3,4,4,5])
# print(t1*2) # multiplies with each element in tensor
# t2 = torch.tensor([1,6,5,3,2,4])
# dot_product = torch.dot(t1,t2) # multiplication of dot product but the size of both the tensor should be same
# print(dot_product) # same indices are multiplied then total are added


################################################
x = torch.linspace(0,10,100) # 0 is the starting point, 10 is the ending point and 5 is the evenly space or gap between them 
y = torch.exp(x) # it is nothing but e^x
plt.plot(x.numpy(),y.numpy())#matplot only works with numpy
plt.show()

################################################