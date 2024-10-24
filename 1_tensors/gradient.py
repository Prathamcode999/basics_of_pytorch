import torch
#############################
#derivative

# x = torch.tensor(2.0, requires_grad= True) # 2 is x's value, required_grad tells to compute all parameters so later we can computer gradient
# y = 9*x**4 + 2*x**3 + 3*x**2 + 6*x + 1
# y.backward() # it finds the gradient dy/dx
# print(x.grad) # it prints gradient of y wrt to x

######################
#partial derivative

x = torch.tensor(1.0, requires_grad= True)
z = torch.tensor(2.0, requires_grad= True)
y = x**2 + z**3
y.backward() #computes partial differtiation

print(z.grad)# gives dy/dz
print(x.grad)# gives dy/dx

