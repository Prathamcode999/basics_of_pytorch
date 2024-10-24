import torch
from torch.nn import Linear


torch.manual_seed(1) # generates random number, 1 is the seed means wherever you use 1 it will generate the same random number. you can set seed value to any number
model = Linear(in_features=1, out_features=2) # it tells for every output there is a single input
print(f" bias:{model.bias}, weight: {model.weight}")
# y= w * x + b , w and b are generated randomly with the help of manualseed.
x = torch.tensor([[2.0], [3.0]]) #!! we have passed only one value
# we will pass only 1 value because in_features=1 otherwise it will give error
print(model(x)) #we pass our inputs here
