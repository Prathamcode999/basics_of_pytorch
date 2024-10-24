import torch

w  = torch.tensor(3.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)

def forward(x):
    y=w*x+b
    return y

test_1 = torch.tensor(2)
test_2 = torch.tensor([1,2])
print(forward(test_1))
print(forward(test_2))