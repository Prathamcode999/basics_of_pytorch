import torch
import torch.nn as nn

class LR (nn.Module): #subclass inhertiting nn.module
    def __init__(self, input_size, output_size):
        super().__init__() # we have not called anyting because the parent class(nn.Module) does'nt need any parameter
        self.linear = nn.Linear(input_size, output_size) # we have passed the i/o sizes in liner variable with the help of nn.linear called with the help of super() from parent class
        # the above line also automatically assigns the weights and the bias
    def forward(self,x):
        predict = self.linear(x) # passes the x argument in linear layer created in constructor(self.linear)
        return predict



torch.manual_seed(1)
model = LR(1,1)

[w,b]=model.parameters()
print(w,b) #prints the model's weights and bias

x = torch.tensor([[1.0],[3.0]])
print(model.forward(x))
