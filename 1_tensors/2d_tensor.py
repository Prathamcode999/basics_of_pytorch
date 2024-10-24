import torch

# one_d = torch.arange(2,10,1) # it creates an 1d tensor ranging from 2 to 10 with step size 1(doesnt leave any element), if 2(leaves one element)
# two_d = one_d.view(4,2)
# print(one_d)
# print(two_d)

# print(two_d.dim()) #prints the dimension of the tensor

#accessing 2d tensor
# one_d = torch.tensor([2,3,4,5,6,7,8,9])
# two_d = one_d.view(4,2)
# print(two_d[2,1]) #output = 7

###############################################
#3d tensors
x = torch.arange(36).view(3,3,4) # 3- no of blocks, 3-rows, 4-columns, 3*3*4 = 36. arange has created total 36 elements ranging from 0 to 36
print(x)

