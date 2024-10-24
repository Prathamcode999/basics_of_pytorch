import torch

x = torch.arange(18).view(3,2,3)# 3- no of blocks, 2-rows, 3-columns, 3*3*4 = 36. arange has created total 36 elements ranging from 0 to 36
print(x)

###########################################
#accessing 3d tensors
# !! silly mistake :: indexing starts from 0
print(x[1,1,1]) #element from 1 block 1row 1 column

#########################################
#slicing the elements
