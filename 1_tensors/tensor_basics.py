import torch
import numpy as np

# v = torch.tensor([1,2,4]) #one dimensional tensor
# print(v.dtype) #tells the tensor type
# print(v) #printing the tensor
# print(v[2]) #accessing the value using index


#######################
# #slicing the tensor
# v = torch.tensor([1,2,7,5,3,5,])
# print(v[2:5]) #slices but 2 is included and 5 is not

#########################
# f=torch.FloatTensor([3,3,4,1,2,6,5])
# print(f.dtype) # will tell that f is a float type
# print(f.size()) # will return the number of elements inside the tensor

#########################
# v = torch.tensor([1,2,3,4,4,5])
# v_view = v.view(3,2) # will allow you to view the tensor in different format, same as reshape in numpy
# v_auto_view=v.view(2,-1) #-1 will automatically understand how many row or colums to fill when other is given
# print(v_view)
# print(v_auto_view)


###################################
#converting numpy array to tensors
arr = np.array([1,2,3,7,5,4,3])
tensor_cnv = torch.from_numpy(arr)
print(tensor_cnv)
print(tensor_cnv.type())

#converting back to numpy
numpy_cnv = tensor_cnv.numpy()
print(numpy_cnv)
