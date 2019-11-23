###################################
## Getting Familier with Pytorch
## https://pytorch.org/tutorials
#####

from __future__ import print_function
import torch


x  =  torch.empty(5,3)

print (x)


y = torch.rand(5,4)

print (y)



x = torch.zeros(3,3,dtype=torch.long)

print (x)



# Generate tensor from data

x = torch.tensor([5.5,3])
print (x)


# Mutate the tensor Copy a tensor
y = torch.rand(2)

x.copy_(y)

print ('New x : {}'.format(x))


# Note
# Torch to generate
# Auto Gradients




