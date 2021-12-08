from pytorchCodePractice import *


# building the loss function

# todo that we need to reshape the tensor and turn it into vector, the way we do that is with the view method
# we pass to the view method -1 for whatever the number of the rows is and 28*28 the number of the columns of the image vector 
train_x = torch.cat([stacked_threes, stacked_sevens]).view(-1,28*28)

# for the labels we will have a 1 for each of the 3s and a 0 for each of the 7s that will give us a vector
# we add the method unsqueeze to add an additional unit to the tensor and turn it into a matrix
# that will turn it from a vector of torch.Size([12396])) rows to a matrix of torch.Size([12396, 1])) rows and 1 column
train_y = tensor([1]*len(threes) + [0]*len(sevens)).unsqueeze(1)

# creating a dataset that returns a tuple
dset = list(zip(train_x,train_y))

# validation dataset
valid_x = torch.cat([valid_3_tens, valid_7_tens]).view(-1, 28*28)
valid_y = tensor([1]*len(valid_3_tens) + [0]*len(valid_7_tens)).unsqueeze(1)
valid_dset = list(zip(valid_x,valid_y))