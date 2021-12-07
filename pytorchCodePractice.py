from fastai.vision.all import *
from matplotlib.pyplot import thetagrids
# fastai is an api build on top of pytorch and i will be learning it along the way with 
# pytorch through this project and the following ones


# the default installation of the library is in the main user folder where all the datasets are downloaded
# untar_data is a function for fast downloading some popular datasets after checking if that's not already done before


path = untar_data(URLs.MNIST_SAMPLE)

# assigning the BASE_PATH to where the mnist dataset is downloaded in the previous step
Path.BASE_PATH = path

# doing some systemCalls on the path object 
print(path.ls())
print((path/'train').ls())
threes = (path/'train'/'3').ls().sorted()
sevens = (path/'train'/'7').ls().sorted()

im3_path = threes[1]
im3 = Image.open(im3_path)

#im3 is of type PIL png python imaging library
print(type(im3))

# to turn the image into numbers is to turn it into array
im_array = array(im3)[4:10,4:10]
print(im_array)

# tensor is the pytorch version of a numpy array
im_tensor = tensor(im3)[4:10,4:10]
print(im_tensor)

# loop over the lest of 7s and 3s, openning the image, trun it into a tensor and then return the list
seven_tensors = [tensor(Image.open(o)) for o in sevens]
three_tensors = [tensor(Image.open(o)) for o in threes]
print(len(three_tensors),len(seven_tensors))

# show_image(three_tensors[1])
# the show_iamge command is in the fastai lib and we can display tensors as images with it
three_shape = three_tensors[1].shape
print(three_shape) # its a tensor of shape torch.Size([28, 28])

three_type = type(three_tensors) 
print(three_type) # its a list

# because three_type is a list, we can't do heavy mathematically heavy computations on it
# so what we can do is we can stack all the 3s on top of each others to form a tensor with x dimensions
