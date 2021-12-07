from fastai.vision.all import *
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
