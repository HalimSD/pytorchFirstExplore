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

# im3 is of type PIL png python imaging library
print(type(im3))

# to turn the image into numbers is to turn it into array
im_array = array(im3)[4:10, 4:10]
print(im_array)

# tensor is the pytorch version of a numpy array
im_tensor = tensor(im3)[4:10, 4:10]
print(im_tensor)

# loop over the lest of 7s and 3s, openning the image, trun it into a tensor and then return the list
seven_tensors = [tensor(Image.open(o)) for o in sevens]
three_tensors = [tensor(Image.open(o)) for o in threes]
print(len(three_tensors), len(seven_tensors))

# show_image(three_tensors[1])
# the show_iamge command is in the fastai lib and we can display tensors as images with it
three_shape = three_tensors[1].shape
print(three_shape)  # its a tensor of shape torch.Size([28, 28])

three_type = type(three_tensors)
print(three_type)  # its a list

# because three_type is a list, we can't do mathematically heavy computations on it
# so what we can do is we can stack all the 3s on top of each others to form a tensor with x dimensions

# to stack the tensors up we use
# we turn them into floting points values so we don't have ints rounding off
# we devide by 255 to have values between 0 and 1
stacked_threes = torch.stack(three_tensors).float()/255.0
# the answer would be: torch_size([6131,28,28]) which is a rank 3 tensor because it has 3 axies or dimension
stacked_sevens = torch.stack(seven_tensors).float()/255.0


# we can take the rank or the dimensions of a tensor by taking the lengh of its shape
len(stacked_threes.shape)

# to take the mean of the stacked three tensor it's important to specify of which dimension we wanna take the mean
mean3 = stacked_threes.mean(0)
# we had the mean of the first dimension containing the images
mean7 = stacked_sevens.mean(0)

# the mean absolute difference of L1 norm is taking all the negative numbers and turn them into positive |-7| = 7
# L2 norm is the RMSE root mean squared error which is the mean of the squared differences and the the square root

a_3 = stacked_threes[1]
# L1 norm the distance using absolute value
dist_3_abs = (a_3 - mean3).abs().mean()
dist_3_sqr = ((a_3 - mean3)**2).mean().sqrt()  # L2 norm RMSE

a_7 = stacked_sevens[1]
dist_7_abs = (a_7 - mean7).abs().mean()
dist_7_sqr = ((a_7 - mean7)**2).mean().sqrt()
# PyTorch already provides both of these as loss functions. You'll find these inside torch.nn.functional,
# which the PyTorch team recommends importing as F
F.l1_loss(a_3.float(), mean7), F.mse_loss(a_3, mean7).sqrt()


valid_3_tens = torch.stack([tensor(Image.open(o))
                            for o in (path/'valid'/'3').ls()])
valid_3_tens = valid_3_tens.float()/255
valid_7_tens = torch.stack([tensor(Image.open(o))
                            for o in (path/'valid'/'7').ls()])
valid_7_tens = valid_7_tens.float()/255


def mnist_distance(a,b): return (a-b).abs().mean((-1,-2))
mnist_distance(a_3, mean3)
# in this function we're taking the distance between a_3, mean3 to do that we need to 
# take the mean of the abs value of the last and second last dims of the tensors
# we do the mean on the last 2 dims because when we do it on the stacked tensor the shape will be torch.Size([1010, 28, 28])

# we can calculate the dist on the entire val set through broadcasting as the following
valid_3_dist = mnist_distance(valid_3_tens, mean3)
# valid_3_tens.shape = torch.Size([1010, 28, 28]) because it's been done for the whole val dataset
# mean3.shape = torch.Size([28, 28]) because it's a single img
# so broadcasting means that if the previous two shapes don't match, broadcasting acts as if there is a 1010 versions of [28,28]
# so it's gonna subtract [28,28] from every single one of valid_3_tens

# this function will check the given tensor if it's closer to the perfect tensor of the 3 or the 7
def is_3(x): return mnist_distance(x,mean3) < mnist_distance(x,mean7)
# so is_3(a_3) will return tensore(True) but is_3(a_3).float() will return a 1.

# using broadcasting we can call the previous function on the entire val set as following is_3(val_3_tens)
# so the accuracy of this 'model' on the 3s is 
accuracy_3s = is_3(valid_3_tens).float().mean()
accuracy_7s = is_3(valid_7_tens).float().mean()


# USING GRADIENT DESCENT
def f(x): return x**2
xt = tensor(3.).requires_grad_() # the _ at the end is an inplace operation which tells pytorch to keep track of it
yt = f(xt) # yt returns tensor(9., grad_fn=<PowBackward0>)
yt.backward() # backward()is the back propagation which means take the derivative
xt.grad # we find the gradient after we kept track of the derivative using the requires_grad_() 
# the derivative of x**2 is 2x wich means xt.grad will return tensor(6.). the derivative is the slope at some point
# the gradient is the slope of the functions. it tells us if we change the input, how will the output change


# An End-to-End SGD Example
time = torch.arange(0,20).float()
speed = torch.randn(20)*3 + 0.75*(time-9.5)**2 + 1
def f(t, params):
    a,b,c = params
    return a*(t**2) + (b*t) + c
def mse(preds, targets): return ((preds-targets)**2).mean().sqrt()

# Step 1: Initialize the parameters
params = torch.randn(3).requires_grad_() # 3 for a,b and c random values for the parameters, we track the gradience to adjust them later


# Step 2: Calculate the predictions
preds = f(time, params)

# Step 3: Calculate the loss
loss = mse(preds, speed)

# Step 4: Calculate the gradients
loss.backward()
params.grad
params.grad * 1e-5

# Step 5: Step the weights
lr = 1e-5
params.data -= lr * params.grad.data
params.grad = None
preds = f(time,params)
mse(preds, speed)

# Step 6: Repeat the process
def apply_step(params, prn=True):
    preds = f(time, params)
    loss = mse(preds, speed)
    loss.backward()
    params.data -= lr * params.grad.data
    params.grad = None
    if prn: print(loss.item())
    return preds

for i in range(10): apply_step(params)
params = orig_params.detach().requires_grad_()

# Step 7: stop

