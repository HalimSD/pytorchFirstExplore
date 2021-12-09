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

# after preparing the training and validation dataset we need to initialize the parameters
# we begin from a random generated parameters, std to give a variance of 1, 
# requires_grad_() to change the tensor such that it requires gradient
def init_params(size, std=1.0): return (torch.randn(size)*std).requires_grad_()

# weights are 28*28 which are the pixels in the image and we need a weight for each pixel, 1 for the columns
weights = init_params((28*28,1))

# the weights and pixels aren't enough because the weights are 0 when the pixels are 0
# so from the formula y = ax +b we need the b which is the bias, which is initialized as random too
bias = init_params(1) 

# y = wx + b , the wights are the w, the bias is the b, and the weights and bias together are the parameters
# the parameters are the things that have gradient and will change and update

# to calculate the predictions of the first image:
(train_x[0]*weights.T).sum() + bias
# we want to dat for each image, we could do it using a for loop but it will be slow and it won't run of the gpu
# so matrix multiplication gives us an optimized way to do the previous linear function for as many rows and columns as we want 
def linear1(xb): return xb@weights + bias
preds = linear1(train_x)

# to check the accuracy we check if the output is greater than 0 then it represents a 3 or a 7 the turn it into float to have 1 for true
corrects = (preds>0.0).float() == train_y

# calculating the derivative on the accuracy, to do that we change the parameter 'weight' by a little 
weights[0].data *= 1.0001
# to calculate how the accuracy changes based on the change in that weight, that would be the gradient of the accuracy with respect to that parameter 
preds = linear1(train_x)
((preds>0.0).float() == train_y).float().mean().item()

# We have a problem here because corrects == preds so they cancels each others out
# that will lead to a gradient of 0, which lead to the step being 0, which mean the predictions won't change
# with a gradient of 0 we can't take a step and we can't make better predictions

# a better loss function that doesn't have the 0 gradient all over the place from: preds>0.0 in the previous preds func
def mnist_loss(predictions, targets):
    return torch.where(targets==1, 1-predictions, predictions).mean()

# the mnist_loss func will only work if the predictions are between 0 and 1, otherwise the 1-predictions won't really work
def sigmoid(x): return 1/(1+torch.exp(-x))

# now we can update the loss func to implement the sigmoid func as following
def mnist_loss(predictions, targets):
    predictions = predictions.sigmoid()
    return torch.where(targets==1, 1-predictions, predictions).mean()


# Putting it all together for SGD:
weights = init_params((28*28,1))
bias = init_params(1)
dl = DataLoader(dset, batch_size=256)
valid_dl = DataLoader(valid_dset, batch_size=256)
batch = train_x[:4]
preds = linear1(batch)
loss = mnist_loss(preds, train_y[:4])
loss.backward()

def calc_grad(xb, yb, model):
    preds = model(xb)
    loss = mnist_loss(preds, yb)
    loss.backward()

calc_grad(batch, train_y[:4], linear1)
weights.grad.zero_()
bias.grad.zero_()

def train_epoch(model, lr, params):
    for xb,yb in dl:
        calc_grad(xb, yb, model)
        for p in params:
            p.data -= p.grad*lr
            p.grad.zero_()

# xb are the predictions and yb are the targets
def batch_accuracy(xb, yb):
    preds = xb.sigmoid()
    # we used to compare the predictions to whether they are greater or less than 0, 
    # but now comparing to 0.5 because of the sigmoid
    correct = (preds>0.5) == yb
    return correct.float().mean()

def validate_epoch(model):
    accs = [batch_accuracy(model(xb), yb) for xb,yb in valid_dl]
    # the output of accs is a list, we convert it to a tensor using stack and round it for 4 decimals just for display
    return round(torch.stack(accs).mean().item(), 4)

# training for 1 epoch:
lr = 1.
params = weights,bias
train_epoch(linear1, lr, params)
validate_epoch(linear1)

# in reiality we don't have to use our own built linear1 function
# pytorch implement it for us and initializes the weights and biases to 1
# the following line of code will a linear model with:
#  a matrix of size 28*28 , 1
# a bias of size 1
# set requires_grad_ = true
linear_model = nn.Linear(28*28,1)






