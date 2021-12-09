from fastai.vision.all import *


path = untar_data(URLs.PETS)
Path.BASE_PATH = path

# the ls function returns L instead of list, it shows the numbers of the items and display only the first 10
(path/"images").ls()

# examin the filenames
fname = (path/"images").ls()[0]

# using regular expressions module re from python
# r to treat all the \ as normal instead of as special by python
# all the parts of a regular expression that have a () around them, . for any letter + for 1 or more times
# followed by _ , followed by a digit one or more times
re.findall(r'(.+)_\d+.jpg$', fname.name)

pets = DataBlock(blocks = (ImageBlock, CategoryBlock),
                 get_items=get_image_files, 
                 splitter=RandomSplitter(seed=42),
                 get_y=using_attr(RegexLabeller(r'(.+)_\d+.jpg$'), 'name'),
                 item_tfms=Resize(460),
                 # this step can happen on the gpu because rotating is slow while the previous can on the cpu
                 # this stem don't change the actual pixels but it does the change on the coordinate values in a non lossy way
                 batch_tfms=aug_transforms(size=224, min_scale=0.75))
dls = pets.dataloaders(path/"images")

# displays all the transformations happening
pets.summary(path/"images")

# training the model 
learn = cnn_learner(dls, resnet34, metrics=error_rate)
learn.fine_tune(2)


acts = torch.randn((6,2))*2
acts.sigmoid()
(acts[:,0]-acts[:,1]).sigmoid()
sm_acts = torch.softmax(acts, dim=1)
targ = tensor([0,1,0,1,1,0])

F.nll_loss(sm_acts,targ, reduce='none')
# F.nll_loss is the same as the: 
#def mnist_loss(predictions, targets):
    #predictions = predictions.sigmoid()
    #return torch.where(targets==1, 1-predictions, predictions).mean()
# but instead of passing the target 1 we pass a tensor containing a list of the indexes mapped to the target predictions
# targs = tensor([0,1,0,1,1,0])
# idx = range(6)
# sm_acts[idx, targs] where each index is maooed to the corresponding target
# so the F.nll_loss = -sm_acts[idx,targs]
# nll stands for: negative log likelihood

# the log function is to turn the loss from between 0 and 1 to negative infinity and positive infinity
# because the previous loss function won't differentiate between 0.99 and 0.999 while the second is 10 times better
# y = b**a so the log tells us b to the power of what equals y and log(a*b) = log(a) + log(b)
# so the negative log likelihood is the mean of the negative or positive log of the probabilities
# so if we take the softmax and then the log likelihoodloss nll_loss the combination is called crossentropyloss
loss_func = nn.CrossEntropyLoss()
loss_func(acts,targ)
# the previous is exactly the same as:
F.cross_entropy(acts,targ) # but people tend to use the first one more