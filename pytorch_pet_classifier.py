import torch as torch 
import torch.nn as nn
import torch.nn.functional as F
from glob import glob
from PIL import Image
from torchvision import transforms
import torch.utils.data as data
from torch.utils.data import DataLoader, SubsetRandomSampler
import re as regex
import torch.optim as optim


data_path = './images'
batch_size = 32
filenames = glob('./images/*.jpg')

def load_image(filename) :
    img = Image.open(filename)
    img = img.convert('RGB')
    return img

# Data preprocessing
classes = set()
data = []
labels = []

# Load the images and get the classnames from the image path
for image in filenames:
    class_name = regex.findall(r'(.+)_\d+.jpg$', image)[0]
    classes.add(class_name)
    img = load_image(image)

    data.append(img)
    labels.append(class_name)

# convert classnames to indices
class2idx = {cl: idx for idx, cl in enumerate(classes)}        
labels = torch.Tensor(list(map(lambda x: class2idx[x], labels))).long()

data = list(zip(data, labels))

class ImagesLoader(DataLoader):
    def __init__(self, data, transform=None):
        self.transform = transform
        self.data = data
        self.len = len(data)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        img, label = self.data[index]
        if self.transform:
            img = self.transform(img)
        
        return (img, label)

# Since the data is not split into train and validation datasets we have to 
# make sure that when splitting between train and val that all classes are represented in both
class Databasket():
    "Helper class to ensure equal distribution of classes in both train and validation datasets"
    
    def __init__(self, data, num_cl, val_split=0.2, train_transforms=None, val_transforms=None):
        class_values = [[] for x in range(num_cl)]
        
        # create arrays for each class type
        for d in data:
            class_values[d[1].item()].append(d)
            
        self.train_data = []
        self.val_data = []
        
        # put (1-val_split) of the images of each class into the train dataset
        # and val_split of the images into the validation dataset
        for class_dp in class_values:
            split_idx = int(len(class_dp)*(1-val_split))
            self.train_data += class_dp[:split_idx]
            self.val_data += class_dp[split_idx:]
            
        self.train_ds = ImagesLoader(self.train_data, transform=train_transforms)
        self.val_ds = ImagesLoader(self.val_data, transform=val_transforms)

from sklearn.model_selection import train_test_split

# Apply transformations to the train dataset
train_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

])

# apply the same transformations to the validation set, with the exception of the
# randomized transformation. We want the validation set to be consistent
val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

])

databasket = Databasket(data, len(classes), val_split=0.2, train_transforms=train_transforms, val_transforms=val_transforms)

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=2)

        self.fc1 = nn.Linear(in_features=12 * 55 * 55, out_features=110)
        self.fc2 = nn.Linear(in_features=110, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=37)

    def forward(self, t):
                # : torch.Tensor) -> torch.Tensor:
        t = t
        t = F.relu(self.conv1(t))
        t = F.max_pool2d(t, kernel_size=2)

        t = F.relu(self.conv2(t))
        t = F.max_pool2d(t, kernel_size=2)
        # Before we pass our input to the first hidden linear layer, we must reshape() or flatten our tensor. 
        # This will be the case any time we are passing output from a convolutional layer as input to a linear layer.
        t = t.reshape(-1, 12 * 55 *55)
        t = F.relu(self.fc1(t))
        t = F.relu(self.fc2(t))
        t = self.out(t)
        t = F.softmax(t)

        return t

def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()

train_indices = list(range(len(databasket.train_ds)))
test_indices = list(range(len(databasket.val_ds)))

train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(test_indices)

# Basic dataloader to retrieve mini-batches from the datasets
train_loader = DataLoader(databasket.train_ds, batch_size=64, sampler=train_sampler, shuffle=False, num_workers=0)
test_loader = DataLoader(databasket.val_ds, batch_size=64, sampler=test_sampler, shuffle=False, num_workers=0)

network = Network()
optimizer = optim.Adam(network.parameters(), lr=0.01)

for epoch in range(10):

    total_loss = 0
    total_correct = 0
    optimizer.zero_grad()

    for batch in train_loader: # Get Batch
        images, labels = batch 
        preds = network(images) # Pass Batch
        loss = F.cross_entropy(preds, labels) # Calculate Loss

        optimizer.zero_grad()
        loss.backward() # Calculate Gradients
        optimizer.step() # Update Weights

        total_loss += loss.item()
        total_correct += get_num_correct(preds, labels)
    print(
        "epoch:", epoch, 
        "total_correct:", total_correct, 
        "loss:", total_loss
    )