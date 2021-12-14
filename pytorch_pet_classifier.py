import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.utils.data as data
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transformers
import os
import re as regex
from skimage import io, transform

data_path = './images'
batch_size = 32

class ImagesLoader(DataLoader):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.files_list = os.listdir(self.root_dir)
    
    def __len__(self):
        return len(self.files_list)

    def __getitem__(self, index):
        image_full_name = self.files_list[index]
        iamge_path = os.path.join(self.root_dir, image_full_name)
        image_name = regex.findall(r'(.+)_\d+.jpg$', image_full_name)

        image = io.imread(iamge_path)
        # io.imshow(image)
        
        if self.transform:
            image = self.transform(image)
        
        return (image, image_name)

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=2)

        self.fc1 = nn.Linear(in_features=12 * 4 * 4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.fc3 = nn.Linear(in_features=60, out_features=37)

    def forward(self, tensor):
        tensor = F.relu(self.conv1(tensor))
        tensor = F.max_pool2d(tensor, kernel_size=2)

        tensor = F.relu(self.conv2(tensor))
        tensor = F.max_pool2d(tensor, kernel_size=2)

        # Before we pass our input to the first hidden linear layer, we must reshape() or flatten our tensor. 
        # This will be the case any time we are passing output from a convolutional layer as input to a linear layer.
        tensor = tensor.reshap(-1, 12 * 4 * 4)
        tensor = F.relu(self.fc1(tensor))
        tensor = F.relu(self.fc2(tensor))
        tensor = self.out(tensor)
        tensor = F.softmax(tensor)

        return tensor

def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()


dataset = ImagesLoader(root_dir=data_path, transform= transformers.Compose([
                                                        transformers.ToPILImage(), 
                                                        transformers.Resize(240 * 240), 
                                                        transformers.ToTensor()]))
train_set, test_set = data.random_split(dataset, [6894,500])
train_loader = DataLoader(dataset=train_set, batch_size= batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size= batch_size, shuffle=True)

network = Network()
optimizer = optim.Adam(network.parameters(), lr=0.01)

for epoch in range(10):

    total_loss = 0
    total_correct = 0

    for batch in train_loader: # Get Batch
        print(">")
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