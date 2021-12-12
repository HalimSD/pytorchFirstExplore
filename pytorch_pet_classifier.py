import torch
import torchvision
import torch.utils.data as data
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transformers
import os
import re as regex
from skimage import io

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

dataset = ImagesLoader(root_dir=data_path, transform= transformers.ToTensor())
train_set, test_set = data.random_split(dataset, [6894,500])
train_loader = DataLoader(dataset=train_set, batch_size= batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size= batch_size, shuffle=True)
