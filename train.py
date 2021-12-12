from fastai import *
from fastai.vision.all import *


path = untar_data(URLs.PETS)
fname = (path/"images").ls()[0]
pets1 = DataBlock(blocks = (ImageBlock, CategoryBlock),
                 get_items=get_image_files, 
                 splitter=RandomSplitter(seed=42),
                 get_y=using_attr(RegexLabeller(r'(.+)_\d+.jpg$'), 'name'),
                 item_tfms=Resize(460),
                 batch_tfms=aug_transforms(size=224, min_scale=0.75))
# pets1.summary(path/"images")
dls = pets1.dataloaders(path/"images")
learn = cnn_learner(dls, resnet34, metrics=error_rate)
learn.fine_tune(2)
# Path.BASE_PATH = path
# path = Path("/content/pets/oxfordpet")


# from pathlib import *
# from fastai.vision.all import *

# get_tar = '!wget https://s3.amazonaws.com/fast-ai-imageclas/oxford-iiit-pet.tgz && mkdir -p /content/pets'
# unzip_dataset = '!tar -xzf oxford-iiit-pet.tgz -C /content/pets'
# rm_dataset = '!rm oxford-iiit-pet.tgz'
# path = untar_data(URLs.MNIST_SAMPLE)
# Path.BASE_PATH = path
# print((path).ls())
