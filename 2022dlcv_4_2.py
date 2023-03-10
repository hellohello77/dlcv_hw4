# -*- coding: utf-8 -*-
"""2022dlcv_4_2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1YDQ-Pe-aTvQgH3pf-otXbXNJf66Jqtyh

"""## Packages"""

# Commented out IPython magic to ensure Python compatibility.
import torch
import PIL
from PIL import Image
import os
from torch.utils.data import DataLoader
import torchvision
from torchvision import models
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple
from tqdm import tqdm
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.utils import save_image, make_grid
from byol_pytorch import BYOL

path_to_datafile = './hw4_data/mini/train'

if torch.cuda.is_available():
      device = torch.device('cuda')
else:
      device = torch.device('cpu')

"""# Dataset

## Dataset
"""

class hw4_2_dataset:
    def __init__(self, filepath, transform):
        self.transform = transform
        self.filepath = filepath
        self.file_list = [file for file in os.listdir(filepath)]
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.filepath, self.file_list[idx])
        img = Image.open(img_path)
        transformed_img = self.transform(img)
        img.close()
        return transformed_img

"""## Data Loader"""

import math
img_transform = T.Compose([
    T.Resize(128),
    T.CenterCrop(128),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
               std=[0.229, 0.224, 0.225])
])
dataset = hw4_2_dataset(path_to_datafile, img_transform)
BATCH_SIZE = 64
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

"""# Model"""

torch.cuda.empty_cache()
resnet = models.resnet50(pretrained=False)
resnet = resnet.to(device)

learner = BYOL(
    resnet,
    image_size = 128,
    hidden_layer = 'avgpool',
    # use_momentum = False       # turn off momentum in the target encoder
)

optimizer = torch.optim.Adam(learner.parameters(), lr=3e-4)

"""# Training

## Train
"""

n_epoch = 100
for epoch in range(n_epoch):
    running_loss = 0
    for imgs in tqdm(dataloader):
        imgs = imgs.to(device)
        loss = learner(imgs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        learner.update_moving_average()
        running_loss+=loss.cpu().item()
    print(epoch, " ", running_loss/len(dataloader))
    if not (epoch+1)%20 :
        torch.save(resnet.state_dict(), f'./hw4_2_{epoch+1}.ckpt')