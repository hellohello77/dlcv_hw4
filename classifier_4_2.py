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
import csv

path_to_datafile = './hw4_data/office'

if torch.cuda.is_available():
      device = torch.device('cuda:3')
else:
      device = torch.device('cpu')

"""# Dataset

## Dataset
"""
all_labels = ['Alarm_Clock', 'Backpack', 'Batteries', 'Bed', 'Bike', 'Bottle', 'Bucket', 'Calculator', 'Calendar', 'Candles', 'Chair', 'Clipboards', 'Computer', 'Couch', 'Curtains', 'Desk_Lamp', 'Drill', 'Eraser', 'Exit_Sign', 'Fan', 'File_Cabinet', 'Flipflops', 'Flowers', 'Folder', 'Fork', 'Glasses', 'Hammer', 'Helmet', 'Kettle', 'Keyboard', 'Knives', 'Lamp_Shade', 'Laptop', 'Marker', 'Monitor', 'Mop', 'Mouse', 'Mug', 'Notebook', 'Oven', 'Pan', 'Paper_Clip', 'Pen', 'Pencil', 'Postit_Notes', 'Printer', 'Push_Pin', 'Radio', 'Refrigerator', 'Ruler', 'Scissors', 'Screwdriver', 'Shelf', 'Sink', 'Sneakers', 'Soda', 'Speaker', 'Spoon', 'TV', 'Table', 'Telephone', 'ToothBrush', 'Toys', 'Trash_Can', 'Webcam']

class hw4_2_dataset:
    def __init__(self, filepath, transform, file):
        self.datatype = os.path.splitext(os.path.basename(file))[0]
        self.transform = transform
        self.filepath = filepath
        self.file_list = []
        self.label_list = []
        with open(os.path.join(filepath, file), newline='') as csvfile:
            reader = csv.reader(csvfile)
            data_list = list(reader)
            _ = data_list.pop(0)
            for i in data_list:
                self.file_list.append(i[1])
                self.label_list.append(all_labels.index(i[2]))
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.filepath, self.datatype, self.file_list[idx])
        img = Image.open(img_path)
        transformed_img = self.transform(img)
        img.close()
        return transformed_img, self.label_list[idx]

"""## Data Loader"""

import math
img_transform = T.Compose([
    T.Resize(128),
    T.CenterCrop(128),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
               std=[0.229, 0.224, 0.225])
])
train_dataset = hw4_2_dataset(path_to_datafile, img_transform, 'train.csv')
val_dataset = hw4_2_dataset(path_to_datafile, img_transform, 'val.csv')
BATCH_SIZE = 64
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

"""# Model"""
class pretrained_resnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.pretrained = models.resnet50(pretrained=False)
        self.pretrained.load_state_dict(torch.load('./hw4_data/pretrain_model_SL.pt', map_location=device))
        # self.pretrained.load_state_dict(torch.load('./hw4_2_99.ckpt', map_location=device))
        self.classifier = nn.Linear(2048, 65)
        self.handler = self.pretrained.avgpool.register_forward_hook(self.features)
        
    def features(self, module, fin, fout):
        self.fs = torch.flatten(fout, 1)
    
    def forward(self, x):
        _ = self.pretrained(x)
        x = self.fs.detach()
        # x = self.fs
        x = self.classifier(x)
        return x

torch.cuda.empty_cache()

model = pretrained_resnet()
# model.load_state_dict(torch.load('./hw4_2_C_15.ckpt', map_location=device))
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
criterion = nn.CrossEntropyLoss()
"""# Training

## Train
"""

n_epoch = 30
current_best = 0
for epoch in range(n_epoch):
    running_loss = 0
    running_corrects = 0
    model.train()
    for imgs, labels in tqdm(train_dataloader):
        imgs = imgs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        preds = torch.max(outputs, dim=1)[1]
        corrects = torch.sum(preds == labels).cpu().item()
        running_corrects += corrects
        
        running_loss+=loss.cpu().item()
      
    print('Epoch ', epoch+16, " D train loss: ", running_loss/len(train_dataloader))
    print('Epoch ', epoch+16, " D train acc: ", running_corrects/len(train_dataset))
    
    model.eval()
    with torch.no_grad():
        val_corrects = 0
        for idx, (imgs, labels) in enumerate(val_dataloader):
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = model(imgs)
            
            preds = torch.max(outputs, dim=1)[1]
            corrects = torch.sum(preds == labels).cpu().item()
            val_corrects += corrects
        acc = val_corrects/len(val_dataset)
        print('Epoch ', epoch+16, " D val acc: ", acc)
    if acc>current_best:
        torch.save(model.state_dict(), f'./hw4_2_D_{epoch+16}.ckpt')
        current_best = acc