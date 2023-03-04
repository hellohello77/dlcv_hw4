import torch
from PIL import Image
import os
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn as nn
import torchvision.transforms as T
import csv
import argparse
# from byol_pytorch import BYOL

parser = argparse.ArgumentParser()
parser.add_argument("img_csv")
parser.add_argument("img_folder")
parser.add_argument("out_csv")
args = parser.parse_args()

if torch.cuda.is_available():
      device = torch.device('cuda')
else:
      device = torch.device('cpu')

"""# Dataset

## Dataset
"""

all_labels = ['Candles', 'Bed', 'Speaker', 'Notebook', 'Scissors', 'Fork', 'Mug', 'Marker', 'Hammer', 'Refrigerator', 'Flowers', 'TV', 'Exit_Sign', 'Telephone', 'Shelf', 'Backpack', 'Calendar', 'Webcam', 'Sink', 'Postit_Notes', 'Pencil', 'Printer', 'Ruler', 'Folder', 'Curtains', 'Knives', 'Soda', 'Bike', 'File_Cabinet', 'Pen', 'Lamp_Shade', 'Push_Pin', 'Monitor', 'Paper_Clip', 'Radio', 'Trash_Can', 'Flipflops', 'Table', 'Couch', 'Clipboards', 'Drill', 'Toys', 'Oven', 'Eraser', 'Pan', 'ToothBrush', 'Helmet', 'Fan', 'Alarm_Clock', 'Sneakers', 'Mop', 'Batteries', 'Chair', 'Spoon', 'Desk_Lamp', 'Computer', 'Keyboard', 'Screwdriver', 'Mouse', 'Bottle', 'Glasses', 'Kettle', 'Calculator', 'Bucket', 'Laptop']
all_labels.sort()

class hw4_2_dataset:
    def __init__(self, filepath, transform, file):
        self.transform = transform
        self.filepath = filepath
        self.file_list = []
        with open(file, newline='') as csvfile:
            reader = csv.reader(csvfile)
            data_list = list(reader)
            _ = data_list.pop(0)
            self.file_list = [i[1] for i in data_list]
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.filepath, self.file_list[idx])
        img = Image.open(img_path)
        transformed_img = self.transform(img)
        img.close()
        return transformed_img, self.file_list[idx]

"""## Data Loader"""

import math
img_transform = T.Compose([
    T.Resize(128),
    T.CenterCrop(128),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
               std=[0.229, 0.224, 0.225])
])
val_dataset = hw4_2_dataset(args.img_folder, img_transform, args.img_csv)
BATCH_SIZE = 64
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

"""# Model"""
class pretrained_resnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.pretrained = models.resnet50(pretrained=False)
        # self.pretrained.load_state_dict(torch.load('./hw4_2_C_29.ckpt', map_location=device))
        self.classifier = nn.Linear(2048, 65)
        self.handler = self.pretrained.avgpool.register_forward_hook(self.features)
        
    def features(self, module, fin, fout):
        self.fs = torch.flatten(fout, 1)
    
    def forward(self, x):
        _ = self.pretrained(x)
        x = self.fs
        x = self.classifier(x)
        return x

torch.cuda.empty_cache()

model = pretrained_resnet()
model.load_state_dict(torch.load('./hw4_2_C_30.ckpt', map_location=device))
model = model.to(device)
"""# Training

## Train
"""

            
            
model.eval()
with open(args.out_csv, "a") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['id', 'filename', 'label'])
    with torch.no_grad():
        id = 0
        for idx, (imgs, filename) in enumerate(val_dataloader):
            imgs = imgs.to(device)
            outputs = model(imgs)
            
            preds = torch.max(outputs, dim=1)[1]
            for k in range(len(imgs)):
                writer.writerow([id, filename[k], all_labels[preds[k].cpu().item()]])
                id += 1