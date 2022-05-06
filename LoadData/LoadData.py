from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import cv2
import glob
import re
import numpy as np
import random

train_data_path = '/mnt/c/PersonalScripts/CDS-5950-Capstone/Data/train' 
test_data_path = '/mnt/c/PersonalScripts/CDS-5950-Capstone/Data/test'

train_image_paths = [] #to store image paths in list
classes = [] #to store class values

for data_path in glob.glob(train_data_path + '/*'):
    classes.append(data_path.split('/')[-1]) 

    train_image_paths.append(glob.glob(data_path + '/*'))
    
train_image_paths = [path for sublist in train_image_paths for path in sublist]
train_image_paths = [re.sub(r"/mnt/c/PersonalScripts", "/content", x) for x in train_image_paths]

random.shuffle(train_image_paths)

print('train_image_path example: ', train_image_paths[0])
print('class example: ', classes[0])

test_image_paths = []
for data_path in glob.glob(test_data_path + '/*'):
    test_image_paths.append(glob.glob(data_path + '/*'))

test_image_paths = [path for sublist in test_image_paths for path in sublist]
test_image_paths = [re.sub(r"/mnt/c/PersonalScripts", "/content", x) for x in test_image_paths]

print("Train size: {}\nTest size:  {}".format(len(train_image_paths), len(test_image_paths)))

idx_to_class = {i:j for i, j in enumerate(classes)}
class_to_idx = {value:key for key,value in idx_to_class.items()}

train_transforms = A.Compose(
    [
        A.SmallestMaxSize(max_size=350),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05,
                           rotate_limit=360, p=0.5),
        A.RandomCrop(height=256, width=256),
        A.RGBShift(r_shift_limit=8, g_shift_limit=8,
                   b_shift_limit=8, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.MultiplicativeNoise(multiplier=[0.5, 2], per_channel=True, p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        A.HueSaturationValue(hue_shift_limit=0.2,
                             sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
        ToTensorV2(),
    ]
)

test_transforms = A.Compose(
    [
        A.SmallestMaxSize(max_size=350),
        A.CenterCrop(height=256, width=256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)

class Brewskis(Dataset):
  def __init__(self, image_paths, transform=False):
    self.image_paths = image_paths
    self.transform = transform
      
  def __len__(self):
    return len(self.image_paths)

  def __getitem__(self, idx):
    image_filepath = self.image_paths[idx]
    image = cv2.imread(image_filepath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    label = image_filepath.split('/')[-2]
    label = class_to_idx[label]
    if self.transform is not None:
      image = self.transform(image=image)["image"]
    
    return image, label

train_dataset = Brewskis(train_image_paths, transform=train_transforms)
test_dataset = Brewskis(test_image_paths, transform=test_transforms)

train_loader = DataLoader(
    train_dataset, batch_size=128, shuffle=True
)

test_loader = DataLoader(
    test_dataset, batch_size=128, shuffle=False
)

torch.save(train_loader, 'train_loader.pt')
torch.save(test_loader, 'test_loader.pt')