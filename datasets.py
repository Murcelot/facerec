import os 
import pandas as pd 
import torch
from torchvision.io import read_image
from torch.utils.data import Dataset

class RealDataset(Dataset):
    def __init__(self, path_train, path_json, img_transform = None):
        data = pd.read_json(os.path.join(os.getcwd(), path_train, path_json), orient = 'index')
        self.real_data = data[data[0] == 0].index
        self.img_transform = img_transform
        self.path_train = path_train

    def __len__(self):
        return len(self.real_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        path_image = self.real_data[idx]
        image = read_image(os.path.join(os.getcwd(), self.path_train, 'images', path_image)) / 255
        label = int(path_image[:6])

        if self.img_transform:
            image = self.img_transform(image)

        return image, label

class WholeDataset(Dataset):
    def __init__(self, path_train, path_json, img_transform = None):
        self.data = pd.read_json(os.path.join(os.getcwd(), path_train, path_json), orient = 'index')
        self.img_transform = img_transform
        self.path_train = path_train

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        path_image = self.data.iloc[idx].name
        fake = self.data.iloc[idx][0]
        
        image = read_image(os.path.join(os.getcwd(), self.path_train, 'images', path_image)) / 255
        label = int(path_image[:6]) + 10000 * int(fake)

        if self.img_transform:
            image = self.img_transform(image)

        return image, label