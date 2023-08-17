import torch
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from skimage import io,transform
from torchvision import transforms, utils
import argparse
from sklearn import preprocessing
import h5py

def normalization(csv_file,mode,indices):
    Data = pd.read_csv(csv_file)
    print(len(Data))
    if mode == "standardization":
        scaler = preprocessing.StandardScaler()
    elif mode == "minmax":
        scaler = preprocessing.MinMaxScaler()
    scaler.fit(Data.iloc[indices,1:])
    return scaler

class Datasets(Dataset):
    def __init__(self,opt, csv_file, image_dir, scaler, transform=None):
        self.image_dir = image_dir
        self.labels = pd.read_csv(csv_file)
        self.transform = transform
        self.opt = opt
        self.scaler = scaler
        
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        #img_name = os.path.join(self.image_dir, str(self.labels.iloc[idx,0]))
        with h5py.File(self.image_dir,'r') as file_h5:
            im = file_h5[str(idx)].astype(np.float32)
            print(np.unique(im))
            lab = self.scaler.transform(self.labels.iloc[:,1:])
            lab = pd.DataFrame(lab)
            lab.insert(0,"File name", self.labels.iloc[:,0], True)
            lab.columns = self.labels.columns
            labels = lab.iloc[idx,-2] # Takes all corresponding labels
            labels = np.array([labels]) 
            labels = labels.astype('float32')
            #image = tio.Subject(ct=tio.ScalarImage(img_name+".nii.gz")) # Loading Image
            #if self.transform:
            #    image = self.transform(image)
            #im = image['ct'][tio.DATA]
            #im = im.type(torch.FloatTensor)
            #print(im.type())
            return {"image":im, "label":labels, "ID":lab.iloc[idx,0]}

class Test_Datasets(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image_name = os.listdir(self.image_dir)
        img_name = os.path.join(self.image_dir,image_name[idx])
        image = io.imread(img_name) # Loading Image
        image = image / 255.0 # Normalizing [0;1]
        image = image.astype('float32') # Converting images to float32
        #sample = {'image': image}
        #if self.transform:
        #    sample = self.transform(sample)
        return image
