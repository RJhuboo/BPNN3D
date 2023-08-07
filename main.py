import torch
import os
import numpy as np
from matplotlib import pyplot as plt
import argparse
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader
import pickle
from sklearn.model_selection import train_test_split
import Model
from trainer import Trainer
import dataloader
from torch.utils.tensorboard import SummaryWriter

# GPU or CPU
if torch.cuda.is_available():  
  device = "cuda:0"
  print("running on gpu")
else:  
  device = "cpu"
  print("running on cpu")
  
''' Options '''

parser = argparse.ArgumentParser()
parser.add_argument("--label_dir", default = "/gpfsstore/rech/tvs/uki75tv/3D_label.csv", help = "path to label csv file")
parser.add_argument("--image_dir", default = "/gpfsstore/rech/tvs/uki75tv/output.h5",help = "path to image directory")
parser.add_argument("--train_cross", default = "./cross_output.pkl", help = "filename of the output of the cross validation")
parser.add_argument("--batch_size", type=int, default = 6, help = "number of batch")
parser.add_argument("--model", default = "MultiNet", help="Choose model : Unet or ConvNet") 
parser.add_argument("--nof", type=int, default = 10, help = "number of filter")
parser.add_argument("--lr", type=float, default = 0.001, help = "learning rate")
parser.add_argument("--nb_epochs", type=int, default = 100, help = "number of epochs")
parser.add_argument("--checkpoint_path", default = "./", help = "path to save or load checkpoint")
parser.add_argument("--mode", default = "Train", help = "Mode used : Train, Using or Test")
parser.add_argument("--k_fold", type=int, default = 1, help = "Number of splitting for k cross-validation")
parser.add_argument("--n1", type=int, default = 240, help = "number of neurons in the first layer of the neural network")
parser.add_argument("--n2", type=int, default = 120, help = "number of neurons in the second layer of the neural network")
parser.add_argument("--n3", type=int, default = 60, help = "number of neurons in the third layer of the neural network")
parser.add_argument("--nb_workers", type=int, default = 0, help ="number of workers for datasets")
parser.add_argument("--norm_method", type=str, default = "standardization", help = "choose how to normalize bio parameters")
parser.add_argument("--NB_LABEL", type=int, default = 1, help = "specify the number of labels")
opt = parser.parse_args()
NB_DATA = 6872
PERCENTAGE_TEST = 20
SIZE_IMAGE = 256
NB_LABEL = opt.NB_LABEL
'''functions'''

## RESET WEIGHT FOR CROSS VALIDATION

def reset_weights(m):
  '''
    Try resetting model weights to avoid
    weight leakage.
  '''
  for layer in m.children():
    if hasattr(layer, 'reset_parameters'):
        print(f'Reset trainable parameters of layer = {layer}')
        layer.reset_parameters()

## FOR TRAINING

def train():
    # Create the folder where to save results and checkpoints
    i=0
    while True:
        i += 1
        if os.path.isdir("./result/train"+str(i)) == False:
            save_folder = "./result/train"+str(i)
            os.mkdir(save_folder)
            break
    score_mse_t = []
    score_mse_v = []
    # defining data
    # transforms_dict = {
    #     tio.RandomAffine(scales = 0,
    #     degrees=(45,10,10),
    #     translation=(0,500,500)),
    #     }  
    # datasets_2 = dataloader.Datasets(csv_file = opt.label_dir, image_dir = opt.image_dir, opt=opt, indices = range(NB_DATA), transform = tio.Compose(transforms_dict))
    # transforms_dict = {
    #     tio.RandomAffine(scales = 0,
    #     degrees=(90,0,0),
    #     translation=(0,500,500)),
    #     }  
    #datasets_3 = dataloader.Datasets(csv_file = opt.label_dir, image_dir = opt.image_dir, opt=opt, indices = range(NB_DATA), transform = tio.Compose(transforms_dict))
    #datasets = torch.utils.data.ConcatDataset([datasets_1, datasets_2, datasets_3])
    split = train_test_split(range(NB_DATA),test_size = 0.2,random_state=42)
    if opt.norm_method == "standardization" or opt.norm_method == "minmax":
        scaler = dataloader.normalization(opt.label_dir,opt.norm_method,split[0])
    else:
        scaler = None
    datasets = dataloader.Datasets(csv_file = opt.label_dir, image_dir = opt.image_dir, opt=opt, scaler=scaler) # Create dataset
    print("start training")
    trainloader = DataLoader(datasets, batch_size = opt.batch_size, sampler = split[0], num_workers = opt.nb_workers )
    testloader =DataLoader(datasets, batch_size = 1, sampler = split[1], num_workers = opt.nb_workers )

 
    # defining the model
    model = Model.ConvNet(features =opt.nof,out_channels=NB_LABEL,n1=opt.n1,n2=opt.n2,n3=opt.n3,k1 = 3,k2 = 3,k3= 3).to(device)
    torch.manual_seed(5)

    writer = SummaryWriter(log_dir='runs/'+'training')
    # Start training
    t = Trainer(opt,model,device,save_folder,scaler)
    for epoch in range(opt.nb_epochs):
        mse_train = t.train(trainloader,epoch)
        mse_test = t.test(testloader,epoch,writer)
        score_mse_t.append(mse_train)
        score_mse_v.append(mse_test)
        writer.add_scalars('Loss',{'train':mse_train, 'test':mse_test},epoch)
    resultat = {"mse_train":score_mse_t, "mse_test":score_mse_v}
    with open(os.path.join(save_folder,opt.train_cross),'wb') as f:
        pickle.dump(resultat, f)
    with open(os.path.join(save_folder,"history.txt"),'wb') as g:
        history = "nof: " + str(opt.nof) + " model:" +str(opt.model) + " lr:" + str(opt.lr) + " neurons: " + str(opt.n1) + " " + str(opt.n2) + " " + str(opt.n3) + " kernel:" + str(3) + " norm data: " + str(opt.norm_method)
        pickle.dump(history,g)
      

''' main '''
if opt.mode == "Train":
    train()
else :
    i=0
    while True:
        i += 1
        if os.path.isdir("./result/train"+str(i)) == False:
            save_folder = "./result/train"+str(i)
            os.mkdir(save_folder)
            break
            
    model = Model.ConvNet(features =opt.nof,out_channels=NB_LABEL,n1=opt.n1,n2=opt.n2,n3=opt.n3,k1 = 3,k2 = 3,k3= 3).to(device)
    index = range(NB_DATA)
    split = train_test_split(index,test_size = 0.2,random_state=1)
    datasets = dataloader.Datasets(csv_file = opt.label_dir, image_dir = opt.image_dir, opt=opt, indices = split[0]) # Create dataset
    testloader = DataLoader(datasets, batch_size = 1, sampler = split[1], num_workers = opt.nb_workers )
    if opt.norm_method == "standardization" or opt.norm_method == "minmax":
        scaler = dataloader.normalization(opt.label_dir,opt.norm_method,split[0])
    else:
        scaler = None
    t = Trainer(opt,model,device,save_fold,scaler)
    t.test(testloader,75)
    

  
  
