import torch
from torchvision.utils import make_grid
import torch.nn as nn
import torch.nn.functional as F

 ## Neural Network for regression ##
class NeuralNet(nn.Module):
    def __init__(self,n1,n2,n3,out_channels):
        super().__init__()
        self.fc1 = nn.Linear(8*8*8*8,n1)
        self.fc2 = nn.Linear(n1,n2)
        self.fc3 = nn.Linear(n2,n3)
        self.fc4 = nn.Linear(n3,out_channels)
    def forward(self,x):
        x = torch.flatten(x,1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

## 3 CNN model ##
class ConvNet(nn.Module):
    def __init__(self,features,out_channels,n1=240,n2=120,n3=60,k1=(3,3,3),k2=(3,3,3),k3=(3,3,3)):
        super(ConvNet,self).__init__()
        # initialize CNN layers 
        self.conv1 = nn.Conv3d(1,features,kernel_size = k1,stride = (1,1,1), padding = 1)
        self.conv2 = nn.Conv3d(features,features*2, kernel_size = k2, stride = (1,1,1), padding = 1)
        self.conv3 = nn.Conv3d(features*2,8, kernel_size = k3, stride = (1,1,1), padding = 1)
        self.pool = nn.MaxPool3d((2,2,2))        
        self.neural = NeuralNet(n1,n2,n3,out_channels)
        # dropout
        # self.dropout = nn.Dropout(0.25)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.neural(x)
        #x = torch.flatten(x,1)
        return x 
       
