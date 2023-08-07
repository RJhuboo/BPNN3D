import torch
import os
import numpy as np
from torch.optim import Adam, SGD
from torch.nn import MSELoss
from sklearn.metrics import r2_score
import pickle
import matplotlib.pyplot as plt


def MSE(y_predicted,y):
    squared_error = (y_predicted - y) **2
    sum_squared_error = np.sum(squared_error)
    mse = sum_squared_error / y.size
    return mse

class Trainer():
    def __init__(self,opt,my_model,device,save_fold,scaler):
        self.scaler = scaler
        self.save_fold = save_fold
        self.device = device
        self.opt = opt
        self.model = my_model
        self.NB_LABEL = opt.NB_LABEL
        self.optimizer = Adam(self.model.parameters(), lr=self.opt.lr)
        self.criterion = MSELoss()
        
    def train(self, trainloader, epoch ,steps_per_epochs=20):
        self.model.train()
        print("starting training")
        print("----------------")
        train_loss = 0.0
        train_total = 0
        running_loss = 0.0
        mse_score = 0.0
        for i, data in enumerate(trainloader,0):
            inputs, labels = data['image'], data['label']
            
            # reshape
            inputs = inputs.reshape(inputs.size(0),1,inputs.size(2),inputs.size(2),inputs.size(2))
            labels = labels.reshape(labels.size(0),1)
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            # zero the parameter gradients
            self.optimizer.zero_grad()
            
            # forward backward and optimization
            outputs = self.model(inputs)
            loss = self.criterion(outputs,labels)
            loss.backward()
            self.optimizer.step()
            
            # statistics
            train_loss += loss.item()
            running_loss += loss.item()
            train_total += 1
            #outputs, labels = outputs.cpu().detach().numpy(), labels.cpu().detach().numpy()
            #labels, outputs = np.array(labels), np.array(outputs)
            #labels, outputs = labels.reshape(self.NB_LABEL,len(inputs)), outputs.reshape(self.NB_LABEL,len(inputs))
            if i % self.opt.batch_size == self.opt.batch_size-1:
                print('[%d %5d], loss: %.3f' %
                      (epoch + 1, i+1, running_loss/self.opt.batch_size))
                running_loss = 0.0
                
        # displaying results
        mse = train_loss / train_total
        print('Epoch [{}], Loss: {}'.format(epoch+1, train_loss/train_total), end='')
        print('Finished Training')
        
        # saving trained model
        check_name = "BPNN_checkpoint_" + str(epoch) + ".pth"
        torch.save(self.model.state_dict(),os.path.join(self.opt.checkpoint_path,check_name))
        return mse

    def test(self,testloader,epoch,writer):

        test_loss = 0
        test_total = 0
        mse_score = 0.0
        output = []
        label = []
        IDs = {}
        # Loading Checkpoint
        if self.opt.mode == "Test":
            check_name = "BPNN_checkpoint_" + str(epoch) + ".pth"
            self.model.load_state_dict(torch.load(os.path.join(self.opt.checkpoint_path,check_name)))
        
        self.model.eval()

        # Testing
        with torch.no_grad():
            for i, data in enumerate(testloader):
                inputs, labels, ID = data['image'],data['label'],data['ID']
                # reshape
                inputs, labels = inputs.to(self.device),labels.to(self.device)
                inputs = inputs.reshape(inputs.size(0),1,inputs.size(2),inputs.size(2),inputs.size(2))
                labels = labels.reshape(labels.size(0),1)
                # loss
                outputs = self.model(inputs)
                loss = self.criterion(outputs,labels)
                test_loss += loss.item()
                test_total += 1
                
                # statistics
                outputs,labels=outputs.reshape(1,self.opt.NB_LABEL), labels.reshape(1,self.opt.NB_LABEL)
                labels, outputs = labels.cpu().detach().numpy(), outputs.cpu().detach().numpy()
                #labels, outputs = np.array(labels), np.array(outputs)
                #labels, outputs = labels.reshape(self.NB_LABEL,1), outputs.reshape(self.NB_LABEL,1)
                #labels=labels.reshape(1,self.NB_LABEL)
                #outputs=outputs.reshape(1,self.NB_LABEL)

                #if self.opt.norm_method == "standardization" or self.opt.norm_method == "minmax":
                #    outputs = self.scaler.inverse_transform(outputs)
                #    labels = self.scaler.inverse_transform(labels)
                output.append(outputs)
                label.append(labels)
                IDs[i] = ID[0]
                label = np.array(label)
                output = np.array(output)
                size_label=len(label)
                output,label = output.reshape((size_label,1)), label.reshape((size_label,1))
                print(np.shape(label))
                for i in range(np.shape(label)[1]):
                    fig,ax = plt.subplots()
                    ax.scatter(label[:,i],output[:,i],label='slice')
                    ax.plot(label[:,i],label[:,i])
                    writer.add_figure(str(i),fig)
            name_out = "./result" + str(epoch) + ".pkl"
            mse = test_loss/test_total
            
            with open(os.path.join(self.save_fold,name_out),"wb") as f:
                pickle.dump({"output":output,"label":label,"ID":IDs},f)
            #with open(os.path.join(self.save_fold,name_lab),"wb") as f:
                #pickle.dump(label,f)
           
        print(' Test_loss: {}'.format(test_loss/test_total))
        return mse
    
    def inference(infloader,epoch):
       
        output = {}
        IDs = {}
        # Loading Checkpoint
        check_name = "BPNN_checkpoint_" + str(epoch) + ".pth"
        self.model.load_state_dict(torch.load(os.path.join(self.opt.checkpoint_path,check_name)))
        self.model.eval()

        # Testing
        with torch.no_grad():
            for i, data in enumerate(testloader):
                inputs, ID = data['image'],data['ID']
                # reshape
                inputs = inputs.reshape(1,1,self.size_image,self.size_image)
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                outputs = outputs.cpu().detach().numpy()
                outputs = np.array(outputs)
                outputs=outputs.reshape(1,self.NB_LABEL)
                
                if self.opt.norm_method == "standardization" or self.opt.norm_method == "minmax":
                    outputs = self.scaler.inverse_transform(outputs)
                print("output :",outputs)
                output[i] = outputs
                label[i] = labels
                IDs[i] = ID[0]
            name_out = "./result" + str(epoch) + ".pkl"
            mse = test_loss/test_total
            with open(os.path.join(self.save_fold,name_out),"wb") as f:
                pickle.dump({"output":output,"label":label,"ID":IDs},f)
            #with open(os.path.join(self.save_fold,name_lab),"wb") as f:
                #pickle.dump(label,f)
           
        print(' Test_loss: {}'.format(test_loss/test_total))
