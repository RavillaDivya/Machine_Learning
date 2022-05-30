from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from PIL import Image
import PIL
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix

classes = np.array(['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck'])
l_num = {'0':0, '1':1, '2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9}

#The model class used for main experiment

class LeNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 6, kernel_size = 5, stride = 1)
        self.pool = nn.MaxPool2d(2, stride = 2)
        self.conv2 = nn.Conv2d(in_channels = 6, out_channels = 16, kernel_size = 5, stride = 1)
        self.pool = nn.MaxPool2d(2, stride = 2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.logSoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


'''Model class used for experimenting with BatchNorm layer, here BatchNorm is used only after the first layer 
without Learnable parameters '''

class LeNetExp(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 6, kernel_size = 5, stride = 1)
        self.conv1_bn=nn.BatchNorm2d(6,affine=False)
        self.pool = nn.MaxPool2d(2, stride = 2)
        self.conv2 = nn.Conv2d(in_channels = 6, out_channels = 16, kernel_size = 5, stride = 1)
        self.pool = nn.MaxPool2d(2, stride = 2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.logSoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.relu(self.conv1_bn(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

#Model class with BatchNorm layer after every ConvLayer and FCLayer

class LeNetBatch(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 6, kernel_size = 5, stride = 1)
        self.conv1_bn=nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(2, stride = 2)
        self.conv2 = nn.Conv2d(in_channels = 6, out_channels = 16, kernel_size = 5, stride = 1)
        self.conv2_bn=nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2, stride = 2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc1_bn=nn.BatchNorm1d(120)
        self.fc2 = nn.Linear(120, 84)
        self.fc2_bn=nn.BatchNorm1d(84)
        self.fc3 = nn.Linear(84, 10)
        self.logSoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.relu(self.conv1_bn(x)))
        x = self.conv2(x)
        x = self.pool(F.relu(self.conv2_bn(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.fc1(x)
        x = F.relu(self.fc1_bn(x))
        x = self.fc2(x)
        x = F.relu(self.fc2_bn(x))
        x = self.fc3(x)
        return x


def normalize(train_ds,test_ds,val_ds,test_loader,val_loader,train_loader):

    pixel_sum = torch.tensor([0.0, 0.0, 0.0])
    pixel_sum_sq = torch.tensor([0.0, 0.0, 0.0])

    for i, data in enumerate(train_loader, 0):

            inputs = data['image']
            pixel_sum += inputs.sum(axis = [0,2,3])
            pixel_sum_sq += (inputs ** 2).sum(axis = [0,2,3])

    count = len(train_ds) * 32 * 32

    total_mean = pixel_sum/count
    total_var = (pixel_sum_sq/count) - (total_mean ** 2)
    total_std = torch.sqrt(total_var)

    # print('mean:' + str(total_mean))
    # print('std:' + str(total_std))


    #The Dataset class used to load the images into the dataset

class lenetdatatset(Dataset):

    def __init__ (self,filename):
        with open(filename) as f:
            self.lines = f.readlines()

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index):
        im_lab = self.lines[index].split()
        img = Image.open('./'+ im_lab[0])
        transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(32), transforms.Normalize(mean = [0.4469, 0.4399, 0.4067], std = [0.2603, 0.2565, 0.2712])])
        img_tr = transform(img)
        label = l_num[im_lab[1]]
        #Each Dataset member is a dictionary with image,label pair
        sample = {'image': img_tr, 'label': label}
        return sample



#Plot confusion matrix as a heat map to analyze it better

def plot_cfm(labs,preds):

    cfmatrix = confusion_matrix(labs,preds)
    print('\nConfusion Matrix:\n',cfmatrix)

    df_cfm = pd.DataFrame(cfmatrix, index = classes, columns = classes)
    plt.figure(figsize = (20,14))
    cfm_plot = sn.heatmap(df_cfm, annot=True, fmt='.3g')
    return cfm_plot
    


#Plot the training loss and Validation loss

def plot_loss(train_loss,val_loss):

    plt.plot(train_loss, 'g', label='Training loss')
    plt.plot(val_loss, 'b', label='validation loss')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    return plt


#Function to train the model and check performance

def experiment(train_ds,test_ds,val_ds,test_loader,val_loader,net,optimizer,exp_name):

    train_loss = []
    val_loss = []

    #gamma indicates the decay of the learning rate after every 2 epoches
    scheduler  = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    criterion = nn.CrossEntropyLoss()
    print('TRAINING')

    #training the model
    for epoch in range(40):

        running_loss = 0.0
        validation_loss = 0.0
        
        #Shuffles the train data after epoch
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0)

        #Iterate over the batches and train the model
        for i, data in enumerate(train_loader, 0):

            inputs = data['image']
            labels = data['label']
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() #Keeping track of the training loss

        # checking validation loss for each epoch
        with torch.no_grad():

            for j, data in enumerate(val_loader, 0):

                inputs = data['image']
                labels = data['label']
                v_outputs = net(inputs)
                v_loss = criterion(v_outputs, labels)

                validation_loss += v_loss.item() #Keeping track of the validation loss

        #Keeping track of the avg loss of the current epoch over the different batches
        avg_train_loss = running_loss/len(train_loader) 
        avg_val_loss = validation_loss/len(val_loader)
        print('\nEPOCH: {}/40'.format(epoch))
        print('Train Loss: {:.4f}    Validation Loss: {:.4f}'.format(avg_train_loss,avg_val_loss))
        train_loss.append(avg_train_loss)
        val_loss.append(avg_val_loss)

    #plotting losses
    plot = plot_loss(train_loss,val_loss)
    plot.savefig(exp_name+'loss.png')

    #testing on test data
    correct = 0
    total = 0
    wcount = 0
    preds = []
    labs = []

    #class wise accuracy and confusion matrix for the whole classification
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    #Using test data to check the performance of the model
    with torch.no_grad():
        
        #Iterating over the batches
        for data in test_loader:
            count = 0
            inputs = data['image']
            labels = data['label']
            outputs = net(inputs)
            
            #Getting the max to assign the image to a class
            _, predictions = torch.max(outputs, 1) 

            preds.extend(predictions.numpy())
            labs.extend(labels.numpy())
            
            # collect the correct predictions for each class
            for idx, label, prediction in zip(range(labels.size(dim=0)),labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                    correct += 1
                else:
                    if count == 0 and wcount<5:
                        wimage = inputs[idx]
                        torchvision.utils.save_image(wimage, 'wrong'+str(wcount)+exp_name+'.png')
                        count += 1
                        wcount += 1
                total_pred[classes[label]] += 1
                total += 1

    print('\nAccuracy of the network on the test images: %d %%' % (100 * correct / total),'\n')
    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print("Accuracy for class {:5s} is: {:.1f} %".format(classname,
                                                   accuracy))
    #confusion matrix
    cfm_plot = plot_cfm(labs,preds)
    cfm_plot.figure.savefig(exp_name+'cfm.png')


if __name__ == "__main__":

    #Loading the dataset
    train_ds = lenetdatatset(filename = 'splits/train.txt')
    test_ds = lenetdatatset(filename = 'splits/test.txt')
    val_ds = lenetdatatset(filename = 'splits/val.txt')
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=128, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=128, num_workers=0)
    
    #Main model
    print('\nMain experiment\n')
    exp_name = 'main'
    net = LeNet()
    optimizer  = optim.Adam(net.parameters(),lr=1e-3)
    experiment(train_ds,test_ds,val_ds,test_loader,val_loader,net,optimizer,exp_name)
    
    #Experiment with L2 norm(decay = 1e-3)
    print('\nL2 norm experiment\n')
    exp_name = 'L2norm'
    netl2 = LeNet()
    optimizer = optim.Adam(netl2.parameters(),lr = 1e-3, weight_decay = 1e-3)
    experiment(train_ds,test_ds,val_ds,test_loader,val_loader,netl2,optimizer,exp_name)
    

    #Experiment with BatchNorm after all the ConvLayers and FC layers
    print('\nBatch Normalization\n')
    exp_name = 'batch'
    netbatch = LeNetBatch()
    optimizer  = optim.Adam(netbatch.parameters(),lr=1e-3)
    experiment(train_ds,test_ds,val_ds,test_loader,val_loader,netbatch,optimizer,exp_name)
    
    #Experiment with Batch Normalization only after the first ConvLayer
    print('\nBatch Normalization only after the first convolution layer\n')
    exp_name = 'batchcl1'
    netbatch = LeNetExp()
    optimizer  = optim.Adam(netbatch.parameters(),lr=1e-3)
    experiment(train_ds,test_ds,val_ds,test_loader,val_loader,netbatch,optimizer,exp_name)
    