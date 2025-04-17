import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

# If have NVIDIA GPU, see https://pytorch.org/get-started/locally/ for correct CUDA version
# After installing the correct version of CUDA, may need to do something like this:
# >>pip install torch==2.5.0+cu118 torchvision==0.20.0+cu118 torchaudio==2.5.0+cu118 --index-url https://download.pytorch.org/whl/cu118
#   I am using Cuda 11.8 so in the command I use (cu118) for the packages and the pytorch wheel
#   If you use 12.1 it would be cu121

# Base class to inherit training/plotting functionality
class DLN_Base(nn.Module):
    def __init__(self,dev):
        super().__init__()
        self.dev = dev
        self.tlosslist = None
        self.vlosslist = None
        self.epoch = 0
        self.best_epoch = 0

    def loss_batch(self,loss_func,xb,yb,opt=None):
        # xb will be of shape bs x 60 in our first test dataset
        # yb will be bs x 1
        loss = loss_func(self(xb),yb)
        if opt is not None:
            loss.backward()  # perform backpropogation
            opt.step()  # take an optimization step
            opt.zero_grad()  # reset gradients to none

        return loss.item()  # get loss value from gpu

    def Augment(self, xb, yb):
        return xb, yb

    def fit(self,epochs,loss_func,opt,train_D,train_y,valid_D,valid_y,bs,savebest=None,plotType='log'):
        # train_D and train_y are torch tensors sitting on CPU
        # valid_D and valid_y are torch tensors sitting on GPU
        N = train_y.size()[0]
        NB = np.ceil(N / bs).astype(np.longlong)
        tlosslist = []
        vlosslist = []
        best_val_loss = np.inf
        for self.epoch in range(epochs):
            self.train() # put model in training mode
            losslist = []
            for i in range(NB):
                start_i = i * bs
                end_i = start_i + bs

                xb, yb = self.Augment(train_D[start_i:end_i].to(self.dev), train_y[start_i:end_i].to(self.dev))
                loss = self.loss_batch(loss_func,xb,yb,opt) # take gradient descent step
                losslist.append(loss)

            self.eval() # put model in validation mode
            with torch.no_grad():
                val_loss = self.loss_batch(loss_func,valid_D,valid_y)

            if val_loss<best_val_loss:
                best_val_loss = val_loss
                self.best_epoch = self.epoch
                if savebest is not None:
                    torch.save(self,savebest)

            tlosslist.append(np.mean(losslist))
            vlosslist.append(val_loss)
            plt.cla()
            plt.plot(np.linspace(1,self.epoch + 1,self.epoch + 1),tlosslist,'r',label='Training')
            plt.plot(np.linspace(1,self.epoch + 1,self.epoch + 1),vlosslist,'g',label='Validation')
            plt.plot(self.best_epoch + 1,best_val_loss,'b*',label='Best result')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            if plotType is not None:
                plt.yscale(plotType)
                plt.pause(.01)  # comment out if live plotting not necessary



        self.tlosslist = tlosslist
        self.vlosslist = vlosslist


# human activity dataset, samples of 60 inertial unit measurements taken during 5 activities:
#  (sitting, standing, walking, running, dancing)
f = open('EECE_395/chiasm.json','rt')
d = json.load(f)

# Split the json data into training, validation, and test sets using 80/10/10 split
dev = 'cuda'

D = torch.tensor(np.array(d['D'], dtype=np.float32))
y = torch.tensor(np.array(d['y'], dtype=np.float32))

testf = 1/10
validf = 1/10
trainf = 1 - testf - validf
lr = 0.1

# Create actual splits based on these fractions
N = len(y)
indices = np.random.permutation(N)
test_size = int(N * testf)
valid_size = int(N * validf)
train_size = N - test_size - valid_size

test_indices = indices[:test_size]
valid_indices = indices[test_size:test_size+valid_size]
train_indices = indices[test_size+valid_size:]

train_D = D[train_indices].to(dev)
train_y = y[train_indices].to(dev)
valid_D = D[valid_indices].to(dev)
valid_y = y[valid_indices].to(dev)
test_D = D[test_indices].to(dev)
test_y = y[test_indices].to(dev)

bs = len(d['y']) # 500

epochs = 300
dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# dtype = np.float32
# D = torch.tensor(np.array(dt['D'], dtype=dtype))
# y = torch.tensor(np.array(dt['y'], dtype=np.longlong) - 1)
# 
# D = torch.tensor(np.array(dt['train_D'],dtype=dtype)[0:500])
# y = torch.tensor(np.array(dt['train_y'],dtype=np.longlong)[0:500] - 1)
# class labels 0 to 4

# Dv = torch.tensor(np.array(dt['validation_D'], dtype=dtype), device=dev)
# yv = torch.tensor(np.array(dt['validation_y'], dtype=np.longlong) - 1, device=dev)# class labels 0 to 4

# what we need to do to create a new DLN class:
class DLN_examp(DLN_Base):
    def __init__(self,dev):
        super().__init__(dev)
        # define layers
    def forward(self, xb):
        pass
        # define how layers are connected and data sent through

# simple fully connected network:
class DLN(DLN_Base):
    def __init__(self,dev):
        super().__init__(dev)
        feature_channels = 64
        self.fc1 = nn.Linear(60, feature_channels)
        self.fc2 = nn.Linear(feature_channels,feature_channels)
        self.fc3 = nn.Linear(feature_channels,feature_channels)
        self.fc4 = nn.Linear(feature_channels,feature_channels)
        self.fc5 = nn.Linear(feature_channels,5)

        self.rm1 = torch.tensor(np.zeros(feature_channels, dtype=np.float32), device=dev)
        self.rv1 = torch.tensor(np.zeros(feature_channels,dtype=np.float32),device=dev)
        self.w1 = torch.tensor(np.ones(feature_channels,dtype=np.float32),device=dev)
        self.b1 = torch.tensor(np.zeros(feature_channels,dtype=np.float32),device=dev)

        self.rm2 = torch.tensor(np.zeros(feature_channels,dtype=np.float32),device=dev)
        self.rv2 = torch.tensor(np.zeros(feature_channels,dtype=np.float32),device=dev)
        self.w2 = torch.tensor(np.ones(feature_channels,dtype=np.float32),device=dev)
        self.b2 = torch.tensor(np.zeros(feature_channels,dtype=np.float32),device=dev)

        self.rm3 = torch.tensor(np.zeros(feature_channels,dtype=np.float32),device=dev)
        self.rv3 = torch.tensor(np.zeros(feature_channels,dtype=np.float32),device=dev)
        self.w3 = torch.tensor(np.ones(feature_channels,dtype=np.float32),device=dev)
        self.b3 = torch.tensor(np.zeros(feature_channels,dtype=np.float32),device=dev)

        self.rm4 = torch.tensor(np.zeros(feature_channels,dtype=np.float32),device=dev)
        self.rv4 = torch.tensor(np.zeros(feature_channels,dtype=np.float32),device=dev)
        self.w4 = torch.tensor(np.ones(feature_channels,dtype=np.float32),device=dev)
        self.b4 = torch.tensor(np.zeros(feature_channels,dtype=np.float32),device=dev)

    def forward(self, xb):
        # FC linear only
        # xb = self.fc1(xb)
        # xb = self.fc2(xb)
        # xb = self.fc3(xb)
        # xb = self.fc4(xb)

        # FC + ReLU
        # xb = F.relu(self.fc1(xb))
        # xb = F.relu(self.fc2(xb))
        # xb = F.relu(self.fc3(xb))
        # xb = F.relu(self.fc4(xb))

        # FC + ReLU + BN
        xb = F.relu(F.batch_norm(self.fc1(xb), running_mean=self.rm1, running_var=self.rv1,
                    weight=self.w1, bias=self.b1, training=self.training))
        xb = F.relu(F.batch_norm(self.fc2(xb), running_mean=self.rm2, running_var=self.rv2,
                    weight=self.w2, bias=self.b2, training=self.training))
        xb = F.relu(F.batch_norm(self.fc3(xb), running_mean=self.rm3, running_var=self.rv3,
                    weight=self.w3, bias=self.b3, training=self.training))
        xb = F.relu(F.batch_norm(self.fc4(xb), running_mean=self.rm4, running_var=self.rv4,
                    weight=self.w4, bias=self.b4, training=self.training))

        xb = self.fc5(xb)
        return F.softmax(xb, dim=1)

model = DLN(dev)
model.to(dev)
print(model)

opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
model.eval()
loss_func = torch.nn.functional.cross_entropy
# print(loss_func(model(valid_y), test_y))

fig, ax = plt.subplots()
# uncomment to re-run training
# model.fit(epochs, loss_func, opt, D, y, Dv, yv, bs, 'FCN_HumAct_3.pth')

# model = torch.load('FCN_HumAct_3.pth')
model.eval()


Dt = torch.tensor(np.array(d['test_D'], dtype=np.float32), device=dev)
yt = torch.tensor(np.array(d['test_y'], dtype=np.longlong) - 1, device=dev)# class labels 0 to 4

y_pred = torch.argmax(model(Dt), axis=1)
acc = 100 * torch.sum( yt == y_pred).item()/yt.size()[0] # Bug from class: had y as denominator here and y is 2x yt
print(acc)
# linear only acc = 82.9%
# linear + relu acc = 95.9%
# linear + relu + BN = 97.8%


# load existing architecture
# can do with pre-trained weights if needed
import torchvision
model = torchvision.models.resnet18()
print(model)

# change input to 1 channel grayscale
model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7,7), stride=(1,1), padding=(3,3), bias=False)
# predict 10 classification labels instead of 1000
model.fc = torch.nn.Linear(512, 10)

# plt.ioff()
# plt.show()




# 3D U-Net solution
class uNet3D(DLN_Base):
    def __init__(self,dev,inChannels=1,outChannels=1,weight=1.,basenumfilt=16,filtsz=(3,3,3)):
        super().__init__(dev)

        self.weight = torch.tensor(weight, device=dev)

        self.cv11 = torch.nn.Conv3d(inChannels,basenumfilt,kernel_size=filtsz,stride=(1,1,1),padding=(1,1,1))
        self.cv12 = torch.nn.Conv3d(basenumfilt,basenumfilt,kernel_size=filtsz,stride=(1,1,1),padding=(1,1,1))
        self.cv21 = torch.nn.Conv3d(basenumfilt,basenumfilt * 2,kernel_size=filtsz,stride=(1,1,1),padding=(1,1,1))
        self.cv22 = torch.nn.Conv3d(basenumfilt * 2,basenumfilt * 2,kernel_size=filtsz,stride=(1,1,1),padding=(1,1,1))
        self.cv31 = torch.nn.Conv3d(basenumfilt * 2,basenumfilt * 2,kernel_size=filtsz,stride=(1,1,1),padding=(1,1,1))
        self.cv32 = torch.nn.Conv3d(basenumfilt * 2,basenumfilt * 2,kernel_size=filtsz,stride=(1,1,1),padding=(1,1,1))
        self.cv41 = torch.nn.Conv3d(basenumfilt * 4,basenumfilt * 2,kernel_size=filtsz,stride=(1,1,1),padding=(1,1,1))
        self.cv42 = torch.nn.Conv3d(basenumfilt * 2,basenumfilt * 2,kernel_size=filtsz,stride=(1,1,1),padding=(1,1,1))
        self.cv43 = torch.nn.Conv3d(basenumfilt * 3,basenumfilt * 2,kernel_size=filtsz,stride=(1,1,1),padding=(1,1,1))
        self.cv44 = torch.nn.Conv3d(basenumfilt * 2,basenumfilt * 2,kernel_size=filtsz,stride=(1,1,1),padding=(1,1,1))
        self.cv45 = torch.nn.Conv3d(basenumfilt * 2,outChannels,kernel_size=filtsz,stride=(1,1,1),padding=(1,1,1))

        self.rm1 = torch.tensor(np.zeros(basenumfilt,dtype=np.float32)).to(dev)
        self.rv1 = torch.tensor(np.zeros(basenumfilt,dtype=np.float32)).to(dev)
        self.w1 = torch.tensor(np.ones(basenumfilt,dtype=np.float32)).to(dev)
        self.b1 = torch.tensor(np.zeros(basenumfilt,dtype=np.float32)).to(dev)
        self.rm2 = torch.tensor(np.zeros(basenumfilt,dtype=np.float32)).to(dev)
        self.rv2 = torch.tensor(np.zeros(basenumfilt,dtype=np.float32)).to(dev)
        self.w2 = torch.tensor(np.ones(basenumfilt,dtype=np.float32)).to(dev)
        self.b2 = torch.tensor(np.zeros(basenumfilt,dtype=np.float32)).to(dev)
        self.rm3 = torch.tensor(np.zeros(basenumfilt * 2,dtype=np.float32)).to(dev)
        self.rv3 = torch.tensor(np.zeros(basenumfilt * 2,dtype=np.float32)).to(dev)
        self.w3 = torch.tensor(np.ones(basenumfilt * 2,dtype=np.float32)).to(dev)
        self.b3 = torch.tensor(np.zeros(basenumfilt * 2,dtype=np.float32)).to(dev)
        self.rm4 = torch.tensor(np.zeros(basenumfilt * 2,dtype=np.float32)).to(dev)
        self.rv4 = torch.tensor(np.zeros(basenumfilt * 2,dtype=np.float32)).to(dev)
        self.w4 = torch.tensor(np.ones(basenumfilt * 2,dtype=np.float32)).to(dev)
        self.b4 = torch.tensor(np.zeros(basenumfilt * 2,dtype=np.float32)).to(dev)
        self.rm5 = torch.tensor(np.zeros(basenumfilt * 2,dtype=np.float32)).to(dev)
        self.rv5 = torch.tensor(np.zeros(basenumfilt * 2,dtype=np.float32)).to(dev)
        self.w5 = torch.tensor(np.ones(basenumfilt * 2,dtype=np.float32)).to(dev)
        self.b5 = torch.tensor(np.zeros(basenumfilt * 2,dtype=np.float32)).to(dev)

        self.apply(self.init_weights)  ###

    def forward(self,xb):
        stride = (2,2,2)
        reluscale = .2
        m = nn.LeakyReLU(reluscale)
        mp = nn.MaxPool3d(kernel_size=2,stride=stride)
        us = nn.Upsample(scale_factor=2,mode='trilinear')  #
        bn = nn.functional.batch_norm


        xb = m(bn(self.cv11(xb),running_mean=self.rm1,running_var=self.rv1,weight=self.w1,bias=self.b1,
                  training=self.training))
        xb = m(self.cv12(xb))
        xb2 = bn(mp(xb),running_mean=self.rm2,running_var=self.rv2,weight=self.w2,bias=self.b2,training=self.training)
        xb2 = m(bn(self.cv21(xb2),running_mean=self.rm3,running_var=self.rv3,weight=self.w3,bias=self.b3,
                   training=self.training))
        xb2 = m(self.cv22(xb2))

        # bottleneck
        xb3 = bn(mp(xb2),running_mean=self.rm4,running_var=self.rv4,weight=self.w4,bias=self.b4,training=self.training)
        xb3 = m(bn(self.cv31(xb3),running_mean=self.rm5,running_var=self.rv5,weight=self.w5,bias=self.b5,
                   training=self.training))
        xb3 = m(self.cv32(xb3))

        xb3 = us(xb3)
        xb4 = torch.cat((xb3,xb2),dim=1)  ###
        xb4 = m(self.cv41(xb4))
        xb4 = m(self.cv42(xb4))  ###
        xb4 = us(xb4)
        xb5 = torch.cat((xb4,xb),dim=1)  ###
        xb5 = m(self.cv43(xb5))
        xb6 = m(self.cv44(xb5))  ###
        xb6 = self.cv45(xb6)

        sm = torch.nn.Sigmoid()
        return sm(xb6)


    def init_weights(self,m):
        if isinstance(m,nn.Conv3d):
            nn.init.xavier_normal_(m.weight)

    def myLoss(self,ypred,y):
        v1 = 1
        v2 = 0.001
        f = torch.nn.functional.binary_cross_entropy_with_logits
        intersection = -torch.sum(ypred * y)  # autograd compatible
        # loss = v1 * f(ypred,y,pos_weight=self.weight) #+ v2 * intersection
        # intersection.backward()
        vol = torch.mean(torch.sum(ypred, dim=[2,3,4])) # autograd compatible; could do self supervision comparing to target volume
        # in the project you will implement self supervision comparing the predictions to the mean shape of the chiasm in the training dataset
        # example of scheduling the BCE positive class weight
        if self.epoch<80:
            bcew = torch.tensor([100 - self.epoch], device=self.dev)
        else:
            bcew = torch.tensor([20.], device=self.dev)
        loss = v1 * f(ypred,y,pos_weight=self.weight)  + v2 * intersection
        return loss

    #example of basic augmentation
    def Augment(self,D,y):
        # rotate +/- 90 degrees about the z axis
        Dn = torch.cat((D,torch.flip(torch.swapaxes(D,2,3),[2])),dim=0)
        yn = torch.cat((y,torch.flip(torch.swapaxes(y,2,3),[2])),dim=0)
        Dn = torch.cat((Dn,torch.flip(torch.swapaxes(D,2,3),[3])),dim=0)
        yn = torch.cat((yn,torch.flip(torch.swapaxes(y,2,3),[3])),dim=0)

        return Dn,yn

# try the UNet on a small Chiasm dataset
f = open('..\\Chiasm.json')
d = json.load(f)
f.close()

D = np.array(d['D'], dtype=np.float32)
y = np.array(d['y'], dtype=np.float32)

testf = 1/10
validf = 1/10
trainf = 1 - testf - validf

# example of randomized sampling of data into training/validation/test sets
rng = np.random.default_rng(0)
rand_ord = np.argsort(rng.uniform(size=np.shape(D)[0])).astype(np.int32)
train_last = np.ceil(np.shape(D)[0] * trainf).astype(np.int32)
train_indx = rand_ord[0:train_last]

valid_last = np.ceil(np.shape(D)[0] * validf).astype(np.int32) + train_last
valid_indx = rand_ord[train_last:valid_last]

test_indx = rand_ord[valid_last::]

train_D = torch.tensor(D[train_indx])
train_y = torch.tensor(y[train_indx, np.newaxis, :, :, :])
valid_D = torch.tensor(D[valid_indx], device=dev)
valid_y = torch.tensor(y[valid_indx, np.newaxis, :, :, :], device=dev)
test_D = torch.tensor(D[test_indx])
test_y = torch.tensor(y[test_indx, np.newaxis, :, :, :])

# check for class imbalance
weight_f = torch.sum(train_y==False)/torch.sum(train_y)
print(f'weight: {weight_f}')
# many more foreground than background voxels

m = uNet3D(dev, weight = weight_f)
m.to(dev)
lr = 5e-3
bs = 50
opt = torch.optim.SGD(m.parameters(), lr=lr, weight_decay=3e-5, momentum=.9, nesterov=True)

# loss_func = nn.functional.binary_cross_entropy_with_logits

loss_func = m.myLoss

fig, ax = plt.subplots()
plt.ion()
plt.pause(.03)

# uncomment to re-run UNet training
# m.fit(100, loss_func, opt, train_D, train_y, valid_D, valid_y, bs, 'ChiasmUNet_BCEpInt.pth', plotType='log')

model = torch.load('ChiasmUNet_BCEpInt.pth')
model.eval()
Dt = test_D.to(dev)
yt = test_y.to(dev)
ypred = model(Dt) >= 0.5
acc = 100*torch.sum(yt==ypred).item()/ torch.prod(torch.tensor(yt.size()))
print(acc)

dice = 2 * torch.sum(ypred * yt )/(torch.sum(ypred) + torch.sum(yt))
print(dice)
print(torch.sum(ypred))

fig, ax = plt.subplots(1,2)
ax[0].imshow(np.squeeze(test_D[0,0,:,:,3]).T, 'gray', vmin=0, vmax=100)
X, Y = np.meshgrid(range(32), range(32))
plt.axes(ax[0])
plt.contour(X,Y, test_y[0,0,:,:,3].numpy().T, levels=[0.5], colors='red' )
ax[1].imshow(np.squeeze(ypred[0,0,:,:,3].detach().cpu().numpy()).T, 'gray')
plt.axes(ax[1])
plt.contour(X,Y, test_y[0,0,:,:,3].numpy().T, levels=[0.5], colors='red' )
plt.ioff()
plt.show()
# exit(0)


# using nnUNet:
# pip install nnunetv2

# define needed environment variables
import os
os.environ['nnUNet_raw'] = 'nnUNet_raw'
os.environ['nnUNet_preprocessed'] = 'nnUNet_preprocessed'
os.environ['nnUNet_results'] = 'nnUNet_results'

# creating the raw dataset:
dr = 'nnUNet_raw\\Dataset001_Chiasm\\'

dict = {
    'channel_names': {"0": "CT"},
    'labels': {
        'background': 0,
        'Chiasm': 1
    },
    'numTraining': 48,
    'file_ending': '.nrrd'
}

f = open(f'{dr}dataset.json','wt')
json.dump(dict, f)
f.close()

import nrrd
for i in range(len(D)):
    header = {'kinds': ['domain', 'domain', 'domain'], 'space': 'left-posterior-superior',
              'space directions': np.array([[1,0,0], [0,1,0], [0,0,1]]), 'encoding': 'ASCII'}
    if i<9:
        fl = f'{dr}imagesTr\\Chiasm_00{i+1}_0000.nrrd'
        ll = f'{dr}labelsTr\\Chiasm_00{i + 1}.nrrd'
    elif i<99:
        fl = f'{dr}imagesTr\\Chiasm_0{i + 1}_0000.nrrd'
        ll = f'{dr}labelsTr\\Chiasm_0{i + 1}.nrrd'
    else:
        fl = f'{dr}imagesTr\\Chiasm_{i+1}_0000.nrrd'
        ll = f'{dr}labelsTr\\Chiasm_{i + 1}.nrrd'

    nrrd.write(fl, D[i, 0,:,:,:], header)
    nrrd.write(ll, y[i,:,:,:],header)


# Run dataset fingerprint extraction
os.system('nnUNetv2_plan_and_preprocess -d 001 --verify_dataset_integrity')

# Train 2d network, fold 0
os.system('nnUNetv2_train 001 2d 0 --npz')

