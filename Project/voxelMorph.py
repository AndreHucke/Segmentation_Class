import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import scipy
from volumeViewer import *
import os
import nrrd
from pathlib import Path

# modified for self-supervised learning and dataloader
class DLN_Base(nn.Module):
    def __init__(self,dev):
        super().__init__()
        self.dev = dev
        self.tlosslist = None
        self.vlosslist = None
        self.epoch = 0
        self.best_epoch = 0
        self.plotType='log'

    def loss_batch(self,loss_func,xb,yb=None,opt=None):
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

    def fit(self,epochs,loss_func,opt,train_D,train_y=None,valid_D=None,valid_y=None,bs=100,savebest=None,plotType='log'):
        # train_D is dataloader on CPU
        # valid_D are torch tensors sitting on GPU
        # N = len(train_D)
        self.plotType = plotType
        NB = len(train_D)
        tlosslist = []
        vlosslist = []
        best_val_loss = np.inf
        for self.epoch in range(epochs):
            self.train() # put model in training mode
            losslist = []
            for i in range(NB):
                # start_i = i * bs
                # end_i = start_i + bs

                xb = next(iter(train_D)).to(self.dev)
                loss = self.loss_batch(loss_func,xb,opt=opt)  # take gradient descent step
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
            self.Plot( self.epoch, tlosslist, vlosslist, self.best_epoch, best_val_loss,valid_D)
        self.tlosslist = tlosslist
        self.vlosslist = vlosslist

    def Plot( self,epoch, tlosslist, vlosslist, best_epoch, best_val_loss, valid_D):
        plt.cla()
        plt.plot(np.linspace(1,epoch + 1,epoch + 1),tlosslist,'r',label='Training')
        plt.plot(np.linspace(1,epoch + 1,epoch + 1),vlosslist,'g',label='Validation')
        plt.plot(best_epoch + 1,best_val_loss,'b*',label='Best result')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        if self.plotType is not None:
            plt.yscale(self.plotType)
        plt.pause(.01)  # comment out if live plotting not necessary



# Create the VoxelMorph network based on U-Net, inheriting from DLN_Base
class RegNet(DLN_Base):
    def __init__(self,dev, atlas):
        super().__init__(dev)
        basenumfilt = 16
        filtsz = (3, 3, 3)


        self.cv11 = torch.nn.Conv3d(1, basenumfilt, kernel_size=filtsz, stride=(1,1,1), padding=(1, 1, 1))
        self.cv12 = torch.nn.Conv3d(basenumfilt, basenumfilt, kernel_size=filtsz, stride=(1,1, 1), padding=(1, 1, 1))
        self.cv21 = torch.nn.Conv3d(basenumfilt, basenumfilt * 2, kernel_size=filtsz, stride=(1,1, 1), padding=(1, 1, 1))
        self.cv22 = torch.nn.Conv3d(basenumfilt * 2, basenumfilt * 2, kernel_size=filtsz, stride=(1, 1, 1), padding=(1, 1, 1))
        self.cv31 = torch.nn.Conv3d(basenumfilt * 2, basenumfilt * 2, kernel_size=filtsz, stride=(1, 1, 1), padding=(1, 1, 1))
        self.cv32 = torch.nn.Conv3d(basenumfilt * 2, basenumfilt * 2, kernel_size=filtsz, stride=(1, 1, 1), padding=(1, 1, 1))
        self.cv41 = torch.nn.Conv3d(basenumfilt * 2, basenumfilt * 2, kernel_size=filtsz, stride=(1, 1, 1), padding=(1, 1, 1))
        self.cv42 = torch.nn.Conv3d(basenumfilt * 2, basenumfilt * 2, kernel_size=filtsz, stride=(1, 1, 1), padding=(1, 1, 1))
        self.cv51 = torch.nn.Conv3d(basenumfilt * 2, basenumfilt * 2, kernel_size=filtsz, stride=(1, 1, 1), padding=(1, 1, 1))
        self.cv52 = torch.nn.Conv3d(basenumfilt * 2, basenumfilt * 2, kernel_size=filtsz, stride=(1, 1, 1), padding=(1, 1, 1))
        self.cv61 = torch.nn.Conv3d(basenumfilt * 4, basenumfilt * 2, kernel_size=filtsz, stride=(1, 1, 1), padding=(1, 1, 1))
        self.cv62 = torch.nn.Conv3d(basenumfilt * 2, basenumfilt * 2, kernel_size=filtsz, stride=(1, 1, 1), padding=(1, 1, 1))
        self.cv63 = torch.nn.Conv3d(basenumfilt * 3, basenumfilt * 2, kernel_size=filtsz, stride=(1, 1, 1), padding=(1, 1, 1))
        self.cv64 = torch.nn.Conv3d(basenumfilt * 2, basenumfilt * 2, kernel_size=filtsz, stride=(1, 1, 1), padding=(1, 1, 1))
        self.cv65 = torch.nn.Conv3d(basenumfilt * 2, basenumfilt, kernel_size=filtsz, stride=(1, 1, 1), padding=(1, 1, 1))
        self.cv66 = torch.nn.Conv3d(basenumfilt, basenumfilt,kernel_size=filtsz,stride=(1,1,1),padding=(1,1,1))
        self.cv67 = torch.nn.Conv3d(basenumfilt,3,kernel_size=filtsz,stride=(1,1,1),padding=(1,1,1))
        self.apply(self.init_weights)

        # pre-define atlas for loss function computing
        self.A = torch.tensor(atlas, device=dev) # ranges from 0~4000
        # level set inspired loss datastructures
        self.Aair = torch.where(self.A < 500)
        self.Asoft = torch.where((self.A>=500) & (self.A<1500))
        self.Abone = torch.where(self.A>=1500)

        self.D = None
        # normalize atlas for cross correlation
        self.mnA = torch.mean(self.A)
        self.A -= self.mnA
        self.A /= torch.linalg.vector_norm(self.A)

        # pre-define grid variables for deform function
        xs = np.linspace(0, np.shape(atlas)[0]-1, np.shape(atlas)[0])
        ys = np.linspace(0,np.shape(atlas)[1] - 1,np.shape(atlas)[1])
        zs = np.linspace(0,np.shape(atlas)[2] - 1,np.shape(atlas)[2])
        X,Y,Z = np.meshgrid(xs,ys,zs, indexing='ij')
        self.X = torch.tensor(X, device=dev, dtype=torch.float32)
        self.Y = torch.tensor(Y,device=dev,dtype=torch.float32)####
        self.Z = torch.tensor(Z,device=dev,dtype=torch.float32)

        # initialize plot variables
        self.defslc = np.shape(atlas)[2]//2 + 10
        self.fig = plt.gcf()
        fig, ax = plt.subplots()
        ax.imshow(atlas[:,:,self.defslc].T, 'gray')
        self.cntr1 = plt.contour(X[:,:,self.defslc], Y[:,:, self.defslc],
                                 atlas[:,:,self.defslc].squeeze(), levels=[500])
        self.cntr2 = plt.contour(X[:,:,self.defslc],Y[:,:,self.defslc],
                                 atlas[:,:,self.defslc].squeeze(),levels=[1500])

        plt.figure(self.fig)
        plt.clf()
        self.ax = np.zeros(5, dtype=plt.Axes)
        self.ax[0] = self.fig.add_subplot(1,3,1)
        self.ax[1] = self.fig.add_subplot(2,3,2)
        self.ax[2] = self.fig.add_subplot(2,3,3)
        self.ax[3] = self.fig.add_subplot(2,3,5)
        self.ax[4] = self.fig.add_subplot(2,3,6)

    def Plot(self, epoch, tlosslist, vlosslist, best_epoch, best_val_loss, valid_D, rind=-1):
        plt.figure(self.fig)
        plt.axes(self.ax[0])
        super().Plot( epoch, tlosslist, vlosslist, best_epoch, best_val_loss,valid_D)

        if epoch == best_epoch:
            if rind<0:
                rind = np.floor(np.random.uniform(low=0, high=valid_D.size(0)-1)).astype(np.longlong)
            vy = self(valid_D[rind,:,:,:,:][np.newaxis,:,:,:,:])
            plt.axes(self.ax[1])
            plt.cla()
            self.ax[1].imshow(valid_D[rind,0,:,:,self.defslc].cpu().detach().numpy().squeeze().T,
                              'gray')
            plt.title(f'Validation case {rind}')

            df = self.Deform(vy[0,:,:,:,:,][np.newaxis,:,:,:,:],
                             valid_D[rind,:,:,:,:][np.newaxis,:,:,:,:])
            plt.axes(self.ax[2])
            plt.cla()
            self.ax[2].imshow(df[0,0,:,:,self.defslc].cpu().detach().numpy().squeeze().T,
                              'gray')
            plt.title(f'target to atlas')
            for i in self.cntr1.allsegs[0]:
                plt.plot(i[:,0], i[:,1], color=[1,0,0])
            for i in self.cntr2.allsegs[0]:
                plt.plot(i[:,0], i[:,1], color=[0,0,1])

            plt.axes(self.ax[3])
            plt.cla()
            self.ax[3].imshow(vy[0,0,:,:,self.defslc].cpu().detach().numpy().squeeze().T, 'gray')
            plt.title('Def_x')
            plt.axes(self.ax[4])
            plt.cla()
            self.ax[4].imshow(vy[0,1,:,:,self.defslc].cpu().detach().numpy().squeeze().T,'gray')
            plt.title('Def_y')

            plt.pause(.01)


    def forward(self,xb):
        self.D = xb.clone()### store for use in loss function
        stride = (2, 2, 2)
        reluscale = .2
        m = nn.LeakyReLU(reluscale)
        mp = nn.MaxPool3d(kernel_size=2,stride=stride)
        us = nn.Upsample(scale_factor=2, mode='trilinear')
        xb = m(self.cv11(xb))
        xb = m(self.cv12(xb))
        xb2 = mp(xb)
        xb2 = m(self.cv21(xb2))
        xb2 = m(self.cv22(xb2))

        xb4 = mp(xb2)
        xb4 = m(self.cv41(xb4))
        xb4 = m(self.cv42(xb4))
        xb4 = us(xb4)

        xb6 = torch.cat((xb4, xb2), dim=1)
        xb6 = m(self.cv61(xb6))
        xb6 = m(self.cv62(xb6)) + xb4
        xb6 = us(xb6)
        xb7 = torch.cat((xb6, xb), dim=1)
        xb7 = m(self.cv63(xb7))
        xb7 = m(self.cv64(xb7)) + xb6 # residual decoder
        xb7 = m(self.cv65(xb7))
        xb7 = m(self.cv66(xb7))
        xb7 = self.cv67(xb7)

        # return multiple decoder levels for deep supervision loss
        if self.training:
            return xb7, xb6, xb4
        else:
            return xb7

    def init_weights(self, m):
        if isinstance(m, nn.Conv3d):
            nn.init.xavier_normal_(m.weight)

    def myLoss(self, ypred_t, y):
        v1 = 1e-8
        v2 = 0.1
        v3 = 1e-7
        v4 = 5e-9
        v5 = 2e-9
        if self.training:
            ypred = ypred_t[0]
            xb6 = ypred_t[1]
            xb4 = ypred_t[2]
        else:
            ypred = ypred_t
        vsz = np.array(ypred.size())

        # final smoothness loss
        gradloss = v1 * (torch.sum((ypred[:, :, 0:-1, :, :] - ypred[:,:,1:,:,:])*
            (ypred[:,:,0:-1,:,:] - ypred[:,:,1:,:,:])) +
            torch.sum((ypred[:,:,:,0:-1,:] - ypred[:,:,:,1:,:]) *
            (ypred[:,:,:,0:-1,:] - ypred[:,:,:,1:,:])) +
            torch.sum((ypred[:,:,:,:,0:-1] - ypred[:,:,:,:,1:]) *
            (ypred[:,:,:,:,0:-1] - ypred[:,:,:,:,1:])) )

        if self.training: # deep supervision losses
            gradloss2 = v4 * (torch.sum((xb6[:,:,0:-1,:,:] - xb6[:,:,1:,:,:]) *
                                       (xb6[:,:,0:-1,:,:] - xb6[:,:,1:,:,:])) +
                             torch.sum((xb6[:,:,:,0:-1,:] - xb6[:,:,:,1:,:]) *
                                       (xb6[:,:,:,0:-1,:] - xb6[:,:,:,1:,:])) +
                             torch.sum((xb6[:,:,:,:,0:-1] - xb6[:,:,:,:,1:]) *
                                       (xb6[:,:,:,:,0:-1] - xb6[:,:,:,:,1:])))
            gradloss3 = v5 * (torch.sum((xb4[:,:,0:-1,:,:] - xb4[:,:,1:,:,:]) *
                                        (xb4[:,:,0:-1,:,:] - xb4[:,:,1:,:,:])) +
                              torch.sum((xb4[:,:,:,0:-1,:] - xb4[:,:,:,1:,:]) *
                                        (xb4[:,:,:,0:-1,:] - xb4[:,:,:,1:,:])) +
                              torch.sum((xb4[:,:,:,:,0:-1] - xb4[:,:,:,:,1:]) *
                                        (xb4[:,:,:,:,0:-1] - xb4[:,:,:,:,1:])))

        Def = self.Deform(ypred)
        mnDef = torch.mean(Def, dim=(1,2,3,4)
                           )[:, np.newaxis,np.newaxis,np.newaxis,np.newaxis].repeat(
            1,1,vsz[2],vsz[3],vsz[4]
        )

        # cannot do in-place operations
        # Def -= mnDef # creates issue with tracking gradients
        Def = Def.clone() - mnDef
        nDef = torch.linalg.vector_norm(Def, dim=(1,2,3,4)
                                        )[:, np.newaxis,np.newaxis,np.newaxis,np.newaxis].repeat(
            1,1,vsz[2],vsz[3],vsz[4]
        )

        A = self.A[np.newaxis,np.newaxis,:,:,:].repeat(vsz[0],1,1,1,1)

        # cross-correlation loss
        ccloss = v2*torch.mean(1 - torch.sum(A * Def/(nDef + 1e-5), dim=(1,2,3,4)))


        varAir = torch.var(Def[:,:,self.Aair[0],self.Aair[1],self.Aair[2]],
                         dim=(1,2))
        varSoft = torch.var(Def[:,:,self.Asoft[0],self.Asoft[1],self.Asoft[2]],
                         dim=(1,2))
        varBone = torch.var(Def[:,:,self.Abone[0],self.Abone[1],self.Abone[2]],
                         dim=(1,2))
        #level set loss
        lsloss = v3*(torch.mean(varAir) + torch.mean(varSoft) + torch.mean(varBone))
        if self.training:
            print(f'Losses gradloss: {gradloss.item()} ccloss: {ccloss.item()} lsloss: {lsloss.item()}')
            return ccloss + gradloss + lsloss + gradloss2 + gradloss3
        else:
            print(f'Losses gradloss: {gradloss.item()} ccloss: {ccloss.item()} lsloss: {lsloss.item()}')
            return ccloss + gradloss + lsloss

    def Deform(self,ypred, D=None):
        if D is None:
            D = self.D

        vsz = np.array(ypred.size())

        X = ((ypred[:,0,:,:,:] + self.X[np.newaxis,:,:,:].repeat(ypred.size(0),1,1,1))
            *2/(vsz[2]-1)-1)
        Y = ((ypred[:,1,:,:,:] + self.Y[np.newaxis,:,:,:].repeat(ypred.size(0),1,1,1))
             * 2 / (vsz[3] - 1) - 1)
        Z = ((ypred[:,2,:,:,:] + self.Z[np.newaxis,:,:,:].repeat(ypred.size(0),1,1,1))
             * 2 / (vsz[4] - 1) - 1)
        grid = torch.cat((Z[:,:,:,:,np.newaxis], Y[:,:,:,:,np.newaxis], X[:,:,:,:,np.newaxis]),
                         dim=4)
        # torch's autograd compatible version of grid interpolation
        Def = nn.functional.grid_sample(D, grid, align_corners=True,
                                  mode='bilinear', padding_mode='zeros' )
        return Def



from torch.utils.data import Dataset, DataLoader
# from PIL import Image # or any image library
import torchvision.transforms as transforms

class RegistrationDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith('.nrrd')]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        image, header = nrrd.read(image_path)

        image = torch.tensor(image[np.newaxis,:,:,:], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image


earDataDir = 'C:\\Users\\noblejh\\Box Sync\\2025_ECE8396_MIS\\ECE_8396_Code\\EarData\\'
atlasDir = earDataDir + 'atlas\\'
trainDir = earDataDir + 'training\\'
validDir = earDataDir + 'validation\\'

atlas = np.array(nrrd.read(atlasDir + 'image.nrrd')[0], dtype=np.float32)

train_D = RegistrationDataset(trainDir) # transforms input can be used for augmentation functions
valid_D = RegistrationDataset(validDir)

train_dataloader = DataLoader(train_D, batch_size=9, shuffle=True)
valid_dataloader = DataLoader(valid_D, batch_size=6, shuffle=False)

dev = torch.device('cuda')

Dv = next(iter(valid_dataloader)).to(dev)

fig, ax = plt.subplots()
model = RegNet(dev, atlas)
model.to(dev)
opt = torch.optim.SGD(model.parameters(), lr=1e-5, momentum=0.9)

model.fit(500, model.myLoss, opt, train_dataloader, valid_D=Dv, savebest='VM_test.pth')

model = torch.load('VoxelMorph.pth')
model.to(dev)
model.eval()
model.Plot(0, 0, 0, 0, 0, Dv, rind=0)

ydef = model(Dv)
y = model.Deform(ydef, Dv)

img = y[0,0,:,:,:].detach().cpu().numpy()

vv = volumeViewer()
vv.setImage(img, [1,1,1])
vv.display()
