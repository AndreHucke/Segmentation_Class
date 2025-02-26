# % Class to create interactive 3D image/mask viewer
# % EECE 8396: Medical Image Segmentation
# % Spring 2024
# % Author: Prof. Jack Noble; jack.noble@vanderbilt.edu
#


import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
from skimage import measure
from myVTKWin import *

class imagevolume:
    def __init__(self,data=0,voxsz=[1,1,1]):
        self.data = data
        self.voxsz = voxsz

class contourclass:
    def __init__(self, data):
        self.data = data

class object:
    def __init__(self, type, data, color=[1.0,0.0,0.0], opacity = 1.0):
        self.type = type
        self.data = data
        self.color = color
        self.opacity = opacity
        self.ms = None

class mask:
    def __init__(self, data, voxsz=[1.0,1.0,1.0], color=[1.0,0.0,0.0], label=''):
        self.data = np.asarray(data)
        self.voxsz = np.asarray(voxsz)
        self.color = np.asarray(color)
        self.label = label
        self.cntrs = np.zeros([3,max(np.shape(data))],dtype=contourclass)

    def updateContours(self, win, opacity=1.0):
        dim = np.shape(self.data)
        X,Y = np.meshgrid(np.linspace(0,dim[0]-1,dim[0]), np.linspace(0,dim[1]-1,dim[1]))
        for i in range(np.shape(self.data)[2]):
            if np.min(self.data[:,:,i])<0.5 and np.max(self.data[:,:,i]) > 0.5:
                cntr = plt.contour(X,Y,np.transpose(self.data[:,:,i]),levels=[0.5])
                self.cntrs[0][i] = contourclass(cntr.allsegs[0])
        X, Y = np.meshgrid(np.linspace(0, dim[0] - 1, dim[0]), np.linspace(0, dim[2] - 1, dim[2]))
        for i in range(np.shape(self.data)[1]):
            if np.min(self.data[:, i, :]) < 0.5 and np.max(self.data[:, i, :]) > 0.5:
                cntr = plt.contour(X, Y, np.transpose(np.squeeze(self.data[:, i, :])), levels=[0.5])
                self.cntrs[1][i] = contourclass(cntr.allsegs[0])
        X, Y = np.meshgrid(np.linspace(0, dim[1] - 1, dim[1]), np.linspace(0, dim[2] - 1, dim[2]))
        for i in range(np.shape(self.data)[0]):
            if np.min(self.data[i, :, :]) < 0.5 and np.max(self.data[i, :, :]) > 0.5:
                cntr = plt.contour(X, Y, np.transpose(np.squeeze(self.data[i, :, :])), levels=[0.5])
                self.cntrs[2][i] = contourclass(cntr.allsegs[0])

        verts,faces,_,_ = measure.marching_cubes(self.data,0.5, spacing=self.voxsz)
        win.addSurf(verts,faces,color=self.color, opacity=opacity)


class volumeViewer(myVtkWin):
    def __init__(self, title='Volume Viewer (Press Esc to quit)'):
        super().__init__(title="3D display")
        self.img = None
        self.objs = []
        self.slc = [0,0,0]
        self.contrast = 1
        self.level = 0
        plt.ion()
        self.fig = plt.figure(figsize=(10, 8))
        self.fig.suptitle(title, fontsize=16)
        
        # Create all subplots at initialization
        self.ax = np.zeros([2,2], dtype=plt.Axes)
        self.ax[0,0] = self.fig.add_subplot(2,2,1)  # Sagittal view
        self.ax[0,1] = self.fig.add_subplot(2,2,2)  # Coronal view
        self.ax[1,0] = self.fig.add_subplot(2,2,3)  # Axial view
        self.ax[1,1] = self.fig.add_subplot(2,2,4)  # Histogram
        
        plt.axes(self.ax[0,0])

        self.quit = False
        binding_id2 = plt.connect('button_press_event',self.onMouseClick)
        binding_id3 = plt.connect('key_press_event',self.onKeyPress)

    def onMouseClick(self,event):
        if event.dblclick:
            if event.button is MouseButton.LEFT:
                direction = -1
                for i in range(0,3):
                    if event.inaxes == self.ax[i//2, i%2]:
                        direction = i
                if direction == 2:
                    pnt = [event.xdata, event.ydata, self.slc[2]]
                elif direction ==1:
                    pnt = [event.xdata, self.slc[1], event.ydata]
                elif direction == 0:
                    pnt = [self.slc[0], event.xdata, event.ydata]
                else:
                    return
                self.centerOnPoint(pnt)
            if event.button is MouseButton.RIGHT:
                if event.inaxes == self.ax[1,0]:
                    self.ax[1,0].set_xlim(left=0, right=np.shape(self.img)[0] - 1)
                    self.ax[1,0].set_ylim(bottom=np.shape(self.img)[1] - 1,top=0)
                elif event.inaxes == self.ax[0,1]:
                    self.ax[0,1].set_xlim(left=0, right=np.shape(self.img)[0] - 1)
                    self.ax[0,1].set_ylim(top=np.shape(self.img)[2] - 1,bottom=0)
                elif event.inaxes == self.ax[0,0]:
                    self.ax[0,0].set_xlim(left=0, right=np.shape(self.img)[1] - 1)
                    self.ax[0,0].set_ylim(top=np.shape(self.img)[2] - 1,bottom=0)


    def centerOnPoint(self,pnt):
        pnt = np.copy(pnt)
        for i in range(3):
            if pnt[i]<0:
                pnt[i]=0
            elif pnt[i]>np.shape(self.img)[i]-1:
                pnt[i] = np.shape(self.img)[i]-1
            self.slc[i] = round(pnt[i])
            self.update(i)
        for i in range(0,3):
            xlim = self.ax[i//2,i%2].get_xlim()
            ylim = self.ax[i//2,i%2].get_ylim()
            xrng = xlim[1] - xlim[0]
            yrng = ylim[1] - ylim[0]
            if i==0:
                x = pnt[1]
                y = pnt[2]
            elif i==1:
                x = pnt[0]
                y = pnt[2]
            else:
                x = pnt[0]
                y = pnt[1]
            self.ax[i//2,i%2].set_xlim(x-xrng/2,x+xrng/2)
            self.ax[i//2,i%2].set_ylim(y-yrng/2,y+yrng/2)
            plt.axes(self.ax[i//2,i%2])
            plt.plot([x, x], [y + 0.02 * yrng, y - 0.02 * yrng], 'r')
            plt.plot([x + 0.02 * xrng, x - 0.02 * xrng], [y, y], 'r')

    def keypress_callback(self,obj,ev): # overloads myVTKWin key press callback function
        key = super().keypress_callback(obj,ev)
        if (key == 'u' or key == 'U'):
            pnt = self.lastpickpos / self.voxsz
            self.centerOnPoint(pnt)

    def onKeyPress(self,event):# for matplotlib window
        if event.key == 'escape' or event.key=='q' or event.key=='Q':
            self.quit = True
        
        # Determine which view is in focus
        direction = -1
        if event.inaxes == self.ax[0,0]:
            direction = 0  # sagittal
        elif event.inaxes == self.ax[0,1]:
            direction = 1  # coronal
        elif event.inaxes == self.ax[1,0]:
            direction = 2  # axial
            
        # Handle slice navigation
        if direction >= 0:
            if event.key == 'a' or event.key == 'up':
                newslc = min(self.slc[direction] + 1, np.shape(self.img)[direction] - 1)
                self.update(direction=direction, slc=newslc)
            elif event.key == 'z' or event.key == 'down':
                newslc = max(self.slc[direction] - 1, 0)
                self.update(direction=direction, slc=newslc)
                
        # Handle contrast and level adjustments
        if event.key == 'd':
            self.update(level=self.level + 0.1*self.contrast)
        if event.key == 'x':
            self.update(level=self.level - 0.1*self.contrast)
        if event.key == 'c':
            self.update(contrast=self.contrast*1.1)
        if event.key == 'v':
            self.update(contrast=self.contrast/1.1)

    def setImage(self,img,voxsz,contrast=1000,level=0,interpolation='bilinear',autocontrast=True,showHistogram=False):
        self.img = np.asarray(img)
        self.voxsz = np.asarray(voxsz)
        self.slc = np.array(np.shape(img.data), dtype=int) //2
        self.contrast=contrast
        self.level = level
        self.interpolation = interpolation
        if autocontrast:
            self.autoContrast()

        if showHistogram:
            plt.axes(self.ax[1,1])
            plt.cla()
            plt.hist(self.img.ravel(), bins=64)
            plt.yscale('log')
            plt.title('Image Histogram')
            plt.xlabel('Intensity')
            plt.ylabel('Log(Count)')
            
        self.update()

    def autoContrast(self):
        mn = np.amin(self.img)
        mx = np.amax(self.img)
        bns = np.linspace(mn,mx,256)
        h,_ = np.histogram(self.img.ravel(), bins=bns)
        h = h.astype(np.float32)/np.sum(h)
        mini = 0
        tot = h[0]
        while tot<0.1:
            mini+=1
            tot += h[mini]

        maxi = mini
        while tot<0.99:
            maxi +=1
            tot += h[maxi]

        self.contrast = (maxi - mini) * (mx - mn) / 256.0
        self.level = 0.5 * (maxi + mini) * (mx - mn) / 256.0 + mn

    def addMask(self,msk,color = [1.0,0.0,0.0], opacity=1.0, label=''):
        mskobj = mask(msk, self.voxsz, color,label)
        mskobj.updateContours(self, opacity=opacity)
        obj = object(1,mskobj,color=color,opacity=opacity)
        self.objs.append(obj)

    def update(self,direction = -1,slc = -1,level = float("nan"),contrast = float("nan"),resize = -1):
        if (not np.isnan(level)):
            self.level = level
        if (not np.isnan(contrast)):
            self.contrast = contrast
        if (resize != -1):
            self.resize = resize
        if direction == -1:
            for i in range(4):
                self.update(direction = i)
        else:
            if slc >= 0:
                self.slc[direction] = slc
            if direction <3:
                plt.figure(self.fig)
                plt.axes(self.ax[direction // 2, direction % 2])
                xlim = self.ax[direction//2, direction%2].get_xlim()
                ylim = self.ax[direction//2, direction%2].get_ylim()
                plt.cla()
                if direction == 2:
                    self.ax[direction//2, direction%2].imshow(np.transpose(self.img[:, :, self.slc[direction]]),
                                                                                   'gray', interpolation=self.interpolation,
                                                                                   vmin=self.level - self.contrast/2,
                                                                            vmax=self.level + self.contrast/2)
                    plt.xlabel('x')
                    plt.ylabel('y')
                    self.ax[direction//2, direction%2].set_aspect(self.voxsz[1]/self.voxsz[0])
                    if xlim[1] != 1:
                        self.ax[direction//2, direction%2].set_xlim(left=xlim[0], right=xlim[1])
                        self.ax[direction//2, direction%2].set_ylim(bottom=ylim[0], top=ylim[1])
                    else:
                        self.ax[direction//2, direction%2].set_xlim(left=0,
                                                                                         right=np.shape(self.img)[0] - 1)
                        self.ax[direction//2, direction%2].set_ylim(bottom=np.shape(self.img)[1] - 1,
                                                                                         top=0)
                    vstr = 'Axial view'
                    z = 'z'
                    for i in range(len(self.objs)):
                        if self.objs[i].type==1 and self.objs[i].data.cntrs[0][self.slc[2]]:
                            for j in range(len(self.objs[i].data.cntrs[0][self.slc[2]].data)):
                                plt.plot(self.objs[i].data.cntrs[0][self.slc[2]].data[j][:, 0],
                                         self.objs[i].data.cntrs[0][self.slc[2]].data[j][:, 1], color=self.objs[i].color)
                elif direction == 1:
                    self.ax[direction//2, direction%2].imshow(np.transpose(np.squeeze(self.img[:, self.slc[direction], :])),
                                                                                   'gray', interpolation=self.interpolation,
                                                                                   vmin=self.level - self.contrast/2, vmax=self.level + self.contrast/2)
                    plt.xlabel('x')
                    plt.ylabel('z')
                    self.ax[direction//2, direction%2].set_aspect(self.voxsz[2] / self.voxsz[0])
                    if xlim[1] != 1:
                        self.ax[direction//2, direction%2].set_xlim(left=xlim[0], right=xlim[1])
                        self.ax[direction//2, direction%2].set_ylim(bottom=ylim[0], top=ylim[1])
                    else:
                        self.ax[direction//2, direction%2].set_xlim(left=0,
                                                                                         right=np.shape(self.img)[0] - 1)
                        self.ax[direction//2, direction%2].set_ylim(bottom=0,
                                                                                         top=np.shape(self.img)[2] - 1)
                    vstr = 'Coronal view'
                    z = 'y'
                    for i in range(len(self.objs)):
                        if self.objs[i].type==1 and self.objs[i].data.cntrs[1][self.slc[1]]:
                            for j in range(len(self.objs[i].data.cntrs[1][self.slc[1]].data)):
                                plt.plot(self.objs[i].data.cntrs[1][self.slc[1]].data[j][:, 0],
                                         self.objs[i].data.cntrs[1][self.slc[1]].data[j][:, 1], color=self.objs[i].color)
                elif direction == 0:
                    self.ax[direction//2, direction%2].imshow(np.transpose(np.squeeze(self.img[self.slc[direction], :, :])),
                                                                                   'gray', interpolation=self.interpolation,
                                                                                   vmin=self.level - self.contrast/2, vmax=self.level + self.contrast/2)
                    plt.xlabel('y')
                    plt.ylabel('z')
                    self.ax[direction//2, direction%2].set_aspect(self.voxsz[2] / self.voxsz[1])
                    if xlim[1] != 1:
                        self.ax[direction//2, direction%2].set_xlim(left=xlim[0], right=xlim[1])
                        self.ax[direction//2, direction%2].set_ylim(bottom=ylim[0], top=ylim[1])
                    else:
                        self.ax[direction//2, direction%2].set_xlim(left=0, right=np.shape(self.img)[1]-1)
                        self.ax[direction//2, direction%2].set_ylim(bottom=0, top=np.shape(self.img)[2]-1)
                    vstr = f'Sagittal view Contrast = {self.contrast:.1f} Level = {self.level:.1f}'
                    z = 'x'
                    for i in range(0,np.size(self.objs)):
                        if self.objs[i].type==1 and self.objs[i].data.cntrs[2][self.slc[0]]:
                            for j in range(len(self.objs[i].data.cntrs[2][self.slc[0]].data)):
                                plt.plot(self.objs[i].data.cntrs[2][self.slc[0]].data[j][:, 0],
                                         self.objs[i].data.cntrs[2][self.slc[0]].data[j][:, 1], color=self.objs[i].color)


                plt.title(f'{vstr}: Slice {z} = {self.slc[direction]}')
            elif direction == 3:
                self.render()

    def repaint(self):
        self.fig.canvas.draw_idle()
        self.fig.canvas.start_event_loop(0.3)

    def display(self,blocking=True):
        self.update()
        if blocking:
            while (self.quit == False):
                self.repaint()
        if self.quit:
            del self

    def __del__(self):
        plt.close(self.fig)
        super().__del__()