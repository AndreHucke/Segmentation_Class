import matplotlib.pyplot as plt
import nrrd
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plot
import numpy as np
from Project1 import *
from Project2 import *
from myVTKWin import *
import scipy.ndimage as ndi
from skimage import feature

####Need lots of updates to pdf code demo

# Load CT image to demo connected component analysis
img,header = nrrd.read('./EECE_395/0522c0001/img.nrrd')
voxsz = [header['space directions'][0][0],header['space directions'][1][1],
         header['space directions'][2][2]]  # mm per voxel

# Crop and downsample so that livewire in native python is faster
fp = np.array([199,129,51])
sp = np.array([320, 240, 77])
crp = img[fp[0]:sp[0]:2, fp[1]:sp[1]:2, fp[2]:sp[2]:2]

voxsz *= 2

# volumeViewer extended in project 1
viewer = volumeViewer()
viewer.setImage(img, voxsz, autocontrast=False, showHistogram=False)

# pick a slice to segment
slc = 3
img2d = crp[:,:,slc]

# compute edge detection features for our edge weight functional
r, c = np.shape(img2d)
sobel = np.zeros((r, c, 4))
sobel[:,:,0] = ndi.convolve(img2d,  np.array([[0,1,2], [-1,0,1], [-2,-1,0]])) #x-y-/x+y+
sobel[:,:,1] = ndi.convolve(img2d,  np.array([[2,1,0], [1,0,-1], [0,-1,-2]])) #x-y+/x+y-
sobel[:,:,2] = ndi.convolve(img2d,  np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])) #y+/-
sobel[:,:,3] = ndi.convolve(img2d,  np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]]).T) #x+/-

# invert magnitude of Sobel for low cost on edges
sobel = np.abs(sobel)
for i in range(4):
    sobel[:,:,i] = 1 - sobel[:,:,i]/np.amax(sobel[:,:,i])

# add canny as a hard constraint
canny = 1 - feature.canny(img2d, sigma=1)

# create a list of edges for each node in the graph
edges = [[] for i in range(r*c)]
alpha = 0.5 # canny vs sobel
beta = 0.1 # smoothness/straightness

# custom class to contain edge and heap sort them
class lwedge:
    def __init__(self, child=-1, parent=-1, cost=0):
        self.parent = parent
        self.child = child
        self.cost = cost

    def __lt__(self, rhs):
        return self.cost < rhs.cost

# build edge lists, accounting for image boundaries 
sq2 = np.sqrt(2)
for x in range(r):
    for y in range(c):
        nd = x + y*r
        if x > 0:
            if y > 0:
                # diagonal x-y-  neib
                neib = x - 1 + (y-1) *r
                edges[nd].append(lwedge(child = neib, parent=nd,
                                        cost = sq2*(beta + alpha * canny[x,y] + (1-alpha) * sobel[x, y, 0]
                                        )))
            if y < c-1:
                # diagonal x-y+ neib
                neib = x - 1 + (y + 1) * r
                edges[nd].append(lwedge(child=neib,parent=nd,
                                        cost=sq2*(beta + alpha * canny[x,y] + (1 - alpha) * sobel[x,y,1]
                                        )))
            # x- neib
            neib = (x-1 + y*r)
            edges[nd].append(lwedge(child=neib,parent=nd,
                                    cost=beta + alpha * canny[x,y] + (1 - alpha) * sobel[x,y,3]
                                    ))
        if x < r-1:
            if y > 0:
                # digaonal x+y- neib
                neib = x + 1 + (y - 1) * r
                edges[nd].append(lwedge(child=neib,parent=nd,
                                        cost=sq2*(beta + alpha * canny[x,y] + (1 - alpha) * sobel[x,y,1]
                                        )))
            if y < c-1:
                # diagonal x+ y+ neib
                neib = x + 1 + (y + 1) * r
                edges[nd].append(lwedge(child=neib,parent=nd,
                                        cost=sq2*(beta + alpha * canny[x,y] + (1 - alpha) * sobel[x,y,0]
                                        )))
            # x + neib
            neib = (x + 1 + y * r)
            edges[nd].append(lwedge(child=neib,parent=nd,
                                    cost=beta + alpha * canny[x,y] + (1 - alpha) * sobel[x,y,3]
                                    ))
        if y>0:
            # y- neib
            neib = (x + (y - 1) * r)
            edges[nd].append(lwedge(child=neib,parent=nd,
                                    cost=beta + alpha * canny[x,y] + (1 - alpha) * sobel[x,y,2]
                                    ))
        if y<c-1:
            # y+ neib
            neib = (x + (y + 1) * r)
            edges[nd].append(lwedge(child=neib,parent=nd,
                                    cost=beta + alpha * canny[x,y] + (1 - alpha) * sobel[x,y,2]
                                    ))

# Sqrt(2) term needed on diagonals to reward straight line paths more than zig-zag ones
###  |    /
###  |    \
###  |    /
fig, ax = plt.subplots()
ax.imshow(img2d.T, 'gray')
plt.ion()
plt.show()

#try the edges with the liveWire solution you will develop in Project 3
from Project3 import *
lw = liveWire(img2d, edges=edges)

print(lw.cntrcur)

plt.ioff()
plt.show()

import heapq as hq
h = []
nd = lwedge(child=0, parent=-1, cost=1)
hq.heappush(h, nd)

hq.heappush(h, lwedge(child=1, parent=0, cost=20))
hq.heappush(h, lwedge(child=2, parent=0, cost=2))
hq.heappush(h, lwedge(child=3, parent=0, cost=3))
hq.heappush(h, lwedge(child=4, parent=2, cost=9))
hq.heappush(h, lwedge(child=5, parent=1, cost=23))
hq.heappush(h, lwedge(child=6, parent=1, cost=21))
hq.heappush(h, lwedge(child=7, parent=2, cost=5))
hq.heappush(h, lwedge(child=8, parent=6, cost=22))

for i in range(len(h)):
    print(h[i].cost)

edge = hq.heappop(h)
print(f'Popped edge with child={edge.child} and cost {edge.cost}')
for i in range(len(h)):
    print(h[i].cost)

print(f'\n\n')
hq.heappush(h, lwedge(child=9, parent=3, cost=4))
for i in range(len(h)):
    print(h[i].cost)

r = 50
c = 60
alpha = .25
beta = 0.1
X,Y = np.meshgrid(range(c), range(r))
img2d = np.zeros((r,c)) + 10*((X-c/2)*(X-c/2) + (Y-r/2)*(Y-r/2) < 400)
# img2d += np.random.default_rng(0).normal(size=(r,c))*8
seed = r//2 + 2*r
endnode = r//2 + 57*r

fig, ax = plt.subplots()
ax.imshow(img2d.T, 'gray')
plt.ioff()

edges = [[] for i in range(r*c)]
sobel = np.zeros((r, c, 4))
sobel[:,:,0] = ndi.convolve(img2d,  np.array([[0,1,2], [-1,0,1], [-2,-1,0]])) #x-y-/x+y+
sobel[:,:,1] = ndi.convolve(img2d,  np.array([[2,1,0], [1,0,-1], [0,-1,-2]])) #x-y+/x+y-
sobel[:,:,2] = ndi.convolve(img2d,  np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])) #y+/-
sobel[:,:,3] = ndi.convolve(img2d,  np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]]).T) #x+/-
sobel = np.abs(sobel)
for i in range(4):
    sobel[:,:,i] = 1 - sobel[:,:,i]/np.amax(sobel[:,:,i])

# add canny as a hard constraint
canny = 1 - feature.canny(img2d, sigma=1)

# sq2=1
for x in range(r):
    for y in range(c):
        nd = x + y*r
        if x > 0:
            if y > 0:
                # diagonal x-y-  neib
                neib = x - 1 + (y-1) *r
                edges[nd].append(lwedge(child = neib, parent=nd,
                                        cost = sq2*(beta + alpha * canny[x,y] + (1-alpha) * sobel[x, y, 0]
                                        )))
            if y < c-1:
                # diagonal x-y+ neib
                neib = x - 1 + (y + 1) * r
                edges[nd].append(lwedge(child=neib,parent=nd,
                                        cost=sq2*(beta + alpha * canny[x,y] + (1 - alpha) * sobel[x,y,1]
                                        )))
            # x- neib
            neib = (x-1 + y*r)
            edges[nd].append(lwedge(child=neib,parent=nd,
                                    cost=beta + alpha * canny[x,y] + (1 - alpha) * sobel[x,y,3]
                                    ))
        if x < r-1:
            if y > 0:
                # digaonal x+y- neib
                neib = x + 1 + (y - 1) * r
                edges[nd].append(lwedge(child=neib,parent=nd,
                                        cost=sq2*(beta + alpha * canny[x,y] + (1 - alpha) * sobel[x,y,1]
                                        )))
            if y < c-1:
                # diagonal x+ y+ neib
                neib = x + 1 + (y + 1) * r
                edges[nd].append(lwedge(child=neib,parent=nd,
                                        cost=sq2*(beta + alpha * canny[x,y] + (1 - alpha) * sobel[x,y,0]
                                        )))
            # x + neib
            neib = (x + 1 + y * r)
            edges[nd].append(lwedge(child=neib,parent=nd,
                                    cost=beta + alpha * canny[x,y] + (1 - alpha) * sobel[x,y,3]
                                    ))
        if y>0:
            # y- neib
            neib = (x + (y - 1) * r)
            edges[nd].append(lwedge(child=neib,parent=nd,
                                    cost=beta + alpha * canny[x,y] + (1 - alpha) * sobel[x,y,2]
                                    ))
        if y<c-1:
            # y+ neib
            neib = (x + (y + 1) * r)
            edges[nd].append(lwedge(child=neib,parent=nd,
                                    cost=beta + alpha * canny[x,y] + (1 - alpha) * sobel[x,y,2]
                                    ))


gs = graphSearchLW()
pathnodes, pathcost = gs.run(edges, seed, endnode)
print(pathcost)
path = np.zeros((len(pathnodes), 2))
for i in range(len(pathnodes)):
    path[i,0] = pathnodes[i] % r
    path[i,1] = pathnodes[i] // r

plt.plot(path[:,0], path[:,1], 'r')
plt.ioff()