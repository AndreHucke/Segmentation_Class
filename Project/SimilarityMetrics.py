import nrrd
from surface import * # hidden (imports complete surfacep5 class)
from levelSet import * # hidden (imports complete confusionMatrixp5 class)
import scipy as sp
import matplotlib.pyplot as plt
from volumeViewer import *
import numpy as np


bsdir = '/Users/bohanjiang/Desktop/240221_Demo/EECE_395/'
pts = {0:'0522c0001'}# can add entries to this dictionary to create a list of patients

gt, hdr, = nrrd.read(bsdir + pts[0] + '/structures/mandible.nrrd')
voxsz = [hdr['space directions'][0][0], hdr['space directions'][1][1], hdr['space directions'][2][2]]

t1, _ = nrrd.read(bsdir + pts[0] + '/structures/target1.nrrd')
t2, _ = nrrd.read(bsdir + pts[0] + '/structures/target2.nrrd')
t3, _ = nrrd.read(bsdir + pts[0] + '/structures/target3.nrrd')

# majority vote
mv = t1 + t2 + t3 > 1.5

# Create and display surfaces for each
gts = surfacep5()
gts.createSurfaceFromVolume(gt, voxsz, 0.5)
t1s = surfacep5()
t1s.createSurfaceFromVolume(t1, voxsz, 0.5)
t2s = surfacep5()
t2s.createSurfaceFromVolume(t2, voxsz, 0.5)
t3s = surfacep5()
t3s.createSurfaceFromVolume(t3, voxsz, 0.5)


win = myVtkWin()
win.addSurf(t1s.verts, t1s.faces, opacity=0.5)
win.addSurf(t2s.verts, t2s.faces, color = [1,1,0], opacity=0.5)
win.addSurf(t3s.verts, t3s.faces, color = [1,0,1], opacity=0.5)
win.start()

# Also show contour on 2D slice
img, _ = nrrd.read(bsdir+pts[0]+'/img.nrrd')
d = volumeViewerp1()
d.setImage(img, voxsz)
d.update(direction=2, slc=61)
d.display(blocking=False)
plt.axes(d.ax[1,0])
X,Y = np.meshgrid(range(np.shape(img)[0]), range(np.shape(img)[1]))
plt.contour(X, Y, t1[:, :, d.slc[2]].T, [0.5], colors = ['red'])
plt.ioff()
plt.show()


# base confusion matrix class (given) calculates TP, FP, TN, FN
# the extension for project 5 adds sensitivity/specificity/dice functions
Cg1 = confusionMatrixp5(gt, t1)
Cg1.print()
#				T2 Fore		|	T2 Back
#__________________________________________
#T1 Fore	|          10502	|	311
#T1 Back	|            200	|	28038395

print(f'Sensitivity (T1=GT): {Cg1.sensitivity()}')
# Sensitivity (T1=GT): 0.9712383242393415
print(f'Specificity (T1=GT): {Cg1.specificity()}')
# Specificity (T1=GT): 0.9999928669749679

print(f'Dice: {Cg1.dice()}')
# Dice: 0.9762491285149896

Cmv = confusionMatrixp5(gt,mv)
print(f'Dice for MV: {Cmv.dice()}')
# Dice for MV: 0.9827824325568548


# # surface distance examples
# computation can take a minute, function uses tqdm package for progress bar display
dg1, MASDg1, HDg1, mp1, mp2 = gts.surfDistances(t1s)

print(MASDg1)
print(HDg1)
# 0.11669648095200053
# 1.1200000047683716

d1g, MASD1g, HD1g, _, _ = t1s.surfDistances(gts)
#Overall stats factor in distances in both directions:
print(f'MSASD gt vs t1: {(MASDg1 + MASD1g)/2}')
print(f'HD gt vs t1: {np.max([HDg1,HD1g])}')
# MSASD gt vs t1: 0.11684233170161465
# HD gt vs t1: 1.6011246056039137

# display results
# print(np.linalg.norm(mp1-mp2))
win2 = myVtkWin()
win2.addSurf(t1s.verts, t1s.faces, opacity=0.5)
win2.addSurf(gts.verts, gts.faces, color=[0,1,0], opacity=0.5)
# win2.addLines(np.concatenate((mp1[np.newaxis,:], mp2[np.newaxis,:]), axis=0),
#               np.array([[0,1]]), lineWidth=5, color=[1,0,1])
win2.start()

pdg1, MAPDg1, HPDg1 = gts.pointsetDistance(t1s)
print(f'MAPSD gt vs t1: {MAPDg1}')
print(f'HPD gt vs t1: {HPDg1}')
# MAPSD gt vs t1: 0.13718189749432708

# demonstrate closest distance computation from point p to triangle q1q2q3
# triangle q1q2q3
q1 = np.array([0, -.5, 0])
q2 = np.array([0, 0.5, 1])
q3 = np.array([0, 0.5, -1])
p = np.array([5, .25, .1])

v1 = q2 - q1
v2 = q3 - q1
V = np.concatenate((v1[:,np.newaxis], v2[:,np.newaxis]), axis=1)
pinv = np.linalg.pinv(V)

# if a and b between 0,1, closest point is on triangle face
ab = pinv @ ((p-q1)[:,np.newaxis])
print(ab)
#[[0.425]
# [0.325]]

# coordinates of closest point on face
print(V @ ab + q1[:,np.newaxis])
# [[0.  ]
#  [0.25]
#  [0.1 ]]

# Distance from p to triangle
print(np.sqrt(np.sum((V @ ab + q1[:,np.newaxis] - p[:,np.newaxis]) ** 2)))
print(np.linalg.norm(V @ ab + q1[:,np.newaxis] - p[:,np.newaxis]))
# 5.0
# 5.0


# try another point with same triangle
p = np.array([0, 5, 0])
ab = pinv @ ((p-q1)[:,np.newaxis])
print(ab)
# [[2.75]
# [2.75]]
# not in triangle bc a, b > 1


v3 = q3 - q2
c = np.sum(v1 * (p - q1))/np.sum(v1*v1)
d = np.sum(v2 * (p - q1))/np.sum(v2*v2)
e = np.sum(v3 * (p - q2))/np.sum(v3*v3)
print(c)
print(d)
print(e)
# 2.75
# 2.75
# 0.5

#bracket to triangle edges
if c<0: c=0
if c>1: c=1
if d<0: d=0
if d>1: d=1
if e<0: e=0
if e>1: e=1

d1 = np.sqrt(np.sum((q1 + c*v1 - p) ** 2))
d2 = np.sqrt(np.sum((q1 + d*v2 - p) ** 2))
d3 = np.sqrt(np.sum((q2 + e*v3 - p) ** 2))
print(d1)
print(d2)
print(d3)
# 4.6097722286464435
# 4.6097722286464435
# 4.5
# point on third leg of triangle is min

#one last example with new point p
p = np.array([0,1,2])
ab = pinv @ ((p-q1)[:,np.newaxis])
print(ab)
# [[ 1.75]
#  [-0.25]]

# again, closest point not on face

v3 = q3 - q2
c = np.sum(v1 * (p - q1))/np.sum(v1*v1)
d = np.sum(v2 * (p - q1))/np.sum(v2*v2)
e = np.sum(v3 * (p - q2))/np.sum(v3*v3)
print(c)
print(d)
print(e)
# 1.75
# -0.25
# -0.5

#bracket to triangle edges
if c<0: c=0
if c>1: c=1
if d<0: d=0
if d>1: d=1
if e<0: e=0
if e>1: e=1

d1 = np.sqrt(np.sum((q1 + c*v1 - p) ** 2))
d2 = np.sqrt(np.sum((q1 + d*v2 - p) ** 2))
d3 = np.sqrt(np.sum((q2 + e*v3 - p) ** 2))

print(d1)
print(d2)
print(d3)
# 1.118033988749895
# 2.5
# 1.118033988749895
#closest point on vertex of triangle (q2)


# try some stats tests
rng = np.random.default_rng(0)
# try size=10,50
a = rng.normal(size=10)
b = rng.normal(size=10)
print(sp.stats.ttest_rel(a,b))
print(sp.stats.wilcoxon(a,b))

a = a + 0.5
print(sp.stats.ttest_rel(a,b))
print(sp.stats.wilcoxon(a,b))

a = a + 0.5
print(sp.stats.ttest_rel(a,b))
print(sp.stats.wilcoxon(a,b))
# need to shift mean by 1 to hit significance with N=10

# using boxplot to visualize difference in distributions
# from myBoxplot import *
# plt.figure()
fig, ax = plt.subplots()
plt.boxplot(np.concatenate((a[:, np.newaxis], b[:, np.newaxis]), axis=1)) #myboxplot...,20)
plt.ioff()
plt.show()
