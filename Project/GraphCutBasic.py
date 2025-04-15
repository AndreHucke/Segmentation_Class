# % Graph Cut class with missing grow, edgeFunc, and tracePath functions
# % ECE 3896/8395: Engineering for Surgery
# % Fall 2024
# % Author: Prof. Jack Noble; jack.noble@vanderbilt.edu

import numpy as np
# class for nlink
class nlink:
    def __init__(self, neib=-1, cap=0):
        self.neib=neib
        self.cap=cap

# graphcut class
class graphCutBasic:
    def __init__(self):
        self.nlinks = None
        self.tlinks = None
        self.tree = None
        self.active = None
        self.parent = None
        self.N = 0
        self.totcap = 0

    # default edge capacity function
    def initEdges(self, img=None, PrS=None, sigma=1, alpha=.1, lmbda=0.5, nlinks=None, tlinks=None):
        if not nlinks is None:
            self.nlinks = nlinks
            self.tlinks = tlinks
            return

        # init the tlinks and nlinks
        r,c,d = np.shape(img)
        rc = r * c
        self.nlinks = [[] for i in range(self.N)]
        self.tlinks = np.zeros((self.N, 2))
        imgr = np.ravel(img, order='F')
        for i in range(self.N):
            z = i//(rc)
            y = (i - z*rc)//r
            x = i%r
            self.tlinks[i, 0] = -lmbda*np.log(1-PrS[i])
            self.tlinks[i, 1] = -lmbda*np.log(PrS[i])
            if x>0:
                self.nlinks[i].append(nlink(i-1,
                    (1-lmbda)*np.exp(-(imgr[i] - imgr[i-1])*(imgr[i] - imgr[i-1])/(2*sigma*sigma)) + alpha))
            if x<r-1:
                self.nlinks[i].append(nlink(i+1,
                    (1-lmbda)*np.exp(-(imgr[i] - imgr[i+1])*(imgr[i] - imgr[i+1])/(2*sigma*sigma)) + alpha))
            if y>0:
                self.nlinks[i].append(nlink(i-r,
                    (1-lmbda)*np.exp(-(imgr[i] - imgr[i-r])*(imgr[i] - imgr[i-r])/(2*sigma*sigma)) + alpha))
            if y<c-1:
                self.nlinks[i].append(nlink(i+r,
                    (1-lmbda)*np.exp(-(imgr[i] - imgr[i+r])*(imgr[i] - imgr[i+r])/(2*sigma*sigma)) + alpha))
            if z>0:
                self.nlinks[i].append(nlink(i-rc,
                    (1-lmbda)*np.exp(-(imgr[i] - imgr[i-rc])*(imgr[i] - imgr[i-rc])/(2*sigma*sigma)) + alpha))
            if z<d-1:
                self.nlinks[i].append(nlink(i+rc,
                    (1-lmbda)*np.exp(-(imgr[i] - imgr[i+rc])*(imgr[i] - imgr[i+rc])/(2*sigma*sigma)) + alpha))

    def segment(self, img, nlinks=None, tlinks=None, pdffore=None, source=None, sink=None, nbins=16, sigma=1, alpha=.1, lmbda=0.5):
        self.N = N = np.prod(np.shape(img))
        # if nlinks/tlinks are predefined, use them
        if not (nlinks is None):
            self.initEdges(nlinks=nlinks, tlinks=tlinks)

        else:
            # init edges using seeds for fore and background to estimate pdfs
            imgf = np.ravel(img,order='F')
            mn = np.min(img)
            mx = np.max(img)
            if not (pdffore is None):
                PrS = pdffore
            else:

                histsource = np.zeros(nbins)
                histsink = np.zeros(nbins)
                for i in source:
                    histsource[np.floor((nbins-1e-4)*(imgf[i]-mn)/(mx-mn)).astype(np.longlong)] += 1
                for i in sink:
                    histsink[np.floor((nbins-1e-4)*(imgf[i]-mn)/(mx-mn)).astype(np.longlong)] += 1
                pdfsource = histsource/np.sum(histsource)
                pdfsink = histsink/np.sum(histsink)
                totpdf = pdfsource + pdfsink
                totpdf[totpdf==0] = 1
                PrS = pdfsource/totpdf
                PrS[totpdf==0] = 0.5
                PrS[PrS<1e-5] = 1e-5
                PrS[PrS>1-1e-5] = 1-1e-5
            self.initEdges(img, PrS[((nbins-1e-4)*(imgf-mn)/(mx-mn)).astype(np.longlong)], sigma, alpha, lmbda)
            # setting hard constraints on user provided seed voxels
            K=0
            for n in self.nlinks:
                tot=0
                for nn in n:
                    tot += nn.cap
                if tot+1>K:
                    K = tot+1
            self.tlinks[source,0] = K
            self.tlinks[source,1] = 0
            self.tlinks[sink,0] = 0
            self.tlinks[sink, 1] = K

        # initialize arrays
        self.totcap = 0
        self.tree = np.zeros(N+2, dtype=np.longlong)
        self.parent = -np.ones(N+2, dtype=np.longlong)
        self.tree[N] = 1
        self.tree[N+1]=2

        iter=0
        while 1:
            iter+=1
            self.active = np.zeros(N + 2,dtype=np.longlong)
            self.active[N] = self.active[N + 1] = 1
            self.fifo = [N, N+1]
            self.tree[0:N]=0
            P = self.grow()
            ###Uncomment below to run  debugging function. Will impact speed.
            # if self.activeCheck()>0:
            #     print('Error some inactive nodes should be active')
            if len(P)==0:
                break

            self.augment(P)

        return np.reshape(self.tree[0:N], np.shape(img), order='F')==1

    def grow(self):
        # implements the grow function
        pass

    # Edge function needed to account for flow in s->t direction
    def edgeFunc(self, n):
        # returns a list of datatype nlink containing the directional nlinks
        pass

    # trace full path from s->t
    def tracePath(self, p, q):
        pass

    # augment path P
    def augment(self,P):
        # find bottleneck capacity
        btlnck = np.min([self.tlinks[P[1],0], self.tlinks[P[-2],1]])
        for i in range(len(P)-3):
            for n in self.nlinks[P[i+1]]:
                if n.neib == P[i+2]:
                    if n.cap < btlnck:
                        btlnck = n.cap
                    break

        self.totcap += btlnck
        # augment tlinks
        self.tlinks[P[1],0] -= btlnck
        if self.tlinks[P[1],0]<=0 and self.tree[P[1]]==1:
            self.parent[P[1]] = -1

        self.tlinks[P[-2],1] -= btlnck
        if self.tlinks[P[-2],1]<=0 and self.tree[P[-2]]==2:
            self.parent[P[-2]] = -1

        #augment nlinks
        for i in range(len(P)-3):
            p = P[i+1]
            q = P[i+2]
            # add capacity in the q to p direction
            for n in self.nlinks[q]: ## pass reference
                if n.neib==p:
                    n.cap += btlnck # what if it was zero and is now non-zero adjacent to inactive node? its ok no nodes in P are free
                    break

            # reduce capacity in p to q direction
            for n in self.nlinks[p]:
                if n.neib==q:
                    n.cap -= btlnck
                    if n.cap <=0 and self.tree[p]==self.tree[q]:
                        if self.tree[p]==1:
                            self.parent[q] = -1
                        else:
                            self.parent[p] = -1

    # debugging function -- anytime there is a treed node next to a free one it must be active,
    # so the total capacity between freed and inactive nodes should be always zero
    def activeCheck(self):
        totcap = 0
        for i in range(len(self.nlinks)):
            for nn in self.edgeFunc(i):
                if self.tree[i] !=0 and self.active[i]==0 and self.tree[nn.neib]==0:
                    totcap += nn.cap

        if np.sum(self.tree[0:self.N]==0)>0:
            if self.active[self.N]==0:
                totcap += np.sum(self.tlinks[self.tree[0:self.N]==0,0])
            if self.active[self.N+1]==0:
                totcap += np.sum(self.tlinks[self.tree[0:self.N]==0,1])
        return totcap
