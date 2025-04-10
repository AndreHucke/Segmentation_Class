# % Class to perform level set segmentation
# % gradientNB, curvatureNB, DuDt2 functions need to be added
# % ECE 8396: Medical Image Segmentation
# % Spring 2025
# % Author: Prof. Jack Noble; jack.noble@vanderbilt.edu
# % Modified by: Andre Hucke > Added gradientNB, curvatureNB, DuDt2 functions
# % Parts of this code were created using AI. All code was reviewed and modified by the author.

import numpy as np
from fastMarching import *
import skimage.filters
import skimage.feature
from laplacianSmoothing import *

class levelSetParams:
    def __init__(self, method=None, alpha=.2, mindist=3.1, sigma=.5, inflation=1, tau=1, beta=0.1,
                 epsilon=1e-7, maxiter=1000, convthrsh=1e-2, reinitrate=1, visrate=0, dtt=None,
                 lmbda=1, mu=1, gvft=.9):
        if method is None:
            self.method='CS' # or 'CV' or 'GVF'
        else:
            self.method=method

        self.alpha = alpha # slide 12
        self.mindist = mindist # size of narrow band
        self.sigma = sigma # slide 12
        self.v = inflation # slide 34: gamma
        self.tau = tau # slide 34
        self.beta = beta # slide 12
        self.epsilon = epsilon # slide 12
        self.maxiter = maxiter # max number of level set updates
        self.convthrsh = convthrsh # convergence threshold (early stopping criterion)
        self.reinitrate = reinitrate # how many LS updates to do before re-running fast marching
        self.visrate = visrate # how often to update visualization
        if dtt is None:
            self.dtt = np.ones(maxiter)
        else:
            self.dtt = dtt # slide 13 dt (time step for level set update)

        self.lmbda = lmbda # slide 36
        self.mu = mu     # slide 36
        self.gvft = gvft # slide 39 rho


class levelSet:
    def __init__(self):
        self.fm = fastMarching()
        self.r=self.c=self.d=0
        self.normG = None

    def DuDt1(self,v,tau,speed,nG,kappa,G,gradspeed): # Casselle Sapiro
        return speed * nG * (kappa + v) + tau * np.sum(G * gradspeed,axis=0) # slide 34
        
    def DuDt2(self, mu, lmbda, kappa, img, c1, c2):
        """
        Compute level set update for Chan-Vese method.
        
        Args:
            mu: Weight for curvature term
            lmbda: Weight for image-based term
            kappa: Curvature values at narrow band voxels
            img: Image intensity values at narrow band voxels
            c1: Mean intensity inside the contour (foreground)
            c2: Mean intensity outside the contour (background)
            
        Returns:
            dudt: Level set update values for narrow band voxels
        """
        # Chan-Vese energy terms
        # F1(I) = (I - c1)²
        # F2(I) = (I - c2)²
        F1 = (img - c1)**2
        F2 = (img - c2)**2
        
        # Regularized Chan-Vese update formula
        # dudt = μ * κ - λ * (F1 - F2)
        dudt = mu * kappa - lmbda * (F1 - F2)
        
        return dudt

    def DuDt4(self, mu, tau, kappa, G, gvf): # GVF LS
        return mu*kappa + tau * np.sum(G * gvf, axis=0)

    def gradientImage(self, img):

        gradim = np.zeros((3,self.r,self.c,self.d))
        gradim[0,1:self.r - 1,:,:] = (img[2:,:,:] - img[0:self.r - 2,:,:]) / 2
        gradim[0,0,:,:] = img[1,:,:] - img[0,:,:]
        gradim[0,-1,:,:] = img[-1,:,:] - img[-2,:,:]
        gradim[1,:,1:self.c - 1,:] = (img[:,2:,:] - img[:,0:self.c - 2,:]) / 2
        gradim[1,:,0,:] = img[:,1,:] - img[:,0,:]
        gradim[1,:,-1,:] = img[:,-1,:] - img[:,-2,:]
        if self.d>1:
            gradim[2,:,:,1:self.d - 1] = (img[:,:,2:] - img[:,:,0:self.d - 2]) / 2
            gradim[2,:,:,0] = img[:,:,1] - img[:,:,0]
            gradim[2,:,:,-1] = img[:,:,-1] - img[:,:,-2]
        return gradim

    def gradientNB(self):
        """
        Compute gradients of the distance map for voxels in the narrow band.
        
        Returns:
            G: Gradient vectors (3xN array)
            xyz: Coordinates of the narrow band voxels (3xN array)
            xf, xb, yf, yb, zf, zb: Forward and backward neighbor coordinates
        """
        # Get the narrow band voxels
        nbin, nbout = self.fm.getNB()
        nb = nbin + nbout
        
        # Get the number of voxels in the narrow band
        N = len(nb)
        if N == 0:
            return np.zeros((3, 0)), np.zeros((3, 0)), [], [], [], [], [], []
        
        # Extract coordinates
        x = np.array([node.x for node in nb], dtype=int)
        y = np.array([node.y for node in nb], dtype=int)
        z = np.array([node.z for node in nb], dtype=int)
        
        # Create the coordinate array
        xyz = np.vstack((x, y, z))
        
        # Compute forward and backward neighbors with boundary handling
        xf = np.minimum(x + 1, self.r - 1)
        xb = np.maximum(x - 1, 0)
        yf = np.minimum(y + 1, self.c - 1)
        yb = np.maximum(y - 1, 0)
        zf = np.minimum(z + 1, self.d - 1)
        zb = np.maximum(z - 1, 0)
        
        # Initialize gradient array
        G = np.zeros((3, N))
        
        # Compute gradients using central differences where possible
        # At boundaries, forward or backward differences are automatically used
        # The denominator is 2 for central differences and 1 for one-sided differences
        G[0, :] = (self.fm.dmap[xf, y, z] - self.fm.dmap[xb, y, z]) / np.where(xf != xb, 2.0, 1.0)
        G[1, :] = (self.fm.dmap[x, yf, z] - self.fm.dmap[x, yb, z]) / np.where(yf != yb, 2.0, 1.0)
        G[2, :] = (self.fm.dmap[x, y, zf] - self.fm.dmap[x, y, zb]) / np.where(zf != zb, 2.0, 1.0)
        
        return G, xyz, xf, xb, yf, yb, zf, zb

    def curvatureNB(self, epsilon):
        """
        Compute curvature of the level set function for voxels in the narrow band.
        
        Args:
            epsilon: Minimum permissible value for gradient norms to avoid division by zero
            
        Returns:
            kappa: Curvature values (1 x N array)
            G: Gradient vectors (3 x N array)
            nG: Gradient norms (1 x N array)
            xyz: Coordinates of the narrow band voxels (3 x N array)
        """
        # Get gradients and coordinates from gradientNB
        G, xyz, xf, xb, yf, yb, zf, zb = self.gradientNB()
        
        # If the narrow band is empty, return empty arrays
        N = G.shape[1]
        if N == 0:
            return np.array([]), G, np.array([]), xyz
        
        # Compute norm of gradients
        nG = np.sqrt(np.sum(G * G, axis=0))
        
        # Normalize gradients, handling small gradient norms
        nG_safe = np.maximum(nG, epsilon)
        N_grad = G / nG_safe  # Normalized gradient field N = ∇u/|∇u|
        
        # Extract coordinates for readability
        x, y, z = xyz[0, :], xyz[1, :], xyz[2, :]
        
        # Initialize curvature array
        kappa = np.zeros(N)
        
        # Compute divergence of normalized gradient field (mean curvature)
        for i in range(N):
            # Compute partial derivatives of each component of the normalized gradient
            # div(N) = ∂Nx/∂x + ∂Ny/∂y + ∂Nz/∂z
            
            # Calculate each partial derivative using central differences where possible
            # For ∂Nx/∂x
            if xf[i] != xb[i]:  # Not at boundary
                dx = xf[i] - xb[i]
                # Forward difference for Nx at (x+1,y,z)
                g_xf = np.zeros(3)
                g_xf[0] = self.fm.dmap[xf[i], y[i], z[i]] - self.fm.dmap[x[i], y[i], z[i]]
                g_xf[1] = (self.fm.dmap[xf[i], min(y[i]+1, self.c-1), z[i]] - 
                           self.fm.dmap[xf[i], max(y[i]-1, 0), z[i]]) / 2
                g_xf[2] = (self.fm.dmap[xf[i], y[i], min(z[i]+1, self.d-1)] - 
                           self.fm.dmap[xf[i], y[i], max(z[i]-1, 0)]) / 2
                
                # Backward difference for Nx at (x-1,y,z)
                g_xb = np.zeros(3)
                g_xb[0] = self.fm.dmap[x[i], y[i], z[i]] - self.fm.dmap[xb[i], y[i], z[i]]
                g_xb[1] = (self.fm.dmap[xb[i], min(y[i]+1, self.c-1), z[i]] - 
                           self.fm.dmap[xb[i], max(y[i]-1, 0), z[i]]) / 2
                g_xb[2] = (self.fm.dmap[xb[i], y[i], min(z[i]+1, self.d-1)] - 
                           self.fm.dmap[xb[i], y[i], max(z[i]-1, 0)]) / 2
                
                # Normalize the gradients
                g_norm_xf = max(np.sqrt(np.sum(g_xf * g_xf)), epsilon)
                g_norm_xb = max(np.sqrt(np.sum(g_xb * g_xb)), epsilon)
                
                nx_xf = g_xf[0] / g_norm_xf
                nx_xb = g_xb[0] / g_norm_xb
                
                dNx_dx = (nx_xf - nx_xb) / dx
            else:
                dNx_dx = 0
                
            # Similar calculations for ∂Ny/∂y
            if yf[i] != yb[i]:
                dy = yf[i] - yb[i]
                # Forward and backward differences for y component
                g_yf = np.zeros(3)
                g_yf[0] = (self.fm.dmap[min(x[i]+1, self.r-1), yf[i], z[i]] - 
                           self.fm.dmap[max(x[i]-1, 0), yf[i], z[i]]) / 2
                g_yf[1] = self.fm.dmap[x[i], yf[i], z[i]] - self.fm.dmap[x[i], y[i], z[i]]
                g_yf[2] = (self.fm.dmap[x[i], yf[i], min(z[i]+1, self.d-1)] - 
                           self.fm.dmap[x[i], yf[i], max(z[i]-1, 0)]) / 2
                
                g_yb = np.zeros(3)
                g_yb[0] = (self.fm.dmap[min(x[i]+1, self.r-1), yb[i], z[i]] - 
                           self.fm.dmap[max(x[i]-1, 0), yb[i], z[i]]) / 2
                g_yb[1] = self.fm.dmap[x[i], y[i], z[i]] - self.fm.dmap[x[i], yb[i], z[i]]
                g_yb[2] = (self.fm.dmap[x[i], yb[i], min(z[i]+1, self.d-1)] - 
                           self.fm.dmap[x[i], yb[i], max(z[i]-1, 0)]) / 2
                
                g_norm_yf = max(np.sqrt(np.sum(g_yf * g_yf)), epsilon)
                g_norm_yb = max(np.sqrt(np.sum(g_yb * g_yb)), epsilon)
                
                ny_yf = g_yf[1] / g_norm_yf
                ny_yb = g_yb[1] / g_norm_yb
                
                dNy_dy = (ny_yf - ny_yb) / dy
            else:
                dNy_dy = 0
                
            # Similar calculations for ∂Nz/∂z
            if self.d > 1 and zf[i] != zb[i]:
                dz = zf[i] - zb[i]
                # Forward and backward differences for z component
                g_zf = np.zeros(3)
                g_zf[0] = (self.fm.dmap[min(x[i]+1, self.r-1), y[i], zf[i]] - 
                           self.fm.dmap[max(x[i]-1, 0), y[i], zf[i]]) / 2
                g_zf[1] = (self.fm.dmap[x[i], min(y[i]+1, self.c-1), zf[i]] - 
                           self.fm.dmap[x[i], max(y[i]-1, 0), zf[i]]) / 2
                g_zf[2] = self.fm.dmap[x[i], y[i], zf[i]] - self.fm.dmap[x[i], y[i], z[i]]
                
                g_zb = np.zeros(3)
                g_zb[0] = (self.fm.dmap[min(x[i]+1, self.r-1), y[i], zb[i]] - 
                           self.fm.dmap[max(x[i]-1, 0), y[i], zb[i]]) / 2
                g_zb[1] = (self.fm.dmap[x[i], min(y[i]+1, self.c-1), zb[i]] - 
                           self.fm.dmap[x[i], max(y[i]-1, 0), zb[i]]) / 2
                g_zb[2] = self.fm.dmap[x[i], y[i], z[i]] - self.fm.dmap[x[i], y[i], zb[i]]
                
                g_norm_zf = max(np.sqrt(np.sum(g_zf * g_zf)), epsilon)
                g_norm_zb = max(np.sqrt(np.sum(g_zb * g_zb)), epsilon)
                
                nz_zf = g_zf[2] / g_norm_zf
                nz_zb = g_zb[2] / g_norm_zb
                
                dNz_dz = (nz_zf - nz_zb) / dz
            else:
                dNz_dz = 0
                
            # Mean curvature is the divergence of the normalized gradient field
            kappa[i] = dNx_dx + dNy_dy + dNz_dz
        
        return kappa, G, nG, xyz

    def segment(self, img, dmap_init, params=levelSetParams(), ax=None):
        self.r,self.c,self.d = np.shape(img)
        self.normG = np.zeros((3,self.r,self.c,self.d))

        if params.method != 'CV':
            if params.sigma > 0:
                imblur = skimage.filters.gaussian(img, params.sigma)
            else:
                imblur = img

            gradim = self.gradientImage(imblur)
            ngradimsq = np.sum(gradim*gradim, axis=0)
            speed = np.exp(-ngradimsq / (2 * params.alpha * params.alpha)) - params.beta # see slide 12
            speed[speed<params.epsilon] = params.epsilon
            gradspeed = self.gradientImage(speed) # slide 34 CS method

        if params.method == 'GVF':
            X,Y,Z = np.meshgrid(range(self.r), range(self.c), range(self.d), indexing='ij')
            X = np.ravel(X, order='F')
            Y = np.ravel(Y, order='F')
            Z = np.ravel(Z, order='F')
            ngradspeed = np.linalg.norm(gradspeed, axis=0)
            mx = np.max(ngradspeed)
            msk = np.ravel(ngradspeed>params.gvft*mx, order='F')
            imsk = 1-msk
            dirn = X[msk] + self.r*(Y[msk] + self.c*Z[msk])
            intn = X[imsk] + self.r*(Y[imsk] + self.c*Z[imsk])
            gvfx = laplacianSmoothing(self.r,self.c,self.d,dirn,gradspeed[0,X[msk],Y[msk],Z[msk]],
                                      intn,gradspeed[0, X[imsk], Y[imsk], Z[imsk]],
                                      (ngradspeed[X[imsk],Y[imsk],Z[imsk]] / (params.gvft * mx))**2) # see 39
            gvfy = laplacianSmoothing(self.r,self.c,self.d,dirn,gradspeed[1,X[msk],Y[msk],Z[msk]],
                                      intn,gradspeed[1,X[imsk],Y[imsk],Z[imsk]],
                                      (ngradspeed[X[imsk],Y[imsk],Z[imsk]] / (params.gvft * mx))**2)
            gvfz = laplacianSmoothing(self.r,self.c,self.d,dirn,gradspeed[2,X[msk],Y[msk],Z[msk]],
                                      intn,gradspeed[2,X[imsk],Y[imsk],Z[imsk]],
                                      (ngradspeed[X[imsk],Y[imsk],Z[imsk]] / (params.gvft * mx))**2)
            gvf = np.concatenate((gvfx[np.newaxis,:,:,:], gvfy[np.newaxis,:,:,:], gvfz[np.newaxis,:,:,:]), axis=0)
            if params.visrate>0:
                f = plt.gcf()
                fig, axgvf = plt.subplots(2,2)
                axgvf[0,0].imshow(gvfx[:,:,self.d//2].T, 'gray')
                plt.axes(axgvf[0,0])
                plt.xlabel('GVF(x)')
                axgvf[0,1].imshow(gvfy[:,:,self.d//2].T, 'gray')
                plt.axes(axgvf[0,1])
                plt.xlabel('GVF(y)')
                axgvf[1,1].imshow(img[:,:,self.d//2].T, 'gray')
                plt.axes(axgvf[1,1])
                axgvf[1,0].imshow(img[:,:,self.d // 2].T,'gray')
                plt.axes(axgvf[1,0])
                plt.figure(f)


        iter=0
        self.fm.dmap = np.array(dmap_init, dtype=np.float64)
        self.fm.voxsz = np.array([1.,1.,1.])
        delta=params.convthrsh+1

        while iter < params.maxiter:
            if iter%params.reinitrate==0:
                self.fm.update(nbdist=params.mindist)

                nbin, nbout = self.fm.getNB()
                nbinone = nbin #useful for convergence
                nboutone = nbout
                for i in range(len(nbin)):
                    if nbin[i].d>1:
                        nbinone = nbin[0:i]
                        break
                for i in range(len(nbout)):
                    if nbout[i].d>1:
                        nboutone = nbout[0:i]
                        break
                if iter>0: # check convergence
                    delta = 0
                    for i in range(len(nbinold)):
                        delta += np.abs(-self.fm.dmap[ nbinold[i].x,  nbinold[i].y,  nbinold[i].z] -  nbinold[i].d)
                    for i in range(len(nboutold)):
                        delta += np.abs(self.fm.dmap[nboutold[i].x, nboutold[i].y, nboutold[i].z] - nboutold[i].d)
                    N = len(nbinold) + len(nboutold)
                    if N==0:
                        break
                    delta /= len(nbinold) + len(nboutold)
                    if delta<params.convthrsh:
                        break

                nbinold = nbinone
                nboutold = nboutone

            kappa, G, nG, xyz = self.curvatureNB(params.epsilon)
            if np.sum(np.isnan(kappa))>0:
                self.fm.update(nbdist=params.mindist)
                kappa,G,nG,xyz = self.curvatureNB(params.epsilon)

            if (params.method != 'CV') and (params.method != 'GVF'):
                dudt = self.DuDt1(params.v,params.tau,speed[xyz[0,:], xyz[1,:], xyz[2,:]],nG,kappa, G,
                                  np.concatenate((
                                      gradspeed[0,xyz[0,:],xyz[1,:],xyz[2,:]][np.newaxis,:],
                                      gradspeed[1,xyz[0,:],xyz[1,:],xyz[2,:]][np.newaxis,:],
                                      gradspeed[2,xyz[0,:],xyz[1,:],xyz[2,:]][np.newaxis,:]), axis=0))
            elif params.method == 'CV':
                msk = self.fm.dmap<=0
                c1 = np.mean(img[msk])
                c2 = np.mean(img[1-msk])
                dudt = self.DuDt2(params.mu, params.lmbda, kappa, img[xyz[0,:], xyz[1,:], xyz[2,:]], c1, c2) # see slide 36

            elif params.method == 'GVF':
                gvfl = gvf[:,xyz[0,:],xyz[1,:],xyz[2,:]]
                dudt = self.DuDt4(params.mu, params.tau, kappa, G, gvfl)

            if len(dudt)>0:
                dt = params.dtt[iter] / np.max(np.abs(dudt))
                self.fm.dmap[xyz[0,:], xyz[1,:], xyz[2,:]] += dt * dudt # do level set update
                if params.visrate>0 and iter%params.visrate==0:
                    plt.axes(ax[0])
                    plt.cla()
                    ax[0].imshow(img[:,:,self.d//2].T, 'gray')
                    X,Y = np.meshgrid(range(self.r),range(self.c),indexing="ij")
                    plt.contour(X,Y,self.fm.dmap[:,:,self.d // 2],levels=[0.0],colors='red')
                    plt.title(f'Iteration {iter}, Delta={delta}')
                    plt.axes(ax[1])
                    plt.cla()
                    plt.gca().imshow(self.fm.dmap[:,:,self.d//2].T, 'gray', vmin=-10, vmax=20)
                    plt.contour(X,Y,self.fm.dmap[:,:,self.d // 2],levels=[0.0],colors='red')
                    plt.title('Distance map')
                    plt.gcf().canvas.draw_idle()
                    plt.gcf().canvas.start_event_loop(0.01)
            iter += 1

        self.fm.update(nbdist=params.mindist)
        return self.fm.dmap



