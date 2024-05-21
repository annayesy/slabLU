import torch
from functools import reduce
from time import time

from scipy.sparse import kron, diags, block_diag
from scipy.sparse import eye as speye
import scipy.sparse.linalg as spla
import numpy as np

def torch_setdiff1d(t1,t2):
    return torch.from_numpy(np.setdiff1d(t1.numpy(),t2.numpy()))

def get_inds_2d(XX,box_geom,h,n0,n1):
    I_L = torch.argwhere(XX[0,:] < 0.5 * h + box_geom[0,0])
    I_L = I_L.clone().reshape(n1,)
    I_R = torch.argwhere(XX[0,:] > -0.5 * h + box_geom[0,1])
    I_R = I_R.clone().reshape(n1,)
    I_D = torch.argwhere(XX[1,:] < 0.5 * h + box_geom[1,0])
    I_D = I_D.clone().reshape(n0,)
    I_U = torch.argwhere(XX[1,:] > -0.5 * h + box_geom[1,1])
    I_U = I_U.clone().reshape(n0,) 
    
    I_DIR = torch.hstack((I_D,I_U))
    I_DIR = torch.unique(I_DIR)
    I_L = torch_setdiff1d(I_L,I_DIR)
    I_R = torch_setdiff1d(I_R,I_DIR)
    return I_L,I_R,I_DIR
    
def get_inds_3d(XX,box_geom,h,n0,n1,n2):
    I_L = torch.argwhere(XX[0,:] < 0.5 * h + box_geom[0,0])
    I_L = I_L.clone().reshape(n1*n2,)
    I_R = torch.argwhere(XX[0,:] > -0.5 * h + box_geom[0,1])
    I_R = I_R.clone().reshape(n1*n2,)
    I_D = torch.argwhere(XX[1,:] < 0.5 * h + box_geom[1,0])
    I_D = I_D.clone().reshape(n0*n2,)
    I_U = torch.argwhere(XX[1,:] > -0.5 * h + box_geom[1,1])
    I_U = I_U.clone().reshape(n0*n2,)

    I_B = torch.argwhere(XX[2,:] < 0.5 * h + box_geom[2,0])
    I_B = I_B.clone().reshape(n0*n1,)
    I_F = torch.argwhere(XX[2,:] > -0.5 * h + box_geom[2,1])
    I_F = I_F.clone().reshape(n0*n1,)
    
    I_DIR = torch.hstack((I_D,I_U,I_B,I_F))
    I_DIR = torch.unique(I_DIR)
    I_L   = torch_setdiff1d(I_L,I_DIR)
    I_R   = torch_setdiff1d(I_R,I_DIR)
    return I_L,I_R,I_DIR
        

def grid(box_geom,h):
    d = box_geom.shape[0]
    xx0 = torch.arange(box_geom[0,0],box_geom[0,1]+0.5*h,h)
    xx1 = torch.arange(box_geom[1,0],box_geom[1,1]+0.5*h,h)
    if (d == 3):
        xx2 = torch.arange(box_geom[2,0],box_geom[2,1]+0.5*h,h)

    if (d == 2):
        n0 = xx0.shape[0]
        n1 = xx1.shape[0]

        XX0 = torch.repeat_interleave(xx0,n1)
        XX1 = torch.repeat_interleave(xx1,n0).reshape(-1,n0).T.flatten()
        XX = torch.vstack((XX0,XX1))
        I_X_inds = get_inds_2d(XX,box_geom,h,n0,n1)
        I_X = torch.hstack((I_X_inds))
        ns = torch.tensor([n0,n1])
        
    elif (d == 3):
        n0 = xx0.shape[0]
        n1 = xx1.shape[0]
        n2 = xx2.shape[0]

        XX0 = torch.repeat_interleave(xx0,n1*n2)
        XX1 = torch.repeat_interleave(xx1,n0*n2).reshape(-1,n0).T.flatten()
        XX2 = torch.repeat_interleave(xx2,n0*n1).reshape(-1,n0*n1).T.flatten()
        XX = torch.vstack((XX0,XX1,XX2))
        I_X_inds = get_inds_3d(XX,box_geom,h,n0,n1,n2)
        I_X = torch.hstack(I_X_inds)
        ns = torch.tensor([n0,n1,n2])
    I_X = torch.unique(I_X)
    return XX,I_X_inds,I_X,ns

class FD_disc:
    def __init__(self,box_geom,h,pdo_op):
        XX, inds_tuple, self.I_X, self.ns = grid(box_geom,h)
        self.XX = XX.T
        self.h = h
        self.box_geom = box_geom
        self.d = self.ns.shape[0]
        
        self.I_L,self.I_R,self.I_DIR = inds_tuple

        I_tot = torch.arange(self.XX.shape[0])
        self.I_C = torch_setdiff1d(I_tot,self.I_X)
        self.pdo_op = pdo_op
        
    def assemble_sparse(self):
        h = self.h; pdo_op = self.pdo_op
        if (self.d == 2):

            n0,n1 = self.ns
            d0sq = (1/(h*h)) * diags([1, -2, 1], [-1, 0, 1], shape=(n0, n0),format='csc')
            d1sq = (1/(h*h)) * diags([1, -2, 1], [-1, 0, 1], shape=(n1, n1),format='csc')
            
            d0   = (1/(2*h)) * diags([-1, 0, +1], [-1, 0, 1], shape=(n0,n0),format='csc')
            d1   = (1/(2*h)) * diags([-1, 0, +1], [-1, 0, 1], shape=(n1,n1),format='csc')

            D00 = kron(d0sq,speye(n1))
            D11 = kron(speye(n0),d1sq)
            
            c00_diag = np.array(pdo_op.c11(self.XX)).reshape(n0*n1,)
            C00 = diags(c00_diag, 0, shape=(n0*n1,n0*n1))
            c11_diag = np.array(pdo_op.c22(self.XX)).reshape(n0*n1,)
            C11 = diags(c11_diag, 0, shape=(n0*n1,n0*n1))
                        
            A = - C00 @ D00 - C11 @ D11
            
            if (pdo_op.c12 is not None):
                c_diag = np.array(pdo_op.c12(self.XX)).reshape(n0*n1,)
                S      = diags(c_diag,0,shape=(n0*n1,n0*n1))
                
                D01 = kron(d0,d1)
                A  -= 2 * S @ D01
                
            if (pdo_op.c1 is not None):
                c_diag = np.array(pdo_op.c1(self.XX)).reshape(n0*n1,)
                S      = diags(c_diag,0,shape=(n0*n1,n0*n1))
                
                D0 = kron(d0,speye(n1))
                A  += S @ D0
            
            if (pdo_op.c2 is not None):
                c_diag = np.array(pdo_op.c1(self.XX)).reshape(n0*n1,)
                S      = diags(c_diag,0,shape=(n0*n1,n0*n1))
                
                D0 = kron(speye(n0),d1)
                A  += S @ D1

            if (pdo_op.c is not None):
                c_diag = np.array(pdo_op.c(self.XX)).reshape(n0*n1,)
                S = diags(c_diag, 0, shape=(n0*n1,n0*n1))
                A += S

        elif (self.d == 3):

            n0,n1,n2 = self.ns
            d0sq = (1/(h*h)) * diags([1, -2, 1], [-1, 0, 1], shape=(n0, n0),format='csc')
            d1sq = (1/(h*h)) * diags([1, -2, 1], [-1, 0, 1], shape=(n1, n1),format='csc')
            d2sq = (1/(h*h)) * diags([1, -2, 1], [-1, 0, 1], shape=(n2, n2),format='csc')

            D00 = kron(d0sq,kron(speye(n1),speye(n2)))
            D11 = kron(speye(n0),kron(d1sq,speye(n2)))
            D22 = kron(speye(n0),kron(speye(n1),d2sq))
            
            N = n0*n1*n2
            c00_diag = np.array(pdo_op.c11(self.XX)).reshape(N,)
            C00 = diags(c00_diag, 0, shape=(N,N))
            c11_diag = np.array(pdo_op.c22(self.XX)).reshape(N,)
            C11 = diags(c11_diag, 0, shape=(N,N))
            c22_diag = np.array(pdo_op.c33(self.XX)).reshape(N,)
            C22 = diags(c22_diag, 0, shape=(N,N))

            A = - C00 @ D00 - C11 @ D11 - C22 @ D22
            
            if ((pdo_op.c1 is not None) or \
                (pdo_op.c2 is not None) or \
                (pdo_op.c3 is not None) or \
                (pdo_op.c12 is not None) or \
                (pdo_op.c13 is not None) or \
                (pdo_op.c23 is not None)):
                raise ValueError
            
            if (pdo_op.c is not None):
                c_diag = np.array(pdo_op.c(self.XX)).reshape(N,)
                S = diags(c_diag, 0, shape=(N,N))
                A += S
                
        return A.tocsr()