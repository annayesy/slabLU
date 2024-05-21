import numpy as np
from collections import namedtuple
import scipy
from time import time
import numpy.polynomial.chebyshev as cheb_py
import scipy.linalg

Pdo_2d   = namedtuple('Pdo_2d',['c11','c22','c12', 'c1','c2','c'])
Ds_2d    = namedtuple('Ds_2d', ['D11','D22','D12','D1','D2'])
JJ_2d    = namedtuple('JJ_2d', ['Jl','Jr','Jd','Ju','Jx','Jc','Jxreorder'])

Pdo_3d   = namedtuple('Pdo_3d',['c11','c22','c33','c12','c13','c23','c1','c2','c3','c'])
Ds_3d    = namedtuple('Ds_3d', ['D11','D22','D33','D12','D13','D23','D1','D2','D3'])
JJ_3d    = namedtuple('JJ_3d', ['Jl','Jr','Jd','Ju','Jb','Jf','Jx','Jc','Jtot'])

def cheb(p):
    x = np.cos(np.pi * np.arange(p+1) / p)
    c = np.concatenate((np.array([2]), np.ones(p-1), np.array([2])))
    c = np.multiply(c,np.power(np.ones(p+1) * -1, np.arange(p+1)))
    X = x.repeat(p+1).reshape((-1,p+1))
    dX = X - X.T
    # create the off diagonal entries of D
    D = np.divide(np.outer(c,np.divide(np.ones(p+1),c)), dX + np.eye(p+1))
    D = D - np.diag(np.sum(D,axis=1))
    return D,x

def diag_mult(diag,M):
    return (diag * M.T).T

def get_loc_interp(x_cheb, x_cheb_nocorners, q):
    Vpoly_cheb = cheb_py.chebvander(x_cheb,q)
    Vpoly_nocorner = cheb_py.chebvander(x_cheb_nocorners,q)

    Interp_loc = np.linalg.lstsq(Vpoly_nocorner.T,Vpoly_cheb.T,rcond=None)[0].T
    err  = np.linalg.norm(Interp_loc @ Vpoly_nocorner - Vpoly_cheb)
    cond = np.linalg.cond(Interp_loc) 
    return Interp_loc,err,cond

class HPS_Disc:
    def __init__(self,a,p,d):
        self._discretize(a,p,d)
        self.a = a; self.p = p; self.d = d
        self._get_interp_mat()
        
    def _discretize(self,a,p,d):
        if (d == 2):
            self.zz,self.Ds,self.JJ,self.hmin = HPS_Disc.leaf_discretization_2d(a,p)
        else:
            self.zz,self.Ds,self.JJ,self.hmin = HPS_Disc.leaf_discretization_3d(a,p)
        self.Nx = HPS_Disc.get_diff_ops(self.Ds,self.JJ,d)
        
    ## Interpolation from data on Ix to Ix_reorder
    def _get_interp_mat(self):
        
        p = self.p; a = self.a
        
        x_cheb = self.zz[1,:p-1] + a
        x_cheb_nocorners  = self.zz[1,1:p-1] + a
        
        cond_min = 3.25; cond_max = 3.5; err_tol = 2e-8
        q = p+5; tic = time()
        while (True):
            Interp_loc,err,cond = get_loc_interp(x_cheb,x_cheb_nocorners,q)
            
            if ((cond < cond_min) or ((err > err_tol) and (cond < cond_max)) ):
                break
            else:
                q += 10
        toc = time() - tic;
        
        Interp_mat_chebfleg = scipy.linalg.block_diag(*np.repeat(np.expand_dims(Interp_loc,0),4,axis=0))
        perm = np.hstack((np.arange(p-2),\
                          np.arange(p-2)+3*(p-2),\
                          np.flip(np.arange(p-2)+1*(p-2),0),\
                          np.flip(np.arange(p-2)+2*(p-2),0)
                         ))
        perm = np.argsort(perm)
        self.Interp_mat = Interp_mat_chebfleg[:,perm]
        print ("--Interp_mat required lstsqfit of q=%d, condition number %5.5f with error %5.5e and time to calculate %12.5f"\
               % (q,cond,err,toc))
        
        
    def get_diff_ops(Ds,JJ,d):
        if (d == 2):
            Nl = Ds.D1[JJ.Jl]
            Nr = Ds.D1[JJ.Jr]
            Nd = Ds.D2[JJ.Jd]
            Nu = Ds.D2[JJ.Ju]

            Nx = np.concatenate((-Nl,+Nr,-Nd,+Nu))
        else:
            Nl = Ds.D1[JJ.Jl][:,JJ.Jtot]
            Nr = Ds.D1[JJ.Jr][:,JJ.Jtot]
            Nd = Ds.D2[JJ.Jd][:,JJ.Jtot]
            Nu = Ds.D2[JJ.Ju][:,JJ.Jtot]
            Nb = Ds.D3[JJ.Jb][:,JJ.Jtot]
            Nf = Ds.D3[JJ.Jf][:,JJ.Jtot]

            Nxtot = np.concatenate((-Nl,+Nr,-Nd,+Nu,-Nb,+Nf))
        return Nx
    
    
#################################### 2D discretization ##########################################

    def leaf_discretization_2d(a,p):
        zz,Ds = HPS_Disc.cheb_2d(a,p)
        hmin  = zz[1,1] - zz[0,1]

        Jc0   = np.abs(zz[0,:]) < a - 0.5*hmin
        Jc1   = np.abs(zz[1,:]) < a - 0.5*hmin
        Jl    = np.argwhere(np.logical_and(zz[0,:] < - a + 0.5 * hmin,Jc1))
        Jl    = Jl.copy().reshape(p-2,)
        Jr    = np.argwhere(np.logical_and(zz[0,:] > + a - 0.5 * hmin,Jc1))
        Jr    = Jr.copy().reshape(p-2,)
        Jd    = np.argwhere(np.logical_and(zz[1,:] < - a + 0.5 * hmin,Jc0))
        Jd    = Jd.copy().reshape(p-2,)
        Ju    = np.argwhere(np.logical_and(zz[1,:] > + a - 0.5 * hmin,Jc0))
        Ju    = Ju.copy().reshape(p-2,)
        Jc    = np.argwhere(np.logical_and(Jc0,Jc1))
        Jc    = Jc.copy().reshape((p-2)**2,)
        Jx    = np.concatenate((Jl,Jr,Jd,Ju))
        
        Jcorner = np.setdiff1d(np.arange(p**2),np.hstack((Jc,Jx)))
        
        Jl_corner = np.hstack((Jcorner[0],   Jl))
        Ju_corner = np.hstack((Jcorner[0+1], Ju))
        Jr_corner = np.hstack((Jcorner[0+3], np.flip(Jr,0)))
        Jd_corner = np.hstack((Jcorner[0+2], np.flip(Jd,0)))
        
        Jxreorder = np.hstack((Jl_corner,Ju_corner,Jr_corner,Jd_corner))
        
        JJ    = JJ_2d(Jl= Jl, Jr= Jr, Ju= Ju, Jd= Jd, 
                 Jx= Jx, Jc= Jc, Jxreorder=Jxreorder)
        return zz,Ds,JJ,hmin

    def cheb_2d(a,p):
        D,xvec = cheb(p-1)
        xvec = a * np.flip(xvec)
        D = (1/a) * D
        I = np.eye(p)
        D1 = -np.kron(D,I)
        D2 = -np.kron(I,D)
        Dsq = D @ D
        D11 = np.kron(Dsq,I)
        D22 = np.kron(I,Dsq)
        D12 = np.kron(D,D)

        zz1 = np.repeat(xvec,p)
        zz2 = np.repeat(xvec,p).reshape(-1,p).T.flatten()
        zz = np.vstack((zz1,zz2))
        Ds = Ds_2d(D1= D1, D2= D2, D11= D11, D22= D22, D12= D12)
        return zz, Ds 
        
        
#################################### 3D discretization ##########################################
    
    def leaf_discretization_3d(a,p):
        zz,Ds = HPS_Disc.cheb_3d(a,p)
        hmin  = zz[2,1] - zz[2,0]

        Jc0   = np.abs(zz[0,:]) < a - 0.5*hmin
        Jc1   = np.abs(zz[1,:]) < a - 0.5*hmin
        Jc2   = np.abs(zz[2,:]) < a - 0.5*hmin
        Jl    = np.argwhere(np.logical_and(zz[0,:] < - a + 0.5 * hmin,
                                           np.logical_and(Jc1,Jc2)))
        Jl    = Jl.copy().reshape((p-2)**2,)
        Jr    = np.argwhere(np.logical_and(zz[0,:] > + a - 0.5 * hmin,
                                           np.logical_and(Jc1,Jc2)))
        Jr    = Jr.copy().reshape((p-2)**2,)
        Jd    = np.argwhere(np.logical_and(zz[1,:] < - a + 0.5 * hmin,
                                           np.logical_and(Jc0,Jc2)))
        Jd    = Jd.copy().reshape((p-2)**2,)
        Ju    = np.argwhere(np.logical_and(zz[1,:] > + a - 0.5 * hmin,
                                           np.logical_and(Jc0,Jc2)))
        Ju    = Ju.copy().reshape((p-2)**2,)
        Jb    = np.argwhere(np.logical_and(zz[2,:] < - a + 0.5 * hmin,
                                           np.logical_and(Jc0,Jc1)))
        Jb    = Jb.copy().reshape((p-2)**2,)
        Jf    = np.argwhere(np.logical_and(zz[2,:] > + a - 0.5 * hmin,
                                           np.logical_and(Jc0,Jc1)))
        Jf    = Jf.copy().reshape((p-2)**2,)

        Jc    = np.argwhere(np.logical_and(Jc0,
                                           np.logical_and(Jc1,Jc2)))
        Jc    = Jc.copy().reshape((p-2)**3,)
        Jx    = np.concatenate((Jl,Jr,Jd,Ju,Jb,Jf))
        Jtot  = np.concatenate((Jx,Jc))
        JJ    = JJ_3d(Jl= Jl, Jr= Jr, Ju= Ju, Jd= Jd, Jb= Jb,
                 Jf=Jf, Jx=Jx, Jc=Jc, Jtot=Jtot)
        return zz,Ds,JJ,hmin

    def cheb_3d(a,p):
        D,xvec = cheb(p-1)
        xvec = a * np.flip(xvec)
        D = (1/a) * D
        I = np.eye(p)
        D1 = -np.kron(D,np.kron(I,I))
        D2 = -np.kron(I,np.kron(D,I))
        D3 = -np.kron(I,np.kron(I,D))
        Dsq = D @ D
        D11 = np.kron(Dsq,np.kron(I,I))
        D22 = np.kron(I,np.kron(Dsq,I))
        D33 = np.kron(I,np.kron(I,Dsq))
        D12 = np.kron(D,np.kron(D,I))
        D13 = np.kron(D,np.kron(I,D))
        D23 = np.kron(I,np.kron(D,D))

        zz1 = np.repeat(xvec,p*p)
        zz2 = np.repeat(xvec,p*p).reshape(p*p,p).T.flatten()
        zz3 = np.repeat(xvec,p*p).reshape(p,p*p).T.flatten()
        zz = np.vstack((zz1,zz2,zz3))
        Ds = Ds_3d(D1= D1, D2= D2, D3= D3, D11= D11, D22= D22, D33= D33,
             D12= D12, D13= D13, D23= D23)
        return zz, Ds
