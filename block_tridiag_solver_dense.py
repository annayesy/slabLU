import numpy as np
import torch
torch.set_default_dtype(torch.double)
from time import time
import sys

#################################### STATIC METHODS   ##################################

# func takes tensor of size b,m*bs,nrhs and returns result of same size
def recover_block_tridiag_via_sampling(b,m,bs,func,func_args=None,device=torch.device("cpu")):
    
    piv_data_nblocks = int(m/bs)+1
    Adata = torch.zeros(b,m + 2*(m-1)+piv_data_nblocks,bs,bs,device=device)
    diag_offset = 0; sub_offset = m; sup_offset = m + (m-1)
    
    num_rep = int(m/3); rem = m-num_rep*3
    rhs = torch.zeros(m*bs,np.min([m,3])*bs,device=device)

    Iden = torch.eye(np.min([3,m])*bs)
    if (num_rep > 0):
        rhs[:num_rep*3*bs] = Iden.repeat(num_rep,1,1).reshape(num_rep*3*bs,3*bs)
    rhs[num_rep*3*bs:] = Iden[:rem*bs]
    rhs = rhs.repeat(b,1,1)

    if (func_args is None):
        print("calling with func args none")
        result = func(func_args)
    else:
        result = func((func_args,rhs))
    result_rep = result[:,:num_rep*3*bs].reshape(b,num_rep,3*bs,3*bs)
    result_rem = result[:,num_rep*3*bs:]
    del result

    ## assign diagonal elements ##
    if (num_rep > 0):
        Adata[:,diag_offset+torch.arange(0,num_rep*3,3)] = result_rep[...,:bs,:bs]
        Adata[:,diag_offset+torch.arange(1,num_rep*3,3)] = result_rep[...,bs:2*bs,bs:2*bs]
        Adata[:,diag_offset+torch.arange(2,num_rep*3,3)] = result_rep[...,2*bs:,2*bs:]

    if (rem > 0):
        Adata[:,diag_offset+num_rep*3] = result_rem[:,:bs,:bs]
    if (rem == 2):
        Adata[:,diag_offset+num_rep*3+1] = result_rem[:,bs:2*bs,bs:2*bs]

    ## assign sub elements ##
    if (num_rep > 0):
        Adata[:,sub_offset+torch.arange(0,num_rep*3,3)] = result_rep[...,bs:2*bs,:bs]
        Adata[:,sub_offset+torch.arange(1,num_rep*3,3)] = result_rep[...,2*bs:,bs:2*bs]
    if (num_rep > 1):    
        Adata[:,sub_offset+torch.arange(2,(num_rep-1)*3,3)] = result_rep[:,1:,:bs,2*bs:]
    
    if ( (rem > 0) ):
        if ( m == 1 ):
            pass
        elif ( m == 2 ):
            Adata[:,sub_offset] = result_rem[:,bs:2*bs,:bs]
        else:
            Adata[:,sub_offset+num_rep*3-1] = result_rem[:,:bs,2*bs:]
    if ( (rem == 2) and (num_rep > 0)):
        Adata[:,sub_offset+num_rep*3] = result_rem[:,bs:2*bs,:bs]

    ## assign sup elements ##
    if (num_rep > 0):
        Adata[:,sup_offset+torch.arange(0,num_rep*3,3)] = result_rep[...,:bs,bs:2*bs]
        Adata[:,sup_offset+torch.arange(1,num_rep*3,3)] = result_rep[...,bs:2*bs,2*bs:]
    if (num_rep > 1):
        Adata[:,sup_offset+torch.arange(2,(num_rep-1)*3,3)] = result_rep[:,:num_rep-1,2*bs:,:bs]

    if (rem > 0):
        if ( m == 1 ):
            pass
        elif ( m == 2 ):
            Adata[:,sup_offset] = result_rem[:,:bs,bs:2*bs]
        else:
            Adata[:,sup_offset+num_rep*3-1] = result_rep[:,-1,2*bs:,:bs]
    if ( (rem == 2) and (num_rep > 0) ):
        Adata[:,sup_offset+num_rep*3] = result_rem[:,:bs,bs:2*bs]
    return Adata



###################################### HELPER FUNCTIONS #################################

def perm_rows(piv,M,transpose=True):
    b,m,n = M.shape
    b,nperm = piv.shape
    
    if (transpose):
        ran = range(piv.size(-1))
    else:
        ran = range(piv.size(-1)-1,-1,-1)

    for i in ran:
        ind1 = i 
        ind2 = piv[:,i] - 1
        ind2 = ind2.long()
        
        if (int(torch.count_nonzero(ind2 - ind1)) == 0):
            continue
        
        if (transpose):
            tmp = M[range(b),ind1]
            M[range(b),ind1] = M[range(b),ind2]
            M[range(b),ind2] = tmp
        else:
            tmp = M[range(b),ind2]
            M[range(b),ind2] = M[range(b),ind1]
            M[range(b),ind1] = tmp
    return M

# solve U \ rhs
def solveU(LU,rhs,transpose=False, fast_LU=True):
    if (fast_LU):
        if (transpose):
            return torch.transpose(torch.triu(LU),-1,-2) @ rhs
        else:
            return torch.triu(LU) @ rhs
    else:
        if (transpose):
            return  torch.linalg.solve_triangular(torch.transpose(LU,-1,-2),rhs,\
                                                  upper=False,unitriangular=False)
        else:
            return torch.linalg.solve_triangular(LU,rhs,upper=True,unitriangular=False)

def solveL(LU,piv,rhs,transpose=False, fast_LU = True):
    P,Linv,Uinv = torch.lu_unpack(LU,piv.contiguous())
    if (fast_LU):
        if (transpose):
            tmp = torch.transpose(Linv,-1,-2) @ rhs
            return P @ tmp
        else:
            rhs = torch.transpose(P,-1,-2) @ rhs
            return Linv @ rhs
    else: 
        if (transpose):
            tmp = torch.linalg.solve_triangular(torch.transpose(LU,-1,-2),rhs,\
                                         upper=True,unitriangular=True)
            return P @ tmp
        else:
            return torch.linalg.solve_triangular(LU,torch.transpose(P,-1,-2) @ rhs,\
                                                 upper=False,unitriangular=True)

def calc_tri_inv(LU):
    b = LU.shape[0]; bs = LU.shape[1]
    I = torch.eye(bs,device=LU.device).repeat(b,1,1)
    invL = torch.linalg.solve_triangular(LU,I,upper=False,unitriangular=True)
    invU = torch.linalg.solve_triangular(LU,I,upper=True,unitriangular=False)
    
    return torch.triu(invU) + torch.tril(invL,diagonal=1) - I

###################################### SWEEPING SOLVER #################################

def build_sweep(b,m,bs,Tdiag,Tsub,Tsup,fast_LU=True):

    pivs = torch.zeros((b,m,bs),device=Tdiag.device).int()
    sub_offset = m; sup_offset = m + m-1; piv_offset = m + 2*(m-1)
    
    LU,piv = torch.linalg.lu_factor(Tdiag[:,0])
    if (fast_LU):
        LU = calc_tri_inv(LU)
    Tdiag[:,0] = LU
    pivs[:,0] = piv
    
    for i in range(1,m):
        
        Ufactor = solveL(LU,pivs[:,i-1],Tsup[:,i-1],fast_LU=fast_LU)
        Lfactor = solveU(LU,torch.transpose(Tsub[:,i-1],-1,-2),transpose=True,fast_LU=fast_LU)
        Lfactor = torch.transpose(Lfactor,-1,-2)
        
        [LU,piv] = torch.linalg.lu_factor(Tdiag[:,i] - Lfactor@Ufactor)
        if (fast_LU):
            LU = calc_tri_inv(LU)
        Tdiag[:,i] = LU
        Tsup[:,i-1] = Ufactor
        Tsub[:,i-1] = Lfactor
        
        pivs[:,i] = piv
    return pivs

def lusolve_sweep(b,m,bs,LUdiag,\
                   Lsub,Usup,pivs,f,Lsweep=True,Usweep=True,fast_LU=True):
    
    ncols = f.shape[2]
    f     = f.reshape(b,m,bs,ncols)
    
    toc_tri = 0
    toc_mat = 0
    
    if (Lsweep):
        # solve Ly = f in a rightward sweep
        tic = time()
        f[:,0] = solveL(LUdiag[:,0],pivs[:,0],f[:,0],fast_LU=fast_LU)
        for i in range(1,m):
            tmp = f[:,i]  - Lsub[:,i-1] @ f[:,i-1]
            f[:,i] = solveL(LUdiag[:,i],pivs[:,i],tmp,fast_LU=fast_LU)
            
    if (Usweep):
        # solve Uu = y in a leftward sweep
        f[:,m-1] = solveU(LUdiag[:,m-1],f[:,m-1],fast_LU=fast_LU)
        for i in np.arange(m-2,-1,-1):
            tmp = f[:,i] - Usup[:,i] @ f[:,i+1]
            f[:,i] = solveU(LUdiag[:,i],tmp,fast_LU=fast_LU)
    return f.reshape(b,m*bs,ncols)

# solve f = A^T u
def lusolve_sweep_transpose(b,m,bs,LUdiag,\
                             Lsub,Usup,pivs,f, fast_LU=True):

    ncols = f.shape[2]
    f     = f.reshape(b,m,bs,ncols)
    
    f[:,0] = solveU(LUdiag[:,0],f[:,0],transpose=True,fast_LU=fast_LU)
    # solve U^T y = f in rightward sweep
    for i in range(1,m):
        tmp = f[:,i] - torch.transpose(Usup[:,i-1],-1,-2) @ f[:,i-1]
        f[:,i] = solveU( LUdiag[:,i], tmp, transpose=True, fast_LU=fast_LU )

    # solve L^T u = y in a leftward sweep
    f[:,m-1] = solveL(LUdiag[:,m-1],pivs[:,m-1],f[:,m-1],transpose=True, fast_LU = fast_LU)
    for i in np.arange(m-2,-1,-1):
        tmp = f[:,i] - torch.transpose(Lsub[:,i],-1,-2) @ f[:,i+1]
        f[:,i] = solveL(LUdiag[:,i],pivs[:,i],tmp,transpose=True, fast_LU=fast_LU)
    return f.reshape(b,m*bs,ncols)

############################ BLOCK TRIDIAGONAL CLASS #################################

def alloc_pivdata(b,m,bs):
    if (m < bs):
        piv_data = torch.zeros(b,1,bs,bs,device=torch.device('cpu'),requires_grad=False)
    else:
        nblocks = int(m/bs)
        piv_data = torch.zeros(b,nblocks+1,bs,bs,device=torch.device('cpu'),requires_grad=False)
    return piv_data

def unpack_pivs(b,m,bs,Tdata,device):
    piv_offset = m + 2*(m-1)                    
    pivs = torch.zeros((b,m,bs),device=device,requires_grad=False).int()
    if ( m < bs ):
        pivs = Tdata[:,piv_offset,:m].int()
    else:
        nblocks = int(m/bs)
        pivs[:,:nblocks*bs] = Tdata[:,piv_offset:piv_offset+nblocks].reshape(b,nblocks*bs,bs)
        rows_left = m - nblocks*bs
        pivs[:,nblocks*bs:] = Tdata[:,piv_offset+nblocks,:rows_left]
    return pivs

def pack_pivs(b,m,bs,Tdata,pivs):
    piv_offset = m + 2*(m-1)
    
    if (m < bs):
        Tdata[:,piv_offset,:m] = pivs
    else:
        nblocks = int(m/bs)
        Tdata[:,piv_offset:piv_offset + nblocks] = pivs[:,:nblocks*bs].reshape(b,nblocks,bs,bs)
        
        rows_left = m - nblocks*bs
        Tdata[:,piv_offset + nblocks,:rows_left] = pivs[:,nblocks*bs:]
    return Tdata

def get_result_chunksize(n0,n1,device):
    if (device == torch.device('cuda')):
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        f = r-a # in bytes
        f = np.min([f,3e9])
    else:
        f = 6e9 #6 GB in bytes
    chunk_max = int(f / (n0*n1*8)) # 8 bytes in 64 bits memory
    return int(chunk_max)

def get_solve_chunksize(m,bs,n1,device):
    if (device == torch.device('cuda')):
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        f = r-a # in bytes
    else:
        f = 2e9 # 1 GB in bytes
    chunk_max = int(f / (m*bs*n1*8)) # 8 bytes in 64 bits memory
    return chunk_max

def get_build_chunksize(b,m,bs,device):
    if (device == torch.device('cuda')):
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        f = r-a # in bytes
    else:
        f = 20e9 # 20 GB in bytes
    chunk_max = int(f / (8*3*m*bs**2)) # 8 bytes in 64 bits memory
    return np.max([chunk_max,5])

# block triagonal solver
class Block_Tridiagonal:
    def __init__(self,b,m,bs):
        self.b = b # number of batches
        self.m = m # number of blocks
        self.bs = bs # block size
        self.pivdata_nblocks = int(m/bs) + 1
        self.storage_blocks = m + 2*(m-1) + self.pivdata_nblocks
        self.nbytes = (self.b * self.storage_blocks * self.bs**2) * 8
        
    def build_sweep(self,Tdiag,Tsub,Tsup,fast_LU=True):
        b = self.b; m = self.m; bs = self.bs
        
        assert Tdiag.shape[1] == m;
        assert Tsub.shape[1] == m-1;
        assert Tsup.shape[1] == m-1;
            
        # transfer to device    
        pivdata = alloc_pivdata(b,m,bs).to(Tdiag.device)
        Tdata_cpu = torch.cat((Tdiag,Tsub,Tsup,pivdata),1)
        del Tdiag, Tsub, Tsup, pivdata
        self.Tdata = Tdata_cpu
        self.fast_LU=fast_LU
        self._build()
        
    def build_sweep_from_data(self,Tdata, fast_LU=True):
        b = self.b; m = self.m; bs = self.bs
        assert Tdata.shape[1] == m + 2*(m-1) + self.pivdata_nblocks
        self.Tdata = Tdata
        self.fast_LU = fast_LU
        self._build()
        
    def to_fastLU(self):
        b = self.b; m = self.m; bs = self.bs
        if (not self.fast_LU):
            
            LUstacked = self.Tdata[:,:self.m].reshape(b*m,bs,bs)
            LUstacked = calc_tri_inv(LUstacked).reshape(b,m,bs,bs)
            self.Tdata[:,:self.m] = LUstacked
            self.fast_LU = True
            
        
    def _build(self,verbose=False):
        
        b = self.b; m = self.m; bs = self.bs
        
        curr_device = self.Tdata.device
        
        if (torch.cuda.is_available()):
            tar_device = torch.device('cuda')
        else:
            tar_device = torch.device('cpu')
            
        if (torch.cuda.is_available()):
            tot_mem = torch.cuda.get_device_properties(0).total_memory
            Tdata_mem = self.Tdata.nelement() * self.Tdata.element_size() 
            if ((b == 1) & (Tdata_mem > tot_mem  - 10e9)):
                tar_device = torch.device('cpu')
                print("warning: Tdata too big for GPU %2.5f GB" % (Tdata_mem/1e9))
            elif (b == 1):
                print("Tdata for b = 1 fits on GPU %2.5f GB" % (Tdata_mem/1e9))
        
        chunk = get_build_chunksize(b,m,bs,tar_device)
        curr_b = 0
        
        while (curr_b < b):
            b_end = np.min([curr_b+chunk,b])
            Tdata_gpu = self.Tdata[curr_b:b_end].to(tar_device)
            if (verbose):
                print("--transfered to gpu",curr_b,b_end)

            sub_offset = m; sup_offset = m + m-1; piv_offset = m + 2*(m-1)

            pivs = build_sweep(b_end-curr_b,m,bs,Tdata_gpu[:,:sub_offset], \
                               Tdata_gpu[:,sub_offset:sup_offset], Tdata_gpu[:,sup_offset:piv_offset],\
                               fast_LU=self.fast_LU)

            Tdata_gpu = pack_pivs(b_end-curr_b,m,bs,Tdata_gpu,pivs)
            if (verbose):
                print("--returning to cpu")
            self.Tdata[curr_b:b_end] = Tdata_gpu.to(curr_device)
            del Tdata_gpu
            chunk = get_build_chunksize(b,m,bs,tar_device)
            curr_b = b_end
                                
    def lusolve_sweep(self,f,b_start=None, b_end=None,\
                      transpose=False):
        b = self.b; m = self.m; bs = self.bs
        
        if (b_start is None):
            b_start = 0
        if (b_end is None):
            b_end = b
        
        tic = time()
        Tdata  = self.Tdata[b_start:b_end].to(f.device)
        sub_offset = m; sup_offset = m + m-1; piv_offset = m + 2*(m-1)
        LUdiag = Tdata[:,:sub_offset]
        Lsub   = Tdata[:,sub_offset:sup_offset]
        Usup   = Tdata[:,sup_offset:piv_offset]
        pivs   = unpack_pivs(b_end-b_start,m,bs,Tdata,f.device)
        
        args = b_end-b_start,m,bs,LUdiag,Lsub,Usup,pivs
        
        if (not transpose):
            return lusolve_sweep(*args,f,fast_LU=self.fast_LU)
        else:
            return lusolve_sweep_transpose(*args,f,fast_LU=self.fast_LU)
        
    def allocrhs_helper(self,b_start,b_end,\
                       n0,n1,get_locmem,get_T,funcA,funcB,funcC,args,device):
        m = self.m; bs = self.bs
        
        loc_mem = get_locmem(*args,b_start,b_end,device)
        
        result = funcA(loc_mem,n0,n1,0,b_end-b_start,device)
        
        Tdiag,Tsub,Tsup = get_T(*loc_mem)
        pivs = build_sweep(b_end-b_start,m,bs,Tdiag,Tsub,Tsup)
        
        b1 = 0
        chunk_size = np.max([get_solve_chunksize(m,bs,n1,device),2])
        # partition into chunks
        while (b1 < b_end-b_start):
            b2 = np.min([b_end-b_start, b1 + chunk_size])
            
            tmp = funcC(loc_mem,n0,n1,b1,b2,device)
            
            tmp = lusolve_sweep(b2-b1,m,bs,Tdiag[b1:b2],Tsub[b1:b2],Tsup[b1:b2],\
                                  pivs[b1:b2],tmp)
            
            result[b1:b2] += funcB(loc_mem,n0,n1,b1,b2,tmp,device)
            b1 = b2
            del tmp; torch.cuda.empty_cache()
            chunk_size = np.max([get_solve_chunksize(m,bs,n1,device),2])
        return result.cpu(),Tdiag.cpu(), Tsub.cpu(), Tsup.cpu(), pivs.cpu()

    # calculate result = A - B * inv(T) * C
    def schur_complement_alloc(self,n0,n1,\
                               get_locmem,get_T,
                               funcA,funcB,funcC,args):
        
        b = self.b; m = self.m; bs = self.bs
        
        if (torch.cuda.is_available()):
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
            
        chunk_size = np.max([get_result_chunksize(n0,n1,device),2])
        result = torch.zeros(self.b,n0,n1)
        
        LUdiag = torch.zeros(b,m,bs,bs); Lsub = torch.zeros(b,m-1,bs,bs)
        Usup   = torch.zeros(b,m-1,bs,bs); pivs = torch.zeros(b,m,bs).int()
        
        helper_args = n0,n1,get_locmem,get_T,funcA,funcB,funcC,args
            
        b1 = 0
        while (b1 < self.b):
            b2 = np.min([self.b,b1 + chunk_size])
            
            tmp = self.allocrhs_helper(b1,b2,*helper_args,device)
            result[b1:b2] += tmp[0]
            LUdiag[b1:b2] += tmp[1]
            Lsub[b1:b2]   += tmp[2]
            Usup[b1:b2]   += tmp[3]
            pivs[b1:b2]   += tmp[4]
            
            chunk_size = np.max([get_result_chunksize(n0,n1,device),2])
            b1 = b2
            
        pivdata = alloc_pivdata(b,m,bs)
        Tdata = torch.cat((LUdiag,Lsub,Usup,pivdata),1)
        Tdata = pack_pivs(b,m,bs,Tdata,pivs)
        self.Tdata = Tdata
        
        return result
    
    def matvec(self,Tdiag,Tsub,Tsup,u):
        u_blocked = u.reshape(self.b,self.m,self.bs,u.shape[2])
        f = torch.zeros(u_blocked.shape,device=u.device)
        for i in range(self.m):
            f[:,i] += Tdiag[:,i] @ u_blocked[:,i]
        for i in range(self.m-1):
            f[:,i+1] += Tsub[:,i] @ u_blocked[:,i]
        for i in range(self.m-1):
            f[:,i] += Tsup[:,i] @ u_blocked[:,i+1]
        return f.reshape(self.b,self.m*self.bs,u.shape[2])

    def matvec_transpose(self,Tdiag,Tsub,Tsup,u):
        u_blocked = u.reshape(self.b,self.m,self.bs,u.shape[2])
        f = torch.zeros(u_blocked.shape,device=u.device)
        for i in range(self.m):
            f[:,i] += torch.transpose(Tdiag[:,i],-2,-1) @ u_blocked[:,i]
        for i in range(self.m-1):
            f[:,i] += torch.transpose(Tsub[:,i],-2,-1) @ u_blocked[:,i+1]
        for i in range(self.m-1):
            f[:,i+1] += torch.transpose(Tsup[:,i],-2,-1) @ u_blocked[:,i]
        return f.reshape(self.b,self.m*self.bs,u.shape[2])
