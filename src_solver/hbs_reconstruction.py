import torch
import numpy as np
from time import time
torch.set_default_dtype(torch.double)

def copy_params_hbs(hbs,b):
    hbs_new = HBS_matrix(b,hbs.n,hbs.k)
    
    uv_shape = torch.tensor(hbs.UVtensor.shape).long()
    d_shape  = torch.tensor(hbs.Dtensor.shape).long()
    uv_shape[0] = b; d_shape[0] = b
    
    hbs_new.UVtensor = torch.zeros(*uv_shape)
    hbs_new.Dtensor  = torch.zeros(*d_shape)
    
    hbs_new.L = hbs.L; hbs_new.m = hbs.m; hbs_new.bs = hbs.bs; 
    hbs_new.n_pad = hbs.n_pad
    return hbs_new

def copy_data_hbs(hbs,b_start,b_end):
    hbs_new = copy_params_hbs(hbs,b_end-b_start)
    
    hbs_new.UVtensor = hbs.UVtensor[b_start:b_end].clone()
    hbs_new.Dtensor  = hbs.Dtensor[b_start:b_end].clone()
    return hbs_new

def get_nearest_div(n,bs):
    while ((np.mod(n,bs) > 0) and bs < n):
        bs += 1
    return bs

class Matrix:
    def __init__(self,b,n,k,diag=1):
        self.U = torch.linalg.qr(torch.rand(b,n,k))[0]
        self.V = torch.linalg.qr(torch.rand(b,n,k))[0]
        self.b, self.n, self.k = b, n,k
        self.diag = torch.rand(n,1); self.diag[:int(n/2)] = 10
        if (diag == 0):
            self.diag = torch.zeros(self.diag.shape)
        
    def matvec(self,v):
        v_hat = v.clone()
        return self.U @ (torch.transpose(self.V,-1,-2) @ v_hat) + self.diag * v_hat;
    
    def matvec_h(self,v):
        v_hat = v.clone()
        return self.V @ (torch.transpose(self.U,-1,-2) @ v_hat) + self.diag * v_hat;
    
class HBS_matrix:
    def __init__(self,b,n,k,n_pad=0):
        self.b = b
        self.n, self.k   = n,k
        #bs_est = int(n**(1/3) * k**(2/3)); 
        bs_est = 2*k
        self.bs    = get_nearest_div(n+n_pad,bs_est)
        self.m     = int((n+n_pad)/self.bs)
        self.n_pad = n_pad
        
        nearest_pow = np.log2(self.m)
        if (np.abs(int(nearest_pow)- nearest_pow) > 1e-14):
            nearest_pow = int(nearest_pow) + 1
            self.n_pad += ( int(2**nearest_pow) - self.m ) * self.bs
            self.m = int(2**nearest_pow);
            
    def nbytes(self):
        nbytes = ( torch.prod(torch.tensor(self.UVtensor.shape).int()) + \
        torch.prod(torch.tensor(self.Dtensor.shape).int()) ) * 8
        return nbytes.item()
        
    def reconstruct(self, Omega, Psi, Y, Z):
        
        b = self.b; m = self.m; bs = self.bs; k = self.k
        device = Omega.device
        
        if (self.n_pad > 0):
            # pad with identity matrix
            Omega_pad = torch.rand(Omega.shape[0],self.n_pad,bs+k,device=device)
            Psi_pad   = torch.rand(Psi.shape[0],  self.n_pad,bs+k,device=device)
           
            Omega = torch.concat((Omega,Omega_pad),axis=1)
            Psi = torch.concat((Psi,Psi_pad),axis=1)
            Y = torch.concat((Y,Omega_pad.repeat(b,1,1)),axis=1)
            Z = torch.concat((Z,Psi_pad.repeat(b,1,1)),axis=1)
            
        #print("reconstruct with  (b,m,bs,k)", self.b,self.m,self.bs,self.k)
            
        tic = time()
        decomp = _reconstruct(Omega,Psi,Y,Z,b,m,bs,k)
        toc = time() - tic
            
        decomp_list,D_root = _correct_reconstruction(b,m,bs,k,*decomp,device=device)
        self.L = len(decomp_list)
        UVtensor,Dtensor   = _store_tensors(b,m,bs,k,decomp_list,D_root,device=device)
        self.UVtensor = UVtensor.cpu(); self.Dtensor = Dtensor.cpu()
        
    def todense_hbsow(self,device,b_start=None,b_end=None):
        b_start = 0 if b_start is None else b_start
        b_end   = self.b if b_end is None else b_end
        
        result = torch.zeros(b_end-b_start,self.n,self.n,device=device)
        
        tic = time() 
        UVtensor = self.UVtensor[b_start:b_end].to(device)
        Dtensor  = self.Dtensor[b_start:b_end].to(device)
        toc = time() - tic; 
        
        for b_ind in range(b_end-b_start):
            tic = time()
            tmp = _todense(UVtensor[b_ind],Dtensor[b_ind],\
                          self.L,int(self.n+self.n_pad)/self.bs,self.bs,self.k)
            toc = time() - tic ;
            result[b_ind] = tmp[:self.n,:self.n]
            del tmp
        tic = time()
        result = result.cpu()
        toc = time() - tic; 
        return result
    
    def todense(self,device=torch.device('cpu'),b_start=None,b_end=None,verbose=False):
        b_start = 0 if b_start is None else b_start
        b_end   = self.b if b_end is None else b_end
        
        result = torch.zeros(b_end-b_start,self.n,self.n,device=device)
        UVtensor = self.UVtensor[b_start:b_end].to(device)
        Dtensor  = self.Dtensor[b_start:b_end].to(device)
        
        for b in range(b_end-b_start):
            q = torch.eye(self.n + self.n_pad, self.n,device=device).unsqueeze(0)
            tmp = _matvec(UVtensor[b:b+1],Dtensor[b:b+1],\
                      self.L,1,self.m,self.bs,self.k,q)
            result[b] = tmp[:,:self.n,:self.n].unsqueeze(0)
        return result
    
    def matvec(self,q,b_start=None,b_end=None,transpose=False,verbose=False):
        b_start = 0 if b_start is None else b_start
        b_end   = self.b if b_end is None else b_end
        
        if (self.n_pad > 0):
            # pad with zeros
            q = torch.concat((q,torch.zeros(q.shape[0],self.n_pad,\
                                            q.shape[2],device=q.device)),axis=1)
        
        tmp = _matvec(self.UVtensor[b_start:b_end].to(q.device),\
                      self.Dtensor[b_start:b_end].to(q.device),\
                      self.L,b_end-b_start,self.m,self.bs,self.k,\
                      q,transpose=transpose,verbose=verbose)
        return tmp[:,:self.n]
    
    def calc_inverse(self):
                  
        self.EFtensor,self.Gtensor = _calc_inverse(self.L,self.m,self.bs,self.k,\
                                 self.UVtensor,self.Dtensor)
        self.UVtensor = 0; self.Dtensor = 0
        
    def apply_inverse(self,f,b_start=None,b_end=None,transpose=False):
        b_start = 0 if b_start is None else b_start
        b_end   = self.b if b_end is None else b_end
        
        if (self.n_pad > 0):
            # pad with zeros
            f = torch.concat((f,torch.zeros(f.shape[0],self.n_pad,\
                                            f.shape[2],device=f.device)),axis=1)
        
        tmp = _matvec(self.EFtensor[b_start:b_end].to(f.device),\
                      self.Gtensor[b_start:b_end].to(f.device),
                       self.L,b_end-b_start,self.m,self.bs,self.k,f,\
                      transpose=transpose)
        return tmp[:,:self.n]
        
################# UTILITIES FOR INVERSE      #################

def _calc_Dhat(U_blocked,V_blocked,D_blocked):
    Dinv    = torch.linalg.inv(D_blocked)
    #print("D_blocked cond",torch.linalg.cond(D_blocked))
    Dhat    = torch.transpose(V_blocked,-1,-2) @ (Dinv @ U_blocked)
    Dhat = torch.linalg.inv(Dhat)
    #print("Dhat cond",torch.linalg.cond(Dhat))
    return Dinv, Dhat

def _add_Dhat(D_blocked,D_hat):
    
    l = int(D_blocked.shape[-1]/2)
    for j in range(D_blocked.shape[1]):
        D_blocked[:,j,:l,:l] += D_hat[:,2*j]
        D_blocked[:,j,l:,l:] += D_hat[:,2*j+1]
    return D_blocked

def _calc_inverse(L,m,bs,k,UVtensor,Dtensor):
    
    EFtensor = torch.zeros(UVtensor.shape,device=UVtensor.device)
    Gtensor  = torch.zeros(Dtensor.shape,device=Dtensor.device)

    for l in range(L):
        b_start = int(2**(L+1) - 2**(L+1-l))
        b_end   = int(2**(L-l)) + b_start
        
        if (l == 0):
            U_blocked = UVtensor[:,b_start:b_end,:,:k]
            V_blocked = UVtensor[:,b_start:b_end,:,k:]
            D_blocked = Dtensor[:,b_start:b_end]
        else:
            U_blocked = UVtensor[:,b_start:b_end,:2*k,:k]
            V_blocked = UVtensor[:,b_start:b_end,:2*k,k:]
            D_blocked = Dtensor[:,b_start:b_end,:2*k,:2*k]
        
        if (l > 0):
            D_blocked = _add_Dhat(D_blocked,Dhat)
        
        Dinv,Dhat = _calc_Dhat(U_blocked,V_blocked,D_blocked)
        
        U_solve = Dinv @ U_blocked
        V_solve = torch.transpose(V_blocked,-1,-2) @ Dinv
        
        E =  U_solve @ Dhat
        F = torch.transpose(Dhat @ V_solve,-1,-2)
        G = Dinv - U_solve @ torch.transpose(F,-1,-2)
        if (l == 0):
            EFtensor[:,b_start:b_end,:,:k] = E
            EFtensor[:,b_start:b_end,:,k:] = F
            Gtensor[:,b_start:b_end] = G
        else:
            EFtensor[:,b_start:b_end,:2*k,:k] = E
            EFtensor[:,b_start:b_end,:2*k,k:] = F
            Gtensor[:,b_start:b_end,:2*k,:2*k] = G
        
    Droot = Dtensor[:,-1,:2*k,:2*k]
    for j in range(U_blocked.shape[0]):
        Droot[j] += torch.block_diag(*Dhat[j])
    #print("conditioning Droot",torch.linalg.cond(Droot))
    Groot = torch.linalg.inv(Droot)
    Gtensor[:,-1,:2*k,:2*k] = Groot
    
    return EFtensor.cpu(),Gtensor.cpu()

        
################# UTILITIES FOR MATVEC       #################

def _matvec(UVtensor,Dtensor,L,b,m,bs,k,q,transpose=False,verbose=False):
    q_blocked = q.reshape(q.shape[0],m,bs,q.shape[-1]).contiguous()
    
    qhat_struct = torch.zeros(b,int(2**(L+1)-2),k,q.shape[-1],device=q.device)
    
    # upward pass
    if verbose:
        print("upward pass",0)
    if not transpose:
        q_hat = torch.transpose(UVtensor[:,:m,:,k:],-1,-2) @ q_blocked
    else:
        q_hat = torch.transpose(UVtensor[:,:m,:,:k],-1,-2) @ q_blocked
    m_curr = m
    qhat_struct[:,:m] = q_hat
    
    for l in range(1,L):
        if verbose:
            print("upward pass",l)
        q_hat_reshape = q_hat.reshape(b,int(m_curr/2),2*k,q.shape[2]).contiguous()
        b_start = int(2**(L+1) - 2**(L+1-l))
        b_end   = int(2**(L-l)) + b_start
        if not transpose:
            q_hat = torch.transpose(UVtensor[:,b_start:b_end,:2*k,k:],-1,-2) @ q_hat_reshape
        else:
            q_hat = torch.transpose(UVtensor[:,b_start:b_end,:2*k,:k],-1,-2) @ q_hat_reshape
        m_curr = int(m_curr/2)
        qhat_struct[:,b_start:b_end] = q_hat
    
    # multiply by top-level root node
    q_hat_reshape = q_hat.reshape(b,m_curr * k, q.shape[2]).contiguous()
    if not transpose:
        u_hat = Dtensor[:,-1,:2*k,:2*k] @ q_hat_reshape
    else:
        u_hat = torch.transpose(Dtensor[:,-1,:2*k,:2*k],-1,-2) @ q_hat_reshape
    u_hat = u_hat.reshape(b,m_curr,k,q.shape[2])
    
    # downward pass
    for l in range(L-1,0,-1):
        if verbose:
            print("downward pass",l)
        b_start = int(2**(L+1) - 2**(L+2-l))
        b_end   = int(2**(L+1-l)) + b_start
        qhat_struct = qhat_struct[:,:b_end].clone()
        q_hat = qhat_struct[:,b_start:b_end].reshape(b,m_curr, 2 * k, q.shape[2]).contiguous()

        b_start = int(2**(L+1) - 2**(L+1-l))
        b_end   = int(2**(L-l)) + b_start
        if not transpose:
            u_hat  = UVtensor[:,b_start:b_end,:2*k,:k] @ u_hat
            u_hat += Dtensor[:,b_start:b_end,:2*k,:2*k] @ q_hat
        else:
            u_hat  = UVtensor[:,b_start:b_end,:2*k,k:] @ u_hat
            u_hat += torch.transpose(Dtensor[:,b_start:b_end,:2*k,:2*k],-1,-2) @ q_hat
        u_hat = u_hat.reshape(b,m_curr * 2, k, q.shape[2])
        m_curr *= 2
        
    # leaf level multiplication
    if verbose:
        print("downward pass",0)
    if not transpose:
        u_blocked = Dtensor[:,:m] @ q_blocked
        u_blocked += UVtensor[:,:m,:,:k] @ u_hat
    else:
        u_blocked  = UVtensor[:,:m,:,k:] @ u_hat
        u_blocked += torch.transpose(Dtensor[:,:m],-1,-2) @ q_blocked
    return u_blocked.reshape(b,m*bs,q.shape[2])

# only works for b == 1
def _todense(UVtensor,Dtensor,L,m,bs,k):
    assert (UVtensor.dim() == 3) and (Dtensor.dim() == 3)
    # top-level root node
    result = Dtensor[-1,:2*k,:2*k]
    
    # at each level, multiply by left and right diagonal bases
    for l in range(L,0,-1):
        b_start = int(2**(L+1) - 2**(L+2-l))
        b_end   = int(2**(L+1-l)) + b_start
        left_bases = UVtensor[b_start:b_end,:2*k,:k]
        right_bases = torch.transpose(UVtensor[b_start:b_end,:2*k,k:],-1,-2)
        diag_blocks = Dtensor[b_start:b_end,:2*k,:2*k]
        
        tic = time() 
        U      = torch.block_diag(*left_bases)
        Vtrans = torch.block_diag(*right_bases)
        D      = torch.block_diag(*diag_blocks)
        toc_blkdiag = time() - tic; 
        
        tic = time()
        result = (U @ result) @ Vtrans + D
        toc_mm = time() - tic; #print("level ops",toc_blkdiag,toc_mm)
    return result

################# UTILITIES  TO ASSESS ERROR  #################

def get_diag_err(decomp,Afull,m,bs):
    
    U_blocked,V_blocked,D_blocked = decomp
    
    U_diag = torch.block_diag(*U_blocked)
    V_diag = torch.block_diag(*V_blocked)
    
    U_proj = U_diag @ U_diag.T
    V_proj = V_diag @ V_diag.T

    Atmp = Afull - (U_proj @ Afull) @ V_proj
    A_diag = Atmp.reshape(m,bs,m,bs)[range(m),:,range(m),:]
    err = D_blocked - A_diag

    Aproj = (U_diag.T @ Afull) @ V_diag
    return err,Aproj
        
################# RECONSTRUCTION FROM MATVECS #################

def _correct_reconstruction(b,mtot,bs,k,decomp_list,D_root,device):
    
    # adjust D_root to improve conditioning
    L = len(decomp_list)
    U_blocked,V_blocked,D_blocked = decomp_list[L-1]

    tmp = torch.zeros(b,2,k,k,device=device)
    tmp[:,0] = D_root[:,:k,:k]; D_root[:,:k,:k] = torch.zeros(b,k,k,device=device)
    tmp[:,1] = D_root[:,k:,k:]; D_root[:,k:,k:] = torch.zeros(b,k,k,device=device)

    D_blocked += U_blocked @ (tmp @ torch.transpose(V_blocked,-1,-2))

    decomp_list[L-1] = U_blocked,V_blocked,D_blocked

    # adjust D_blocked for subsequent levels to improve conditioning
    for l in range(L-1,0,-1):
        U_blocked, V_blocked, D_blocked = decomp_list[l]

        nblocks = D_blocked.shape[1]
        tmp = torch.zeros(b,2*nblocks,k,k,device=device)

        tmp[:,range(0,2*nblocks,2)] = D_blocked[:,range(nblocks),:k,:k]
        D_blocked[:,range(nblocks),:k,:k] = torch.zeros(b,nblocks,k,k,device=device)
        tmp[:,range(1,2*nblocks,2)] = D_blocked[:,range(nblocks),k:,k:]
        D_blocked[:,range(nblocks),k:,k:] = torch.zeros(b,nblocks,k,k,device=device)


        U_blocked,V_blocked,D_blocked = decomp_list[l-1]
        D_blocked += U_blocked @ (tmp @ torch.transpose(V_blocked,-1,-2))
        decomp_list[l-1] = U_blocked,V_blocked, D_blocked
    return decomp_list,D_root

def _store_tensors(b,mtot,bs,k,decomp_list,D_root,device):
    L = len(decomp_list)
    nblocks = int(2**(L+1) - 2)
    UVtensor = torch.zeros(b,nblocks,bs,2*k,device=device)
    Dtensor  = torch.zeros(b,nblocks+1,bs,bs,device=device)
    for l in range(L):
        U,V,D = decomp_list[l]

        b_start = int(2**(L+1) - 2**(L+1-l))
        b_end   = int(2**(L-l)) + b_start
        if (l == 0):
            UVtensor[:,b_start:b_end,:,:k] = U
            UVtensor[:,b_start:b_end,:,k:] = V
        else:
            UVtensor[:,b_start:b_end,:2*k,:k] = U
            UVtensor[:,b_start:b_end,:2*k,k:] = V
        if (l == 0):
            Dtensor[:,b_start:b_end] = D
        else:
            Dtensor[:,b_start:b_end,:2*k,:2*k] = D

    Dtensor[:,-1,:2*k,:2*k] = D_root
    return UVtensor,Dtensor

# B @ pinv(A)
def pinv_right(B,A):
    tmp = torch.linalg.lstsq(torch.transpose(A,-1,-2),\
                           torch.transpose(B,-1,-2)).solution
    sol = torch.transpose(tmp,-1,-2)
    return sol
    
def _reconstruct(Omega,Psi,Y,Z,b,m,bs,k):
    
    decomp_list = []; m_curr = m;

    while(m_curr > 0):
        if (m_curr == m):
            # leaf node
            Omega_blocked = Omega.reshape(Omega.shape[0],m,bs,bs+k)
            Psi_blocked   = Psi.reshape(Psi.shape[0],m,bs,bs+k)
            Y_blocked = Y.reshape(b,m,bs,bs+k)
            Z_blocked = Z.reshape(b,m,bs,bs+k)
        else:
            tic = time()
            sample_args = Omega_blocked,Psi_blocked,Y_blocked,Z_blocked
            Omega_proj, Y_proj, Psi_proj, Z_proj = project_samples(*sample_args,*decomp)
            toc = time() - tic
            #print("project samples",toc)
            
            Omega_blocked = Omega_proj[...,:3*k].reshape(b,m_curr,2*k,3*k)
            Y_blocked     = Y_proj[...,:3*k].reshape(b,m_curr,2*k,3*k)
            Psi_blocked   = Psi_proj[...,:3*k].reshape(b,m_curr,2*k,3*k)
            Z_blocked     = Z_proj[...,:3*k].reshape(b,m_curr,2*k,3*k)

        if (m_curr  == 1):
            nsamples = Omega_blocked.shape[-1]
            Omega_root = Omega_blocked.reshape(b,m_curr*2*k,nsamples)
            Y_root     = Y_blocked.reshape(b,m_curr*2*k,nsamples)
            
            # Yroot @ pinv(Omega_root)
            D_root     = pinv_right(Y_root,Omega_root)
            break
        else:
            sample_args = Omega_blocked,Psi_blocked,Y_blocked,Z_blocked
            tic = time()
            decomp = get_decomp_blocks(*sample_args,k)
            #print("get decomp blocks",time() - tic)
            m_curr = int(m_curr/2) if np.mod(m_curr,2) == 0 else int(m_curr/2) + 1
            decomp_list += [decomp]
    return decomp_list,D_root

def calc_null_rowspace(A_blocked,k):
    A_blocked_H = torch.transpose(A_blocked,-1,-2)
    m = A_blocked.shape[-1] - k;
    Q,R = torch.linalg.qr(A_blocked_H,mode='complete')
    Null = Q[...,m:]
    return Null

def get_decomp_blocks(Omega_blocked,Psi_blocked,Y_blocked,Z_blocked,k):
    
    tic = time()
    P_null = calc_null_rowspace(Omega_blocked,k)
    Q_null   = calc_null_rowspace(Psi_blocked,k)
    toc = time() - tic
    
    # calculate U_blocked, V_blocked
    tic = time()
    U_blocked = torch.linalg.qr(Y_blocked @ P_null)[0]
    U_blocked = U_blocked[...,:k]
    
    V_blocked = torch.linalg.qr(Z_blocked @ Q_null)[0]
    V_blocked = V_blocked[...,:k]
    toc = time() - tic

    tic = time()
    tmp_row = pinv_right(Y_blocked,Omega_blocked)
    tmp_col = pinv_right(Z_blocked,Psi_blocked)
    
    
    # calculate D_blocked
    D_blocked  = tmp_row - U_blocked @ (torch.transpose(U_blocked,-1,-2) @ tmp_row)
    tmp        = tmp_col - V_blocked @ (torch.transpose(V_blocked,-1,-2) @ tmp_col)
    D_blocked += U_blocked @ (torch.transpose(U_blocked,-1,-2) @ torch.transpose(tmp,-1,-2))
    
    return U_blocked,V_blocked,D_blocked

def project_samples(Omega_blocked,Psi_blocked,Y_blocked,Z_blocked,\
                    U_blocked,V_blocked,D_blocked):

    Omega_proj = torch.transpose(V_blocked,-1,-2) @ Omega_blocked
    Y_proj     = torch.transpose(U_blocked,-1,-2) @ (Y_blocked - D_blocked @ Omega_blocked)

    Psi_proj   = torch.transpose(U_blocked,-1,-2) @ Psi_blocked
    Z_proj     = torch.transpose(V_blocked,-1,-2) @ (Z_blocked - torch.transpose(D_blocked,-1,-2) @ Psi_blocked)
    return Omega_proj,Y_proj, Psi_proj, Z_proj
