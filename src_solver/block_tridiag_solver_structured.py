import torch
torch.set_default_dtype(torch.double)
import numpy as np
from time import time

import src_solver.block_tridiag_solver_dense as block_tridiag_solver_dense
from   src_solver.hbs_reconstruction import copy_data_hbs

class HBS_BlockTridiagonal:
    def __init__(self,b,n,cyclic=False):
        assert (b < n)
        self.b = b
        self.n = n
        self.cyclic = cyclic
        
    def build_simple(self,hbs_diag0,hbs_diag1,hbs_sub,hbs_sup):
        self.build_mode = 'simple'
        b = self.b; n = self.n
        
        Tdiag  = hbs_diag0.todense()
        Tdiag += hbs_diag1.todense()
        Tdiag  = Tdiag.unsqueeze(0)

        Tsub = hbs_sub.todense().unsqueeze(0)
        Tsup = hbs_sup.todense().unsqueeze(0)

        Tridiag = block_tridiag.Block_Tridiagonal(1,b,n)
        Tridiag.build_sweep(Tdiag,Tsub,Tsup,\
                            fast_LU=False)
        self.solve_op = Tridiag
        
        self.nbytes = Tridiag.nbytes
        
    def build_lowmem_helper(self,b_start,b_end,hbs_diag0,hbs_diag1,hbs_sub,hbs_sup,\
                            device,LU_saved=0):
        
        Schur_saved = torch.zeros(b_end-b_start,self.n,self.n,device=device)
        perm_trans_saved = torch.zeros(b_end-b_start,self.n,device=device).long()
        
        for j in range(b_start,b_end):
            
            S  = hbs_diag0.todense(b_start=j,b_end=j+1,device=device).squeeze(0)
            S += hbs_diag1.todense(b_start=j,b_end=j+1,device=device).squeeze(0)
        
            if (j == 0):
                LU_saved = torch.linalg.lu_factor(S)
                P,L,U = torch.lu_unpack(*LU_saved); del L; del U
                
                Schur_saved[j-b_start]      = LU_saved[0]
                perm_trans_saved[j-b_start] = torch.transpose(P,-1,-2) @ torch.arange(self.n,device=device).double()
                del P
            else:
                sup = hbs_sup.todense(b_start=j-1,b_end=j,device=device).squeeze(0)
                sub = hbs_sub.todense(b_start=j-1,b_end=j,device=device).squeeze(0)

                tic = time()
                try:
                    S -= sub @ torch.linalg.lu_solve(*LU_saved,sup)
                except AttributeError:
                    S -= sub @ torch.lu_solve(sup,*LU_saved)
                toc_lusolve = time() - tic
                del sub; del sup
                
                tic = time()
                LU_saved = torch.linalg.lu_factor(S)
                toc_lu = time() - tic
                del S
                
                #print("LU solve, build",j,toc_lusolve,toc_lu)
                Schur_saved[j-b_start] = LU_saved[0]
                P,L,U = torch.lu_unpack(*LU_saved); del L; del U
                
                perm_trans_saved[j-b_start] = torch.transpose(P,-1,-2) @ torch.arange(self.n,device=device).double()
                del P

        return Schur_saved,perm_trans_saved,LU_saved
    
    def __build_lowmem(self,hbs_diag0,hbs_diag1,hbs_sub,hbs_sup,device):
        
        self.build_mode = 'lowmem'
        
        b = self.b; n = self.n
        
        hbs_data = hbs_diag0,hbs_diag1,hbs_sub,hbs_sup
        
        if (device == torch.device('cpu')):
            Schur_data,perm_tmp,_ = self.build_lowmem_helper(0,b,*hbs_data,device)
            Schur_saved = Schur_data[:b]
            perm_trans_saved  = perm_tmp.long()
        else:
            Schur_saved = torch.zeros(b,n,n)
            perm_trans_saved  = torch.zeros(b,n).long()
            
            # calculate maximum chunk size
            max_mem = 3e9
            max_chunk = int(max_mem / (n**2 * 8));
            max_chunk = np.min([max_chunk,b])
            #print("--- chunk size for lowmem build",max_chunk,"max mem (GB)",max_mem / 1e9)
            
            b_curr = 0; LU_saved = 0
            while (b_curr < b):
                b_end = np.min([b_curr + max_chunk,b])
                
                Schur_tmp,perm_tmp,LU_saved = self.build_lowmem_helper(b_curr,b_end,\
                                                              *hbs_data,device,LU_saved)
                
                Schur_saved[b_curr:b_end] = Schur_tmp.cpu()
                perm_trans_saved[b_curr:b_end]  = perm_tmp.cpu()
                b_curr = b_end
        del hbs_diag0; del hbs_diag1
        
        nbytes = (b*n**2 + b*n) * 8
        nbytes += hbs_sup.nbytes() + hbs_sub.nbytes()
        self.nbytes   = nbytes
        self.solve_op = Schur_saved,perm_trans_saved, hbs_sub, hbs_sup
            
        
    def build_lowmem(self,hbs_diag0,hbs_diag1,hbs_sub,hbs_sup,device):
        b = self.b; n = self.n
        if (not self.cyclic):
            self.__build_lowmem(hbs_diag0,hbs_diag1,hbs_sub,hbs_sup,device)
            
        else:
            hbs_diag0_prime = copy_data_hbs(hbs_diag0,1,b)
            hbs_diag1_prime = copy_data_hbs(hbs_diag1,1,b)

            hbs_sub_prime   = copy_data_hbs(hbs_sub,2,b)
            hbs_sup_prime   = copy_data_hbs(hbs_sup,1,b-1)

            HBT = HBS_BlockTridiagonal(b-1,n,cyclic=False)

            HBT.build_lowmem(hbs_diag0_prime,hbs_diag1_prime,hbs_sub_prime,hbs_sup_prime,device)
            self.HBT_prime = HBT

            V_tmp = torch.zeros((b-1)*n,n)
            V_tmp[:n]       = - hbs_sub.todense(b_start=1,b_end=2) 
            V_tmp[(b-2)*n:] = -hbs_sup.todense(b_start=b-1,b_end=b)
            V_tmp = HBT.solve_lowmem(V_tmp).reshape(b-1,n,n)

            X1_block  = hbs_diag0.todense(b_start=0,b_end=1).squeeze(0)
            X1_block += hbs_diag1.todense(b_start=0,b_end=1).squeeze(0)
            X1_block += hbs_sup.todense(b_start=0,b_end=1).squeeze(0) @ V_tmp[0]
            X1_block += hbs_sub.todense(b_start=0,b_end=1).squeeze(0) @ V_tmp[-1]

            LU,piv = torch.linalg.lu_factor(X1_block)
            self.acyclic_solve_data = V_tmp,LU,piv,\
            copy_data_hbs(hbs_sub,0,1),copy_data_hbs(hbs_sup,0,1)
            
            self.nbytes    = self.HBT_prime.nbytes
            self.nbytes   += b*n*n * 8
            self.build_mode = 'lowmem'
            
            #self.hbs_diag0 = hbs_diag0; self.hbs_diag1 = hbs_diag1
            #self.hbs_sub   = hbs_sub;   self.hbs_sup   = hbs_sup
            
            
    def __solve_lowmem(self,f):
        b = self.b; n = self.n
        Schur_saved,perm_trans_saved,hbs_sub,hbs_sup = self.solve_op
        
        nrhs = f.shape[-1]
        f = f.reshape(b,n,nrhs)
        
        ### forward sweep
        f[0] = torch.linalg.solve_triangular(Schur_saved[0],f[0,perm_trans_saved[0]],\
                                             unitriangular=True,upper=False)
        for j in range(b-1):
            tmp = torch.linalg.solve_triangular(Schur_saved[j],f[j],\
                                          unitriangular=False,upper=True)
            f[j+1] -= hbs_sub.matvec(tmp.unsqueeze(0),\
                                    b_start=j,b_end=j+1).squeeze(0)
            del tmp
            f[j+1] = torch.linalg.solve_triangular(Schur_saved[j+1],f[j+1,perm_trans_saved[j+1]],\
                                          unitriangular=True,upper=False)

        ### backward sweep
        f[b-1] = torch.linalg.solve_triangular(Schur_saved[b-1],f[b-1],\
                                      unitriangular=False,upper=True)
        for j in range(b-2,-1,-1):
            tmp = hbs_sup.matvec(f[j+1].unsqueeze(0),b_start=j,b_end=j+1).squeeze(0)
            f[j] -= torch.linalg.solve_triangular(Schur_saved[j], tmp[perm_trans_saved[j]],\
                                                  unitriangular=True,upper=False)
            del tmp
            f[j] = torch.linalg.solve_triangular(Schur_saved[j],f[j],\
                                          unitriangular=False,upper=True)
            
        return f.reshape(b*n,nrhs)
    
    def solve_lowmem(self,f):
        b = self.b; n = self.n
        
        if (not self.cyclic):
            return self.__solve_lowmem(f)
        else:
            nrhs    = f.shape[-1] 
            u_vec   = self.HBT_prime.__solve_lowmem(f[n:])
            
            V_blocked, X1_LU, X1_piv, hbs_sub0, hbs_sup0 = self.acyclic_solve_data
            
            rhs_tmp  = f[:n] - hbs_sup0.matvec(u_vec[:n].unsqueeze(0),b_start=0,b_end=1)
            rhs_tmp -= hbs_sub0.matvec(u_vec[(b-2)*n:].unsqueeze(0),b_start=0,b_end=1)
            
            result = torch.zeros(f.shape)
            result[:n] = torch.lu_solve(rhs_tmp.squeeze(0), X1_LU, X1_piv)
            result[n:] = (V_blocked @ result[:n]).reshape((b-1)*n,nrhs) + u_vec
            return result
            
        
    def solve(self,f):
        if (self.build_mode == 'simple'):
            return self.solve_simple(f)
        elif (self.build_mode == 'lowmem'):
            return self.solve_lowmem(f)
        else:
            raise ValueError
            
    def solve_simple(self,f):   
        assert self.build_mode == 'simple'
        return self.solve_op.lusolve_sweep(f.unsqueeze(0)).squeeze(0)
    
    #def apply_tmp(self,f):
    #    return self.apply(self.hbs_diag0,self.hbs_diag1,self.hbs_sub,self.hbs_sup,f)
            
    def apply(self,hbs_diag0,hbs_diag1,hbs_sub,hbs_sup,f):
        nrhs = f.shape[-1]
        b = self.b; n = self.n
        f_blocked = f.reshape(b,n,nrhs)
        result = torch.zeros(f_blocked.shape,device=f_blocked.device)
        
        ### apply diagonal
        result += hbs_diag0.matvec(f_blocked)
        result += hbs_diag1.matvec(f_blocked)

        if (not self.cyclic):
            ### apply sub and sup diagonals
            result[1:]   += hbs_sub.matvec(f_blocked[:b-1])
            result[:b-1] += hbs_sup.matvec(f_blocked[1:])
        else:
            result       += hbs_sub.matvec( torch.cat((f_blocked[-1].unsqueeze(0),\
                                                      f_blocked[:-1])))
            result       += hbs_sup.matvec( torch.cat((f_blocked[1:], \
                                                        f_blocked[0].unsqueeze(0))))
            
        return result.reshape(b*n,nrhs)                                
