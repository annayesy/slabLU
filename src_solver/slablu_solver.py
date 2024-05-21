import torch
torch.set_default_dtype(torch.double)
import numpy as np
from time import time
from functools import reduce

# scipy sparse imports
from scipy.sparse import kron, diags, eye as speye, hstack as sp_hstack
import scipy.sparse
import scipy.sparse.linalg as spla

# discretization imports
from src_disc import pdo, hps_multidomain_disc
from src_disc.fd_disc import FD_disc

# solver imports
import src_solver.block_tridiag_solver_dense as block_tridiag
from src_solver.hbs_reconstruction import HBS_matrix,copy_params_hbs
from src_solver.block_tridiag_solver_structured import HBS_BlockTridiagonal

try:
    from petsc4py import PETSc
    petsc_available = True
except:
    petsc_available = False
    
def to_torch_csr(A,device=torch.device('cpu')):
    A.sort_indices()
    A = torch.sparse_csr_tensor(A.indptr,A.indices,A.data,\
                                        size=(A.shape[0],A.shape[1]),device=device)
    return A

# find the divisor of n that is nearest to x
# if there are two options, pick the larger divisor
def get_nearest_div(n,x):
    factors =  list(reduce(list.__add__, 
                ([i, int(n/i)] for i in range(1, int(n**0.5) + 1) if n % i == 0)))
    factors = torch.tensor(factors)
    nearest_div = n; dist = np.abs(nearest_div-x)
    for f in factors:
        dist_f = np.abs(f-x)
        if (dist_f < dist):
            nearest_div = f; dist = dist_f
        elif (dist_f == dist):
            nearest_div = np.max([f,nearest_div])
    return nearest_div.item()

def apply_sparse_lowmem(A,I,J,v,transpose=False):
    vec_full = torch.zeros(A.shape[0],v.shape[-1])
    if (not transpose):
        vec_full[J] = v
        vec_full = A @ vec_full
        return torch.tensor(vec_full[I])
    else:
        vec_full[J] = v
        vec_full = A.T @ vec_full
        return torch.tensor(vec_full[I])

class Domain_Solver:
    
    def __init__(self,box_geom,pdo_op,kh,h,p=0, \
                 buf_constant=0.5,periodic_bc=False):
        self.kh = kh;
        self.periodic_bc = periodic_bc
        if (periodic_bc):
            assert p > 0
        
        ## buffer size is chosen as buf_constant * n^{2/3}
        self.buf_constant = buf_constant
        
        if (p==0):
            self.fd_disc(box_geom,h,pdo_op)
            self.fd_panel_split()
            self.disc='fd'
            self.ntot = self.fd.XX.shape[0]
        else:
            # interpret h as parameter a
            self.hps_disc(box_geom,h,p,pdo_op)
            self.hps_panel_split()
            self.disc='hps'
            self.ntot = self.hps.xx_active.shape[0]

        # local inds for each slab
        I_L = self.I_L; I_R = self.I_R; I_U = self.I_U; I_D = self.I_D
        I_C = self.I_C; Npan = self.Npan

        # all internal nodes for slab
        I_slabC = self.inds_pans[:,I_C].flatten(); slab_Cshape = I_C.shape[0]
        
        # interfaces between slabs
        if (periodic_bc):
            I_slabX = torch.cat((self.inds_pans[:,I_L].flatten(),self.inds_pans[Npan-1,I_R])); 
        else:
            I_slabX = self.inds_pans[1:,I_L].flatten()
        slab_Xshape = I_L.shape[0]
        self.I_slabX = I_slabX
        I_Ctot   = torch.hstack((I_slabC,I_slabX))
        
        if (periodic_bc):
            slab_bnd_size = self.I_L.shape[0]; slab_int_size = self.I_C.shape[0]
            slab_interior_offset = slab_int_size * self.Npan
            self.I_Ctot_unique = torch.arange(slab_interior_offset + slab_bnd_size * (self.Npan))
            self.I_Ctot_copy1  = torch.arange(slab_bnd_size) + slab_interior_offset
            self.I_Ctot_copy2  = torch.arange(slab_bnd_size) + slab_interior_offset + slab_bnd_size * self.Npan

        # dirichlet data for entire domain
        I_Ldir = self.inds_pans[0,I_L]
        I_Rdir = self.inds_pans[Npan-1,I_R]
        if (self.disc == 'hps'):
            I_Ddir = self.inds_pans[:,I_D].flatten();
            I_Udir = self.inds_pans[:,I_U].flatten();
        elif (self.disc == 'fd'):
            I_Ddir = torch.hstack((self.inds_pans[0,I_D],\
                                   self.inds_pans[1:, I_D[1:]].flatten()))
            I_Udir = torch.hstack((self.inds_pans[0,I_U],\
                           self.inds_pans[1:, I_U[1:]].flatten()))
        
        self.I_slabX = I_slabX; self.I_slabC = I_slabC
        self.I_Ctot  = I_Ctot;
        if (periodic_bc):
            self.I_Xtot  = torch.hstack((I_Ddir,I_Udir))
        else:
            self.I_Xtot  = torch.hstack((I_Ldir,I_Rdir,I_Ddir,I_Udir))
            
            
    ############################### HPS discretiation and panel split #####################
    def hps_disc(self,box_geom,a,p,pdo_op):

        HPS_multi = hps_multidomain_disc.HPS_Multidomain(pdo_op,box_geom,a,p)

        # find buf
        size_face = HPS_multi.p-2; n0,n1 = HPS_multi.n
        n0 = n0.item(); n1 = n1.item()
        npan_max = torch.max(HPS_multi.n).item()
        n_tmp = (npan_max) * size_face - 1; n_tmp = n_tmp
        
        # set constant to 0.5,1.0 works on ladyzhen
        buf_points = int(n_tmp**(2/3)*self.buf_constant); buf_points = np.min([400,buf_points])
            
        buf = np.max([int(buf_points/size_face)+1,2]); 
        buf = get_nearest_div(n0,buf);

        Npan = int(n0/buf); 
        print("HPS discretization a=%5.2e,p=%d"%(a,p))
        print("\t--params(n0,n1,buf) (%d,%d,%d)"%(n0,n1,buf))

        nfaces_pan  = (2*n1+1)*buf + n1
        inds_pans   = torch.zeros(Npan,nfaces_pan*size_face).long()

        for j in range(Npan):
            npan_offset  = (2*n1+1)*buf * j
            inds_pans[j] = torch.arange(nfaces_pan*size_face) + npan_offset*size_face

        self.Npan      = Npan;
        self.Npan_loc  = n1;
        self.buf_pans  = buf
        self.inds_pans = inds_pans
        
        self.elim_nblocks = buf-1;      
        self.elim_bs = size_face
        self.rem_nblocks  = self.Npan_loc-1; 
        self.rem_bs  = buf*size_face
        self.hps = HPS_multi
        
    def hps_panel_split(self):
        
        size_face = self.hps.p-2; n0,n1 = self.hps.n; buf = self.buf_pans
        Npan_loc  = self.Npan_loc
        
        elim_nblocks = self.elim_nblocks;      elim_bs = self.elim_bs
        rem_nblocks  = self.rem_nblocks;       rem_bs  = self.rem_bs
        
        self.I_L = torch.arange(n1*size_face)
        self.I_R = torch.arange(n1*size_face) + (2*n1+1)*buf*size_face

        I_elim = torch.zeros(Npan_loc,elim_nblocks,elim_bs).long()
        I_rem  = torch.zeros(rem_nblocks,buf,size_face).long()

        I_D    = torch.zeros(buf,size_face).long()
        I_U    = torch.zeros(buf,size_face).long()

        for b in range(buf):

            buf_offset = (2*n1+1)*b

            # exterior down index
            I_D[b] = torch.arange(size_face) + (buf_offset+n1)*size_face
            # exterior up index
            I_U[b] = torch.arange(size_face) + (buf_offset+2*n1)*size_face
            # rem index
            for box_j in range(1,n1):
                I_rem[box_j-1,b] = torch.arange(size_face) + (buf_offset+n1+box_j) * size_face
            if (b > 0):
                for box_j in range(n1):
                    I_elim[box_j,b-1] = torch.arange(size_face) + (buf_offset+box_j) * size_face

        I_rem  = I_rem.flatten(start_dim=1,end_dim=-1)
        I_elim = I_elim.flatten(start_dim=1,end_dim=-1)
        self.I_D = I_D.flatten()
        self.I_U = I_U.flatten()

        self.I_C = torch.hstack((I_elim.flatten(),I_rem.flatten()))
        
    ############################### FD discretiation and panel split #####################
    
    def fd_disc(self,box_geom,h,pdo_op):
        
        ## fd discretization
        fd = FD_disc(box_geom,h,pdo_op)
        self.fd = fd
     
    def fd_panel_split(self):
        
        ns = self.fd.ns; 
        n  = torch.max(ns)
        
        buf = (n-1)**(2/3) * self.buf_constant; # set to 0.4,0.6 works on ladyzhen
        buf_prime = np.sqrt(buf);
        
        # find nearest divisible
        buf = get_nearest_div(ns[0]-1,int(buf)+1);
        buf_prime = get_nearest_div(ns[1]-1,int(buf_prime)+1);
        self.buf = buf; self.buf_prime = buf_prime
        print("FD discretization")
        print("\t--(n0,n1,buf,buf_prime) (%d,%d,%d,%d)"%(ns[0],ns[1],buf,buf_prime))

        Npan = ns[0]/(self.buf)
        Npan = int(Npan)

        Npan_loc = ns[1]/(self.buf_prime)
        Npan_loc = int(Npan_loc)

        inds_pans = torch.zeros(Npan,(buf+1)*(ns[1])).long()

        for j in range(Npan):
            tmp = torch.arange(0,(buf+1)*(ns[1]))+j*buf*(ns[1])
            inds_pans[j] = tmp.long()

        self.inds_pans = inds_pans; self.Npan = Npan; self.Npan_loc = Npan_loc
        
        self.elim_nblocks = buf-1;      self.elim_bs = buf_prime-1
        self.rem_nblocks  = Npan_loc-1; self.rem_bs  = buf-1
        
        
        n = self.fd.ns[1]-1; buf = self.buf; buf_prime = self.buf_prime
        Npan_loc = self.Npan_loc
        
        I_L = torch.arange(1,n)
        I_R = torch.arange(1,n) + (buf) * (n+1)

        I_D = torch.arange(0,(buf+1)*n,n+1)+0
        I_U = torch.arange(0,(buf+1)*n,n+1)+n
        
        self.I_L = I_L; self.I_R = I_R; self.I_D = I_D; self.I_U = I_U
        
        #### internal nodes
        I_C = torch.zeros((buf-1)*(n-1)).long()
        offset = 0
        for j in range(1,n):
            tmp = torch.arange(n+1,buf*n+1,n+1)+j
            I_C[offset : offset + buf-1] = tmp
            offset += buf-1

        elim_nblocks = self.elim_nblocks;      elim_bs = self.elim_bs
        rem_nblocks  = self.rem_nblocks;       rem_bs  = self.rem_bs
        
        ### reorder I_C as I_elim, I_rem
        I_elim = torch.zeros(Npan_loc,elim_nblocks*elim_bs).long()
        I_rem = torch.zeros(rem_nblocks,rem_bs).long()

        for j in range(0,n-1):

            part_index = int(j/buf_prime)
            rem = np.mod(j,buf_prime)

            if (rem == buf_prime-1 ):
                part_index = int(j/buf_prime)
                I_rem[part_index] = torch.arange(j*(buf-1), (j+1)*(buf-1))

            else:
                I_elim[part_index,(rem)*(buf-1):(rem+1)*(buf-1)] = torch.arange(j*(buf-1),(j+1)*(buf-1))
            
        # reorder I_C
        I_reorder = torch.zeros(I_C.shape[0]).long()
        offset = 0
        for j in range(Npan_loc):
            inds = I_elim[j]
            I_reorder[ offset : offset + elim_nblocks*elim_bs ] = torch.sort(I_C[inds]).values
            offset += inds.shape[0]
        for j in range(Npan_loc-1):
            inds = I_rem[j]
            I_reorder[ offset : offset + rem_bs ] = torch.sort(I_C[inds]).values
            offset += inds.shape[0]    
        self.I_C = I_reorder
        

    def build_slabLU(self,verbose):
        
        tic = time()
        self.build_Aloc_solver()
        toc_Aloc = time() - tic; 
        
        tic = time()
        self.construct_Ared()
        toc_Ared = time() - tic;
        
        Aloc_solver_stor = self.Aloc_solver.nbytes /1e9
        Ared_stor = self.Ared_solver.nbytes /1e9
             
        if verbose:
            print("SPARSE SLAB OPERATIONS")
            print("\t--time for (Aloc solver, Ared solver) (%5.2f,%5.2f) s"\
                  % (toc_Aloc, toc_Ared))
            print("\t--memory for (Aloc solver, Ared solver) (%5.2f,%5.2f) GB"\
              % (Aloc_solver_stor, Ared_stor))
        
            rel_err_Aloc = self.test_Aloc_solver()
            rel_err_Ared = self.test_Ared_solver()
            rel_err_slab = self.test_slab_solve()

            print("\t--relerror in (Aelim solve,Ared solve,slab solve) (%5.2e,%5.2e,%5.2e)"\
                  % (rel_err_Aloc,rel_err_Ared,rel_err_slab))
            
        total_sparse_time = toc_Aloc+toc_Ared
        
        ########## construct and build tridiag solve operator for Tred ##########
        print("TRED ASSEMBLY MESSAGES")

        toc_Tred_assemble,toc_Tred_factorize = self.build_Tred_solver()
        Tred_stor = self.Tred_solver.nbytes/1e9
        
        if (verbose):
            print("BLOCK TRIDIAG TRED OPERATIONS")
            print("\t--time for (Tred assembly,Tred solver) (%5.2f,%5.2f) s" \
                  % (toc_Tred_assemble,toc_Tred_factorize))
            print("\t--memory for (Tred solver) (%5.2f) GB"\
                  % (Tred_stor))

            rel_err = self.test_Tred_solver()
            print("\t--relerror in Tred solve (%5.2e)"%rel_err)
            
        total_Tred_time  = toc_Tred_assemble+toc_Tred_factorize
        total_build_time = total_sparse_time + total_Tred_time
        
        total_memory = Aloc_solver_stor + Ared_stor + Tred_stor
        
        if (verbose):
            print("BUILD SUMMARY")
            print("\t--time for sparse ops + Tred ops = total, %5.2f + %5.2f = %5.2f seconds"\
                  % (total_sparse_time,total_Tred_time,total_build_time))
            print("\t--total memory = %5.2f GB"\
                  % (total_memory))
            
        ## record relevant information in dictionary
        info_dict = dict()
        info_dict['toc_build']  = total_build_time
        info_dict['mem_build']  = total_memory
        return info_dict
    
    def build_superLU(self,verbose):
        info_dict = dict()
        try:
            tic = time()
            LU = spla.splu(self.A_CC)          
            toc_superLU = time() - tic
            if (verbose):
                print("SUPER LU BUILD SUMMARY")
            mem_superLU  = LU.L.data.nbytes + LU.L.indices.nbytes + LU.L.indptr.nbytes
            mem_superLU += LU.U.data.nbytes + LU.U.indices.nbytes + LU.U.indptr.nbytes
            stor_superLU = mem_superLU/1e9
            if (verbose):
                print("\t--time superLU build = %5.2f seconds"\
                      % (toc_superLU))
                print("\t--total memory = %5.2f GB"\
                      % (stor_superLU))
            self.superLU = LU

            info_dict['toc_build_superLU'] = toc_superLU
            info_dict['toc_build_blackbox']= toc_superLU
            info_dict['solver_type']       = 'scipy_superLU'
            
            info_dict['mem_build_superLU'] = stor_superLU
        except:
            print("SuperLU had an error.")
        return info_dict
    
    def build_petsc(self,solvertype,verbose):
        info_dict = dict()
        tmp = self.A_CC.tocsr()
        pA = PETSc.Mat().createAIJ(tmp.shape, csr=(tmp.indptr,tmp.indices,tmp.data))
        
        ksp = PETSc.KSP().create()
        ksp.setOperators(pA)
        ksp.setType('preonly')
        
        ksp.getPC().setType('lu')
        ksp.getPC().setFactorSolverType(solvertype)
        
        px = PETSc.Vec().createWithArray(np.ones(tmp.shape[0]))
        pb = PETSc.Vec().createWithArray(np.ones(tmp.shape[0]))
        
        tic = time()
        ksp.solve(pb, px)
        toc_build = time() - tic
        if (verbose):
            print("\t--time for %s build through petsc = %5.2f seconds"\
                  % (solvertype,toc_build))
               
        info_dict['toc_build_blackbox']   = toc_build
        info_dict['solver_type']          = "petsc_"+solvertype
        
        self.petsc_LU = ksp
        return info_dict
    
    def build_blackboxsolver(self,solvertype,verbose):
        info_dict = dict()
        
        if (not self.periodic_bc):
            A_CC = self.A[self.I_Ctot][:,self.I_Ctot].tocsc()
            self.A_CC = A_CC
        else:
            A_copy = self.A[np.ix_(self.I_Ctot,self.I_Ctot)].tolil()
            A_copy[np.ix_(self.I_Ctot_unique,self.I_Ctot_copy1)] += \
            A_copy[np.ix_(self.I_Ctot_unique,self.I_Ctot_copy2)]
        
            A_copy[np.ix_(self.I_Ctot_copy1,self.I_Ctot_unique)] += \
            A_copy[np.ix_(self.I_Ctot_copy2,self.I_Ctot_unique)] 
        
            A_copy[np.ix_(self.I_Ctot_copy1,self.I_Ctot_copy1)] += \
            A_copy[np.ix_(self.I_Ctot_copy2,self.I_Ctot_copy2)]
            
            A_CC = A_copy[np.ix_(self.I_Ctot_unique,self.I_Ctot_unique)].tocsc()
            self.A_CC = A_CC
        if (not petsc_available):
            info_dict = self.build_superLU(verbose)
        else:
            info_dict = self.build_petsc(solvertype,verbose)
        return info_dict
        
    def build(self,sparse_assembly,
              solver_type,verbose=True):
        
        self.sparse_assembly = sparse_assembly
        self.solver_type     = solver_type
        ########## sparse assembly ##########
        if (self.disc == 'fd'):
            tic = time()
            self.A = self.fd.assemble_sparse();
            toc_assembly_tot = time() - tic;
        elif (self.disc == 'hps'):
            if (sparse_assembly == 'reduced_cpu'):
                device = torch.device('cpu')
                tic = time()
                self.A,assembly_time_dict    = self.hps.sparse_mat(device,verbose)
                toc_assembly_tot = time() - tic;
            elif (sparse_assembly == 'reduced_gpu'):
                device = torch.device('cuda')
                tic = time()
                self.A,assembly_time_dict    = self.hps.sparse_mat(device,verbose)
                toc_assembly_tot = time() - tic;
                
        csr_stor  = self.A.data.nbytes
        csr_stor += self.A.indices.nbytes + self.A.indptr.nbytes
        csr_stor /= 1e9
        if (verbose):
            print("SPARSE ASSEMBLY")
            print("\t--time for (sparse assembly) (%5.2f) s"\
                  % (toc_assembly_tot))
            print("\t--memory for (A sparse) (%5.2f) GB"\
              % (csr_stor))
        
        assert self.ntot == self.A.shape[0]
        ########## sparse slab operations ##########
        info_dict = dict()
        if (solver_type == 'slabLU'):
            info_dict = self.build_slabLU(verbose)
            info_dict['toc_build'] += toc_assembly_tot
        else:
            info_dict = self.build_blackboxsolver(solver_type,verbose)
            if ('toc_build_blackbox' in info_dict):
                info_dict['toc_build_blackbox'] += toc_assembly_tot

        if (self.disc == 'fd'):
            info_dict['toc_assembly'] = toc_assembly_tot
        else:
            info_dict['toc_assembly'] = assembly_time_dict['toc_DtN']
        return info_dict
                
    def get_rhs(self,uu_dir_func,ff_body_func=None,sum_body_load=True):
        I_slabX = self.I_slabX; I_slabC = self.I_slabC
        I_Ctot  = self.I_Ctot;  I_Xtot  = self.I_Xtot; 
        
        slab_Cshape = self.I_C.shape[0]; slab_Xshape = self.I_L.shape[0]
        Npan = self.Npan
        nrhs = 1
        
        ## assume that XX has size npoints, 2
        if (self.disc == 'fd'):
            XX = self.fd.XX
        elif (self.disc == 'hps'):
            XX = self.hps.xx_active
            
        # Dirichlet data
        uu_dir = uu_dir_func(XX[I_Xtot,:])

        # body load on I_Ctot
        ff_body = -apply_sparse_lowmem(self.A,I_Ctot,I_Xtot,uu_dir)
        if (ff_body_func is not None):
            
            if (self.disc == 'hps'):
                
                if (self.sparse_assembly == 'reduced_gpu'):
                    device = torch.device('cuda')
                    ff_body += self.hps.reduce_body(device,ff_body_func)[I_Ctot]
                elif (self.sparse_assembly == 'reduced_cpu'):
                    device = torch.device('cpu')
                    ff_body += self.hps.reduce_body(device,ff_body_func)[I_Ctot]
            elif (self.disc == 'fd'):
                ff_body += ff_body_func(XX[I_Ctot,:])
        
        # adjust to sum body load on left and right boundaries
        if (self.periodic_bc and sum_body_load):
            ff_body[self.I_Ctot_copy1] += ff_body[self.I_Ctot_copy2]
            ff_body = ff_body[self.I_Ctot_unique]
        
        return ff_body
    
    def solve_residual_calc(self,sol,ff_body):
        if (not self.periodic_bc):
            res = apply_sparse_lowmem(self.A,self.I_Ctot,self.I_Ctot,sol) - ff_body
        else:
            res  = - ff_body
            res += apply_sparse_lowmem(self.A,self.I_Ctot[self.I_Ctot_unique],\
                                       self.I_Ctot[self.I_Ctot_unique], sol)
            res += apply_sparse_lowmem(self.A,self.I_Ctot[self.I_Ctot_unique],\
                                       self.I_Ctot[self.I_Ctot_copy2], sol[self.I_Ctot_copy1])
            res[self.I_Ctot_copy1] += apply_sparse_lowmem(self.A,self.I_Ctot[self.I_Ctot_copy2],\
                                       self.I_Ctot[self.I_Ctot_unique], sol)
            res[self.I_Ctot_copy1] += apply_sparse_lowmem(self.A,self.I_Ctot[self.I_Ctot_copy2],\
                                       self.I_Ctot[self.I_Ctot_copy2], sol[self.I_Ctot_copy1])
        return res
        
    def solve_helper(self,uu_dir_func,ff_body_func=None):
        
        I_slabX = self.I_slabX; I_slabC = self.I_slabC
        I_Ctot  = self.I_Ctot;  I_Xtot  = self.I_Xtot; 
        
        slab_Cshape = self.I_C.shape[0]; slab_Xshape = self.I_L.shape[0]
        Npan = self.Npan
        nrhs = 1
        
        tic = time()
        ## for slabLU solver, do not sum body load on L and R inds
        ## in the case of periodic BC
        ff_body = self.get_rhs(uu_dir_func,ff_body_func,sum_body_load=False)
        
        # now that we have the right hand side, proceed with solving
        ff_slabC = self.solve_slab(0,Npan,ff_body[:Npan*slab_Cshape].reshape(Npan,slab_Cshape,nrhs))
        ff_slabC = ff_slabC.flatten(start_dim=0,end_dim=-2)

        # equivalent body load on I_slabX
        ff_slabX = ff_body[Npan*slab_Cshape:] - apply_sparse_lowmem(self.A,I_slabX,I_slabC,ff_slabC)
        del ff_slabC
        if (self.periodic_bc):
            ff_slabX[:slab_Xshape] += ff_slabX[slab_Xshape*Npan:]
            ff_slabX = ff_slabX[:slab_Xshape*Npan]
        
        # solution on I_slabX
        sol_X = self.Tred_solver.solve(ff_slabX).flatten(start_dim=0,end_dim=-2)
        del ff_slabX
        
        if (self.periodic_bc):
            sol_X_tmp = torch.zeros(I_slabX.shape[0],1)
            sol_X_tmp[:slab_Xshape*Npan] = sol_X
            sol_X_tmp[slab_Xshape*Npan:] = sol_X[:slab_Xshape]
            sol_X = sol_X_tmp
        
        sol = torch.zeros(I_Ctot.shape[0],1)
        sol[Npan*slab_Cshape:] = sol_X
        del sol_X
        
        tmp = apply_sparse_lowmem(self.A,I_slabC,I_slabX, sol[Npan*slab_Cshape:])
        sol[:Npan*slab_Cshape] =  self.solve_slab(0,Npan,
                                                  (ff_body[:Npan*slab_Cshape]-tmp).\
                                                  reshape(Npan,slab_Cshape,nrhs)).\
        flatten(start_dim=0,end_dim=-2)
        del tmp;
        toc_solve = time() - tic
            
        ### calculating residual
        if (self.periodic_bc):
            ff_body = self.get_rhs(uu_dir_func,ff_body_func,sum_body_load=True)
            sol     = sol[self.I_Ctot_unique]
        res = self.solve_residual_calc(sol,ff_body)

        rel_err = torch.linalg.norm(res) / torch.linalg.norm(ff_body)
        del res
        
        return sol,rel_err,toc_solve
    
    def solve_helper_blackbox(self,uu_dir_func,ff_body_func=None):
        
        tic = time()
        ff_body = self.get_rhs(uu_dir_func,ff_body_func); ff_body = np.array(ff_body)
        try:
            if (not petsc_available):
                sol = self.superLU.solve(ff_body)
            else:
                psol = PETSc.Vec().createWithArray(np.ones(ff_body.shape))
                pb   = PETSc.Vec().createWithArray(ff_body.copy())
                self.petsc_LU.solve(pb,psol)
                sol  = psol.getArray().reshape(ff_body.shape)
        except:
            return 0,0,0
        sol = torch.tensor(sol); ff_body = torch.tensor(ff_body)
        toc_solve = time() - tic
        res = self.solve_residual_calc(sol,ff_body)
        
        rel_err = torch.linalg.norm(res) / torch.linalg.norm(ff_body)
        return sol,rel_err,toc_solve
        
        
    def solve(self,uu_dir_func,ff_body_func=None,known_sol=False):
        
        if (self.solver_type == 'slabLU'):
            sol,rel_err,toc_solve = self.solve_helper(uu_dir_func,ff_body_func)
        else:
            sol,rel_err,toc_solve = self.solve_helper_blackbox(uu_dir_func,ff_body_func)
        
        sol_tot = torch.zeros(self.A.shape[0],1)
        
        if (not self.periodic_bc):
            sol_tot[self.I_Ctot] = sol
        else:            
            sol_tot[self.I_Ctot[self.I_Ctot_unique]] = sol
            sol_tot[self.I_Ctot[self.I_Ctot_copy2]]  = sol[self.I_Ctot_copy1]
        
        
        if (self.disc == 'fd'):
            sol_tot[self.I_Xtot] = uu_dir_func(self.fd.XX[self.I_Xtot])
        elif(self.disc == 'hps'):
            sol_tot[self.I_Xtot] = uu_dir_func(self.hps.xx_active[self.I_Xtot])
        del sol
        
        resloc_hps = np.float64('nan')
        if ((self.disc == 'hps') and (self.sparse_assembly.startswith('reduced'))):
            if (self.sparse_assembly == 'reduced_gpu'):
                device=torch.device('cuda')
            else:
                device = torch.device('cpu')
            tic = time()
            sol_tot,resloc_hps = self.hps.solve(device,sol_tot,ff_body_func=ff_body_func)
            toc_solve += time() - tic
            sol_tot = sol_tot.cpu()

        true_err = torch.tensor([float('nan')])
        if (known_sol):
            if (self.disc=='fd'):
                XX = self.fd.XX
            elif (self.disc=='hps'):
                XX = self.hps.xx_tot
            uu_true = uu_dir_func(XX.clone())
            true_err = torch.linalg.norm(sol_tot-uu_true) / torch.linalg.norm(uu_true)
            del uu_true

        return sol_tot,rel_err,true_err,resloc_hps,toc_solve
        
    def build_Tred_solver(self):
        tic = time()
        hbs_data = self.assemble_build_Tred()
        toc_assemble = time() - tic;

        if (not self.periodic_bc):
            hbs_blocktri = HBS_BlockTridiagonal(self.Npan-1,self.I_L.shape[0])
        else:
            hbs_blocktri = HBS_BlockTridiagonal(self.Npan,self.I_L.shape[0],cyclic=True)

        if (torch.cuda.is_available()):
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        
        tic = time()
        hbs_blocktri.build_lowmem(*hbs_data,device=device)
        toc_factorize = time() - tic;
        self.Tred_solver = hbs_blocktri
        return toc_assemble,toc_factorize

    def test_Tred_solver(self):
        f = torch.rand(self.Tred_solver.b*self.Tred_solver.n,1)
        
        sol = self.Tred_solver.solve(f.clone())
        res = self.apply_Tred(sol) - f
        err = torch.linalg.norm(res) / torch.linalg.norm(f)
        return err
        
    def apply_Aloc(self,Npan_start,Npan_end,rhs):
        
        Npan = Npan_end-Npan_start
        Npan_loc = self.Npan_loc;
        I_C = self.I_C

        bs = self.elim_bs; m = self.elim_nblocks
        
        I = self.inds_pans[Npan_start:Npan_end,I_C[:Npan_loc*m*bs]].flatten()
        A_CC = self.A[I][:,I].tocsr()
        
        C_shape = I.shape[0]
        
        rhs_flat = rhs.flatten(start_dim=0,end_dim=-2)
        del rhs
        
        A_CC = to_torch_csr(A_CC,device=rhs_flat.device)
        result = A_CC @ rhs_flat
        return result.reshape(Npan*Npan_loc,m*bs,rhs_flat.shape[-1])
        

    def build_Aloc_solver(self):
        Npan = self.Npan
        Npan_loc = self.Npan_loc;
        I_C = self.I_C

        bs = self.elim_bs; m = self.elim_nblocks

        I = self.inds_pans[:,I_C[:Npan_loc*m*bs]].flatten()
        A_CC = self.A[I][:,I].tocsr()

        # allocate Ablock_data
        Aloc_solver = block_tridiag.Block_Tridiagonal(Npan*Npan_loc,m,bs)
        
        Ablock_data = torch.zeros(Npan*Npan_loc,Aloc_solver.pivdata_nblocks+m+2*(m-1),\
                                  bs, bs)
        
        if (torch.cuda.is_available()):
            device=torch.device('cuda')
        else:
            device=torch.device('cpu')
        
        for j in range(Npan):
            func = lambda args: args[0].apply_Aloc(j,j+1,args[1])
            Ablock_data[j*Npan_loc : (j+1) * Npan_loc] = \
            block_tridiag.recover_block_tridiag_via_sampling(Npan_loc,m,bs,func,self).cpu()
        
        Aloc_solver.build_sweep_from_data(Ablock_data,fast_LU=False)
        self.Aloc_solver = Aloc_solver
        
    def test_Aloc_solver(self):
        Npan = self.Npan; Npan_loc = self.Npan_loc
        
        bs = self.elim_bs; m = self.elim_nblocks

        cpart_size = m*bs;
        xoffset = m*bs*Npan_loc

        I_elim = self.inds_pans[:,self.I_C[:xoffset]].flatten()
        I_rem  = self.inds_pans[:,self.I_C[xoffset:]].flatten()
        Aelim = self.A[I_elim][:,I_elim]

        x = torch.rand(Npan*xoffset,5)
 
        tmp = Aelim @ x
        tmp = torch.tensor(tmp).reshape(Npan*Npan_loc,m*bs,x.shape[-1])
        err = torch.linalg.norm(self.Aloc_solver.lusolve_sweep(tmp).reshape(Npan*xoffset,x.shape[-1]) - x)
        rel_err = err / torch.linalg.norm(x)
        return rel_err
        
        
    def elim_lowmem(self,Alocsolver_start,Alocsolver_end,\
                     A_Xelim,A_elimX,rhs_flat,transpose=False,verbose=False):
        
        b_loc = Alocsolver_end - Alocsolver_start
        m = self.elim_nblocks; bs = self.elim_bs
        nvecs = rhs_flat.shape[-1]
        
        result = torch.zeros(A_Xelim.shape[0],nvecs,device=rhs_flat.device)
        
        b_curr = 0;
        while (b_curr < b_loc):
            
            # calculate appropriate chunk_size
            if (torch.cuda.is_available()):
                r = torch.cuda.memory_reserved(0)
                a = torch.cuda.memory_allocated(0)
                f = r-a # in bytes
                f = np.min([f,12e9])
            else:
                f = 12e9 # 12 GB in bytes
                
            chunk_size = int(f / (m*bs*(nvecs+3*bs)*8) )
            chunk_size = np.min([int(chunk_size/3),b_loc - b_curr])
            if (verbose):
                print("elim lowmem chunk", Alocsolver_start+b_curr,\
                      Alocsolver_start+b_curr+chunk_size, rhs_flat.device,\
                     "with free space(GB)", f/1e9)
            
            Atmp_elimX = to_torch_csr(A_elimX[b_curr*m*bs : (b_curr+chunk_size)*m*bs],\
                                      device=rhs_flat.device)
            Atmp_Xelim = to_torch_csr(A_Xelim[:,b_curr*m*bs : (b_curr+chunk_size)*m*bs],\
                                      device=rhs_flat.device)
            
            tmp = Atmp_elimX @ rhs_flat
            del Atmp_elimX
            tmp = tmp.reshape(chunk_size,m*bs,nvecs)
            tmp = self.Aloc_solver.lusolve_sweep(tmp, b_start=Alocsolver_start+b_curr,\
                                                 b_end=Alocsolver_start+b_curr+chunk_size,\
                                                transpose=transpose)
            tmp = tmp.reshape(chunk_size*m*bs,nvecs)
            result += Atmp_Xelim @ tmp
            del tmp; del Atmp_Xelim
            b_curr += chunk_size
        return result
    
    # rhs is of size Npan, (Npan_loc-1)*(buf-1), nrhs
    def apply_Ared_lowmem(self,Npan_start,Npan_end,rhs):
        Npan = Npan_end-Npan_start
        I_C   = self.I_C; Npan_loc = self.Npan_loc
        elim_nblocks = self.elim_nblocks; elim_bs = self.elim_bs
        rem_nblocks = self.rem_nblocks; rem_bs = self.rem_bs
        
        cpart_size = elim_nblocks * elim_bs
        xpart_size = rem_bs
        xoffset    = cpart_size*Npan_loc
        
        I_elim = self.inds_pans[Npan_start:Npan_end,I_C[:xoffset]].flatten()
        I_rem = self.inds_pans[Npan_start:Npan_end,I_C[xoffset:]].flatten()
        
        Arem = self.A[I_rem][:,I_rem].tocsr()
        Arem_elim = self.A[I_rem][:,I_elim].tocsr()
        Aelim_rem = self.A[I_elim][:,I_rem].tocsr()
        
        rem_shape = I_rem.shape[0]
        elim_shape = I_elim.shape[0]
        
        rhs_flat = rhs.flatten(start_dim=0,end_dim=-2)
        del rhs
        
        sol_rem  = -self.elim_lowmem(Npan_start*Npan_loc,Npan_end*Npan_loc,\
                                      Arem_elim, Aelim_rem,rhs_flat)
        
        Arem = to_torch_csr(Arem,device=rhs_flat.device)
        
        sol_rem += Arem @ rhs_flat
        return sol_rem.reshape(Npan,rem_nblocks * rem_bs,rhs_flat.shape[-1])
        
    def construct_Ared_helper(self,Npan_start,Npan_end,device):    
        m  = self.rem_nblocks
        bs = self.rem_bs
        Npan = Npan_end - Npan_start

        func = lambda args: args[0].apply_Ared_lowmem(Npan_start,Npan_end,args[1])
        
        Ablock_data = block_tridiag.recover_block_tridiag_via_sampling(Npan,m,bs,func,self,\
                                                                      device=device)
        return Ablock_data.cpu()
        
    def construct_Ared(self,verbose=False):
        m  = self.rem_nblocks
        bs = self.rem_bs
        Npan = self.Npan
        
        self.Ared_solver = block_tridiag.Block_Tridiagonal(Npan,m,bs)
        
        Ablock_data = torch.zeros(Npan,self.Ared_solver.pivdata_nblocks+m+2*(m-1),\
                                  bs,bs)
        
        if (torch.cuda.is_available()):
            device=torch.device('cuda')
        else:
            device=torch.device('cpu')
            
        chunk_max = int(2.5e9 / (m*bs*bs*3*8))
        chunk_max = np.max([chunk_max,1])
        if (verbose):
            print("----Chunk max for Ared construction", chunk_max)
        
        b_curr = 0
        while (b_curr < Npan):
            chunk_size = np.min([chunk_max,Npan-b_curr])
            Ablock_data[b_curr:b_curr+chunk_size] = self.construct_Ared_helper(b_curr,b_curr+chunk_size,device)
            b_curr += chunk_size
            
        self.Ared_solver.build_sweep_from_data(Ablock_data,fast_LU=False)
        
    def test_Ared_solver(self):
        Npan_loc = self.Npan_loc; Npan = self.Npan
        xoffset = self.elim_nblocks * self.elim_bs * Npan_loc

        I_elim = self.inds_pans[:,self.I_C[:xoffset]].flatten()
        I_rem  = self.inds_pans[:,self.I_C[xoffset:]].flatten()
        Aelim = self.A[I_elim][:,I_elim]
        
        x = torch.rand(I_rem.shape[0],3)
        x_blocked = x.reshape(Npan,self.rem_nblocks * self.rem_bs,x.shape[-1])
        Taction_blocked = self.apply_Ared_lowmem(0,Npan,x_blocked)

        err_solve = torch.linalg.norm(self.Ared_solver.lusolve_sweep(Taction_blocked.clone()) - x_blocked)
        rel_err_solve = err_solve / torch.linalg.norm(x_blocked)
        return rel_err_solve
        
    # f has shape Npan, shape(I_C)
    def solve_slab(self,Npan_start,Npan_end,f,transpose=False):
        elim_nblocks = self.elim_nblocks; elim_bs = self.elim_bs
        rem_nblocks  = self.rem_nblocks;  rem_bs  = self.rem_bs
        
        Npan = Npan_end-Npan_start
        Npan_loc = self.Npan_loc
        
        xoffset = elim_nblocks*elim_bs*Npan_loc
        I_C = self.I_C

        I_elim = self.inds_pans[Npan_start:Npan_end,I_C[:xoffset]].flatten()
        I_rem  = self.inds_pans[Npan_start:Npan_end,I_C[xoffset:]].flatten()
        elim_shape = I_elim.shape[0]; rem_shape = I_rem.shape[0]

        f_elim = f[:,:xoffset]
        f_rem = f[:,xoffset:]
        
        f_elim_blocked = f_elim.reshape(Npan*Npan_loc,elim_nblocks*elim_bs,f.shape[-1])
        
        sol = torch.zeros(Npan,I_C.shape[0],f.shape[-1],device=f.device)

        # reduce to I_rem
        tmp = self.Aloc_solver.lusolve_sweep(f_elim_blocked.clone(), b_start=Npan_loc*Npan_start,\
                                            b_end = Npan_loc * Npan_end,transpose=transpose)
        tmp = apply_sparse_lowmem(self.A,I_rem,I_elim, \
                                  tmp.flatten(start_dim=0,end_dim=-2),transpose=transpose)
        fequiv_rem = f_rem - tmp.reshape(Npan,rem_nblocks*rem_bs,f.shape[-1])

        # solve for I_rem
        sol[:,xoffset:] = self.Ared_solver.lusolve_sweep(fequiv_rem,b_start=Npan_start,\
                                                        b_end = Npan_end,transpose=transpose)

        # solve for I_elim
        tmp = apply_sparse_lowmem(self.A,I_elim,I_rem,\
                                  sol[:,xoffset:].flatten(start_dim=0,end_dim=-2),transpose=transpose)
        
        f_elim_blocked -= tmp.reshape(Npan*Npan_loc,elim_bs*elim_nblocks,f.shape[-1])
        sol[:,:xoffset] = self.Aloc_solver.lusolve_sweep(f_elim_blocked,\
                                                         b_start=Npan_loc*Npan_start,\
                                                        b_end=Npan_loc*Npan_end,transpose=transpose).\
        reshape(Npan,Npan_loc*elim_bs*elim_nblocks,f.shape[-1])

        return sol
    
    def test_slab_solve(self,transpose=False):
        Npan_start=0; Npan_end=self.Npan

        f = torch.rand(Npan_end-Npan_start,self.I_C.shape[0],5);

        calc_sol = self.solve_slab(Npan_start,Npan_end,f,transpose=transpose)
        calc_sol_flat = calc_sol.flatten(start_dim=0,end_dim=-2)

        Itot = self.inds_pans[Npan_start:Npan_end,self.I_C].flatten()
        A_CC = self.A[Itot][:,Itot]
        if (transpose):
            A_CC = A_CC.T
        err = torch.linalg.norm(torch.tensor(A_CC @ calc_sol_flat) - f.flatten(start_dim=0,end_dim=-2) )
        rel_err = err / torch.linalg.norm(f)
        return rel_err
    
    ########################################## forming T ##################################################
    
    def apply_Tred(self,rhs,transpose=False):


        Npan_loc = self.Npan_loc; Npan = self.Npan
        xoffset = self.elim_nblocks * self.elim_bs * Npan_loc

        I_X    = self.I_slabX; slab_ext = self.I_L.shape[0]

        I_elim = self.inds_pans[:,self.I_C[:xoffset]].flatten()
        I_rem  = self.inds_pans[:,self.I_C[xoffset:]].flatten()

        device = rhs.device

        if (not transpose):
            Aelim_rem = self.A[I_elim][:,I_rem]
            Arem_elim = self.A[I_rem][:,I_elim]

            Ax_elim = self.A[I_X][:,I_elim]
            Aelim_x = self.A[I_elim][:,I_X]

            Ax_rem = to_torch_csr(self.A[I_X][:,I_rem].tocsr(),device=device)
            Arem_x = to_torch_csr(self.A[I_rem][:,I_X].tocsr(),device=device)
            Axx    = to_torch_csr(self.A[I_X][:,I_X].tocsr(),device=device)
        else:
            Arem_elim = self.A[I_elim][:,I_rem].T.tocsr()
            Aelim_rem = self.A[I_rem][:,I_elim].T.tocsr()

            Aelim_x = self.A[I_X][:,I_elim].T.tocsr()
            Ax_elim = self.A[I_elim][:,I_X].T.tocsr()

            Arem_x = to_torch_csr(self.A[I_X][:,I_rem].T.tocsr(),device=device)
            Ax_rem = to_torch_csr(self.A[I_rem][:,I_X].T.tocsr(),device=device)
            Axx    = to_torch_csr(self.A[I_X][:,I_X].T.tocsr(),device=device)

        Aelim_xrem = sp_hstack([Aelim_x,-Aelim_rem],format="csr")
        
        if (self.periodic_bc):
            rhs_tmp = torch.zeros(I_X.shape[0],rhs.shape[-1])
            rhs_tmp[:slab_ext*Npan] = rhs
            rhs_tmp[slab_ext*Npan:] = rhs[:slab_ext]
            rhs = rhs_tmp

        g = Arem_x @ rhs
        g -= self.elim_lowmem(0, Npan*Npan_loc,\
                              Arem_elim,Aelim_x,rhs,transpose=transpose)

        u_rem = self.Ared_solver.lusolve_sweep(g.reshape(Npan,int(I_rem.shape[0]/Npan),\
                                                        rhs.shape[-1]),\
                                               transpose=transpose)
        u_rem = u_rem.flatten(start_dim=0,end_dim=-2)

        tmp = Ax_rem @ u_rem
        tmp  += self.elim_lowmem(0,(Npan)*Npan_loc,\
                                Ax_elim,Aelim_xrem,\
                                torch.vstack((rhs,u_rem)),transpose=transpose)

        sol_X = Axx @ rhs - tmp
        
        if (self.periodic_bc):
            sol_X[:slab_ext] += sol_X[slab_ext*Npan:]
            sol_X = sol_X[:slab_ext*Npan]

        return sol_X
    
    def apply_Trowcol(self,Npan_ind,rowcol,rhs,transpose=False):

        I_C = self.inds_pans[Npan_ind,self.I_C]
        if (rowcol[0] == 'L'):
            I_row = self.inds_pans[Npan_ind,self.I_L]
        elif (rowcol[0] == 'R'):
            I_row = self.inds_pans[Npan_ind,self.I_R] 
        else:
            raise ValueError

        if (rowcol[1] == 'L'):
            I_col = self.inds_pans[Npan_ind,self.I_L]
        elif (rowcol[1] == 'R'):
            I_col = self.inds_pans[Npan_ind,self.I_R] 
        else:
            raise ValueError

        Csize = self.I_C.shape[0]; Lsize = self.I_L.shape[0]

        if (not transpose):
            A_Ccol = to_torch_csr(self.A[I_C][:,I_col],device=rhs.device)
            A_rowcol = to_torch_csr(self.A[I_row][:,I_col],device=rhs.device)
            A_rowC = to_torch_csr(self.A[I_row][:,I_C],device=rhs.device)
        else:
            A_Ccol = to_torch_csr(self.A[I_row][:,I_C].T.tocsr(),device=rhs.device)
            A_rowcol = to_torch_csr(self.A[I_col][:,I_row].T.tocsr(),device=rhs.device)
            A_rowC = to_torch_csr(self.A[I_C][:,I_col].T.tocsr(),device=rhs.device)

        rhs_flat = rhs.flatten(start_dim=0,end_dim=-2)

        tmp = (A_Ccol @ rhs_flat)
        tmp = self.solve_slab(Npan_ind,Npan_ind+1,tmp.unsqueeze(0),\
                             transpose=transpose).flatten(start_dim=0,end_dim=-2)
        result = - A_rowC @ tmp
        if ((rowcol[0] == 'L') and (rowcol[1] == 'L')): 
            return result + A_rowcol @ rhs_flat
        else:
            return result
    
    def apply_TXX_lowmem(self,Npan_ind,rhs,transpose=False):

        Npan_loc = self.Npan_loc; Npan = self.Npan
        xoffset = self.elim_nblocks * self.elim_bs * Npan_loc

        I_C    = self.inds_pans[Npan_ind,self.I_C].flatten()
        I_X    = torch.hstack((self.inds_pans[Npan_ind,self.I_L].flatten(),\
                               self.inds_pans[Npan_ind,self.I_R].flatten()))

        I_elim = self.inds_pans[Npan_ind,self.I_C[:xoffset]].flatten()
        I_rem  = self.inds_pans[Npan_ind,self.I_C[xoffset:]].flatten()

        device = rhs.device

        if (not transpose):
            Aelim_rem = self.A[I_elim][:,I_rem]
            Arem_elim = self.A[I_rem][:,I_elim]

            Ax_elim = self.A[I_X][:,I_elim]
            Aelim_x = self.A[I_elim][:,I_X]

            Ax_rem = to_torch_csr(self.A[I_X][:,I_rem].tocsr(),device=device)
            Arem_x = to_torch_csr(self.A[I_rem][:,I_X].tocsr(),device=device)
            
            if (Npan_ind < Npan-1):
                # only apply A_LL, applying A_XX leads to "double counting"
                # interactions on slab interfaces
                I_L    = self.inds_pans[Npan_ind,self.I_L]
                Axx    = self.A[I_L][:,I_L].tocsr()
                Axx.resize(I_X.shape[0],I_X.shape[0])
                Axx    = to_torch_csr(Axx,device=device)
            else:
                Axx = to_torch_csr(self.A[I_X][:,I_X].tocsr(),device=device)
        else:
            Arem_elim = self.A[I_elim][:,I_rem].T.tocsr()
            Aelim_rem = self.A[I_rem][:,I_elim].T.tocsr()

            Aelim_x = self.A[I_X][:,I_elim].T.tocsr()
            Ax_elim = self.A[I_elim][:,I_X].T.tocsr()

            Arem_x = to_torch_csr(self.A[I_X][:,I_rem].T.tocsr(),device=device)
            Ax_rem = to_torch_csr(self.A[I_rem][:,I_X].T.tocsr(),device=device)
            
            if (Npan_ind < Npan-1):
                # only apply A_LL, applying A_XX leads to "double counting"
                # interactions on slab interfaces
                I_L    = self.inds_pans[Npan_ind,self.I_L]
                Axx    = self.A[I_L][:,I_L].T.tocsr()
                Axx.resize(I_X.shape[0],I_X.shape[0])
                Axx    = to_torch_csr(Axx.tocsr(),device=device)
            else:
                Axx = to_torch_csr(self.A[I_X][:,I_X].T.tocsr(),device=device)

        Aelim_xrem = sp_hstack([Aelim_x,-Aelim_rem],format="csr")

        g = Arem_x @ rhs
        g -= self.elim_lowmem(Npan_ind*Npan_loc, (Npan_ind+1)*Npan_loc,\
                              Arem_elim,Aelim_x,rhs,transpose=transpose)

        u_rem = self.Ared_solver.lusolve_sweep(g.unsqueeze(0),b_start=Npan_ind,b_end=Npan_ind+1,\
                                             transpose=transpose)
        u_rem = u_rem.squeeze(0)

        tmp = Ax_rem @ u_rem
        tmp  += self.elim_lowmem(Npan_ind*Npan_loc,(Npan_ind+1)*Npan_loc,\
                                Ax_elim,Aelim_xrem,\
                                torch.vstack((rhs,u_rem)),transpose=transpose)

        sol_X = Axx @ rhs - tmp

        return sol_X
    
    
    def get_TXX_slabind(self,Npan_ind,rank,n_pad=0,todense=False,verbose=False):

        Lsize = self.I_L.shape[0]
        hbs = HBS_matrix(4,Lsize,rank,n_pad=n_pad)
        nsamples = hbs.bs + rank

        if (torch.cuda.is_available()):
            device=torch.device('cuda')
        else:
            device=torch.device('cpu')
        
        if (nsamples > Lsize):
            raise ValueError("num samples is greater than Lsize")
            
            
        Omega = torch.rand(Lsize,nsamples,device=device); 
        Psi   = torch.rand(Lsize,nsamples,device=device);

        Omega_sample = torch.block_diag(*Omega.repeat(2,1,1))
        tic = time()
        tmp = self.apply_TXX_lowmem(Npan_ind,\
                               Omega_sample)
        del Omega_sample

        Psi_sample = torch.block_diag(*Psi.repeat(2,1,1))
        tmp_trans = self.apply_TXX_lowmem(Npan_ind,\
                               Psi_sample,transpose=True)
        del Psi_sample
        toc_sample = time() - tic;

        Omega = Omega.unsqueeze(0); Psi = Psi.unsqueeze(0)
        Y = Omega.repeat(4,1,1)
        Z = Psi.repeat(4,1,1)

        Y[0] = tmp[:Lsize,:nsamples]       #Y_LL
        Y[1] = tmp[:Lsize,nsamples:]       #Y_LR
        Y[2] = tmp[Lsize:,:nsamples]       #Y_RL
        Y[3] = tmp[Lsize:,nsamples:]       #Y_RR

        Z[0] = tmp_trans[:Lsize,:nsamples] #Z_LL
        Z[2] = tmp_trans[:Lsize,nsamples:] #Z_RL
        Z[1] = tmp_trans[Lsize:,:nsamples] #Z_LR
        Z[3] = tmp_trans[Lsize:,nsamples:] #Z_RR

        # note that reconstruct function expects Omega.shape[0] == 1

        tic = time()
        hbs.reconstruct(Omega,Psi,Y,Z)
        toc_reconstruct = time() - tic;

        test_samples = 5
        if (verbose or (Npan_ind==0)):
            err = torch.linalg.norm(hbs.matvec(Omega[...,:test_samples]) - Y[...,:test_samples])\
                  / torch.linalg.norm(Y[...,:test_samples])
            err_trans = torch.linalg.norm(hbs.matvec(Psi[...,:test_samples],transpose=True) - Z[...,:test_samples])\
                  / torch.linalg.norm(Z[...,:test_samples])
        del Omega; del Psi; del Y; del Z

        if (todense):
            tic = time()
            result = hbs.todense(device).cpu()
            toc_todense = time() - tic
        else:
            result = hbs
            toc_todense = 0

        if (verbose or (Npan_ind==0)):
            # report timings
            print("\t--compressing T_%d with rank=%d and nsamples=%d" % (Npan_ind,rank,nsamples),
                  "(toc_sample,toc_reconstruct,toc_todense)",\
                  "(%5.2f, %5.2f, %5.2f) seconds\n \t with relerror (%5.2e,%5.2e)" \
                  % (toc_sample,toc_reconstruct,toc_todense,err,err_trans))
        return result


    ### assemble Treduced system in compressed form
    def assemble_build_Tred(self):

        if (self.disc =='fd'):
            buf = self.buf
            n_pad = 1
        elif (self.disc == 'hps'):
            buf = self.buf_pans * (self.hps.p-2)
            n_pad = 0
        
        n = self.I_L.shape[0]; Npan = self.Npan
        nwaves_thin = (self.kh / (2*np.pi)) * (buf/n)
        
        if (self.disc == 'fd'):
            rank = int(4*nwaves_thin)+int(buf/10)+20     # 4*nwaves_thin + 52
        elif (self.disc == 'hps'):
            rank = int(4*nwaves_thin)+int(1.5*self.hps.p)+30  # 4*nwaves_thin + 2*p + 10
            
        #### no periodic BC, assemble block tridiagonal matrix 
        if (not self.periodic_bc):
            Tslab = self.get_TXX_slabind(0,rank,n_pad=n_pad)

            # allocate diag
            hbs_diag_RR = copy_params_hbs(Tslab,Npan-1)
            hbs_diag_LL = copy_params_hbs(Tslab,Npan-1)

            # allocate sub and sup diag
            hbs_sub = copy_params_hbs(Tslab,Npan-2)
            hbs_sup = copy_params_hbs(Tslab,Npan-2)

            ## store relevant information from slab 0
            hbs_diag_RR.UVtensor[0] = Tslab.UVtensor[3]
            hbs_diag_RR.Dtensor[0]  = Tslab.Dtensor[3]

            for j in range(1,Npan-1):
                Tslab = self.get_TXX_slabind(j,rank,n_pad=n_pad)

                hbs_diag_LL.UVtensor[j-1] = Tslab.UVtensor[0]
                hbs_diag_LL.Dtensor[j-1]  = Tslab.Dtensor[0]

                hbs_sup.UVtensor[j-1] = Tslab.UVtensor[1]
                hbs_sup.Dtensor[j-1]  = Tslab.Dtensor[1]

                hbs_sub.UVtensor[j-1] = Tslab.UVtensor[2]
                hbs_sub.Dtensor[j-1]  = Tslab.Dtensor[2]

                hbs_diag_RR.UVtensor[j] = Tslab.UVtensor[3]
                hbs_diag_RR.Dtensor[j]  = Tslab.Dtensor[3]

            Tslab = self.get_TXX_slabind(Npan-1,rank,n_pad=n_pad)
            hbs_diag_LL.UVtensor[Npan-2] = Tslab.UVtensor[0]
            hbs_diag_LL.Dtensor[Npan-2]  = Tslab.Dtensor[0]
            
        #### periodic BC, assemble cyclic block tridiagonal matrix 
        else:
            Tslab = self.get_TXX_slabind(0,rank,n_pad=n_pad)

            # allocate diag
            hbs_diag_RR = copy_params_hbs(Tslab,Npan)
            hbs_diag_LL = copy_params_hbs(Tslab,Npan)

            # allocate sub and sup diag
            hbs_sub = copy_params_hbs(Tslab,Npan)
            hbs_sup = copy_params_hbs(Tslab,Npan)
            
            for j in range(Npan-1):
                hbs_diag_LL.UVtensor[j] = Tslab.UVtensor[0]
                hbs_diag_LL.Dtensor[j]  = Tslab.Dtensor[0]

                hbs_sup.UVtensor[j] = Tslab.UVtensor[1]
                hbs_sup.Dtensor[j]  = Tslab.Dtensor[1]

                hbs_sub.UVtensor[j+1] = Tslab.UVtensor[2]
                hbs_sub.Dtensor[j+1]  = Tslab.Dtensor[2]

                hbs_diag_RR.UVtensor[j+1] = Tslab.UVtensor[3]
                hbs_diag_RR.Dtensor[j+1]  = Tslab.Dtensor[3]
                
                Tslab = self.get_TXX_slabind(j+1,rank,n_pad=n_pad)
            
            # assign last slab
            hbs_diag_LL.UVtensor[Npan-1] = Tslab.UVtensor[0]
            hbs_diag_LL.Dtensor[Npan-1]  = Tslab.Dtensor[0]

            hbs_sup.UVtensor[Npan-1] = Tslab.UVtensor[1]
            hbs_sup.Dtensor[Npan-1]  = Tslab.Dtensor[1]

            hbs_sub.UVtensor[0] = Tslab.UVtensor[2]
            hbs_sub.Dtensor[0]  = Tslab.Dtensor[2]

            hbs_diag_RR.UVtensor[0] = Tslab.UVtensor[3]
            hbs_diag_RR.Dtensor[0]  = Tslab.Dtensor[3]
            

        return hbs_diag_RR, hbs_diag_LL, hbs_sub,hbs_sup