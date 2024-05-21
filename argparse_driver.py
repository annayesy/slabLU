import argparse
import torch
import numpy as np
from time import time
torch.set_default_dtype(torch.double)

from domain_driver import *
from built_in_funcs import *
import pickle
import os

################################# ARGUMENT PARSER ############################################################


parser = argparse.ArgumentParser("Call direct solver for 2D domain.")

parser.add_argument('--disc', type=str, required=True)
parser.add_argument('--p',type=int,required=False)
parser.add_argument('--n', type=int, required=True)

parser.add_argument('--pde', type=str, required=True)
parser.add_argument('--domain', type=str, required=True)
parser.add_argument('--box_xlim', type=float, required=False, default=1.0)
parser.add_argument('--box_ylim', type=float, required=False, default=1.0)

parser.add_argument('--bc', type=str, required=True)
parser.add_argument('--ppw',type=int, required=False)
parser.add_argument('--nwaves',type=float, required=False)

parser.add_argument('--solver',type=str,required=False)
parser.add_argument('--sparse_assembly',type=str,required=False, default='reduced_gpu')
parser.add_argument('--pickle',type=str,required=False)
parser.add_argument('--store_sol',action='store_true')
parser.add_argument('--disable_cuda',action='store_true')
parser.add_argument('--buf_constant',type=float,required=False)
parser.add_argument('--periodic_bc', action='store_true')

args = parser.parse_args()
n = args.n; disc = args.disc
box_geom = torch.tensor([[0,args.box_xlim],[0,args.box_ylim]])

if (args.ppw is not None):
    print("\n RUNNING PROBLEM WITH %s discretization with %d points. PDE is %s discretized at %d ppw with boundary %s on %s domain with periodic_bc=%s."\
          % (args.disc,args.n,args.pde,args.ppw,args.bc,args.domain,args.periodic_bc))
    print("boxgeom is",box_geom,"\n")
elif (args.nwaves is not None):
    print("\n RUNNING PROBLEM WITH %s discretization with %d points. PDE is %s discretized at %d nwaves with boundary %s on %s domain with periodic_bc=%s."\
          % (args.disc,args.n,args.pde,args.nwaves,args.bc,args.domain,args.periodic_bc))
    print("boxgeom is",box_geom,"\n")
else:
    print("RUNNING PROBLEM WITH %s discretization with %d points. PDE is %s with boundary %s on %s domain with periodic_bc=%s.\n\n"\
          % (args.disc,args.n,args.pde,args.bc,args.domain,args.periodic_bc))
    print("boxgeom is",box_geom,"\n")
    
if (args.disable_cuda):
    os.environ["CUDA_VISIBLE_DEVICES"]=""
    
print("CUDA available %s"%torch.cuda.is_available())
if (torch.cuda.is_available()):
    print("--num cuda devices %d"% torch.cuda.device_count())
    
if ((not torch.cuda.is_available()) and (args.sparse_assembly == 'reduced_gpu')):
    args.sparse_assembly = 'reduced_cpu'
    print("Changed sparse assembly to reduced_cpu")

##### set the PDE operator according to args.pde
if ((args.pde == 'poisson') and (args.domain == 'square')):
    if (args.ppw is not None):
        raise ValueError
    
    # laplace operator
    op = pdo.PDO_2d(pdo.ones,pdo.ones)
    kh = 0
    curved_domain = False

elif ( (args.pde).startswith('bfield')):
    ppw_set = args.ppw is not None
    nwaves_set = args.nwaves is not None
    
    if ((not ppw_set and not nwaves_set)):
        raise ValueError('oscillatory bfield chosen but ppw and nwaves NOT set')
    elif (ppw_set and nwaves_set):
        raise ValueError('ppw and nwaves both set')
    elif (ppw_set):
        nwaves = int(n/args.ppw)
    else:
        nwaves = args.nwaves
    kh = (nwaves+0.03)*2*np.pi+1.8;
    print("kh is %5.2f" % kh)
        
    
      
    if (args.pde == 'bfield_constant'):
        bfield = bfield_constant
    elif (args.pde == 'bfield_bumpy'):
        bfield = bfield_bumpy
    elif (args.pde == 'bfield_gaussian_bumps'):
        bfield = bfield_gaussian_bumps
    elif (args.pde == 'bfield_cavity'):
        bfield = bfield_cavity_scattering
    elif (args.pde == 'bfield_crystal'):
        bfield = bfield_crystal
    elif (args.pde == 'bfield_crystal_waveguide'):
        bfield = bfield_crystal_waveguide
    elif (args.pde == 'bfield_crystal_rhombus'):
        bfield = bfield_crystal_rhombus
    else:
        raise ValueError
        
    curved_domain = False
    if (args.domain == 'square'):
        
        def c(xx):
            return bfield(xx,kh)
        # var coeff Helmholtz operator
        op = pdo.PDO_2d(pdo.ones,pdo.ones,c=c) 
        
    elif (args.domain == 'curved'):
        
        op, param_map, \
        inv_param_map = pdo.get_param_map_and_pdo('sinusoidal', bfield, kh)
        curved_domain=True
        
    elif (args.domain == 'annulus'):
        
        op, param_map, \
        inv_param_map = pdo.get_param_map_and_pdo('annulus', bfield, kh)
        curved_domain=True
        
    elif (args.domain == 'curvy_annulus'):
        
        op, param_map, \
        inv_param_map = pdo.get_param_map_and_pdo('curvy_annulus', bfield, kh)
        curved_domain=True
    else:
        raise ValueError
    
else:
    raise ValueError
    
##### set the domain and discretization parameters
if (args.periodic_bc):
    assert disc == 'hps'

if (disc=='fd'):
    if (args.buf_constant is None):
        args.buf_constant = 0.6
    h = 1/n;
    dom = Domain_Driver(box_geom,op,\
                        kh,h,buf_constant=args.buf_constant)
    N = dom.fd.ns[0] * dom.fd.ns[1]
elif (disc=='hps'):
    if (args.p is None):
        raise ValueError('HPS selected but p not provided')
    if (args.buf_constant is None):
        args.buf_constant = 1.0
    p = args.p
    npan = n / (p-2); a = 1/(2*npan)
    dom = Domain_Driver(box_geom,op,\
                        kh,a,p=p,buf_constant=args.buf_constant,periodic_bc = args.periodic_bc)
    N = (p-2) * (p*dom.hps.n[0]*dom.hps.n[1] + dom.hps.n[0] + dom.hps.n[1])
else:
    raise ValueError

################################# BUILD OPERATOR ############################################################
    
build_info = dom.build(sparse_assembly=args.sparse_assembly,\
                       solver_type = args.solver, verbose=True)
build_info['N']    = N
build_info['n']    = n
build_info['disc'] = disc
build_info['buf']  = dom.buf if disc == 'fd' else dom.buf_pans * (p-2)
build_info['pde']  = args.pde
build_info['bc']   = args.bc
build_info['domain'] = args.domain
build_info['solver'] = args.solver
build_info['sparse_assembly'] = args.sparse_assembly
build_info['box_geom'] = box_geom
build_info['kh']   = kh 
build_info['periodic_bc'] = args.periodic_bc
if (disc == 'fd'):
    build_info['h'] = h
elif (disc == 'hps'):
    build_info['a'] = a
    build_info['p'] = p
    
    
################################# SOLVE PDE ############################################################

print("SOLVE RESULTS")
solve_info = dict()

if (args.bc == 'free_space'):
    assert args.pde == 'bfield_constant'
    ff_body = None; known_sol = True
    
    if (not curved_domain):
        uu_dir = lambda xx: uu_dir_func_greens(xx,kh)
    else:
        uu_dir = lambda xx: uu_dir_func_greens(param_map(xx),kh)
        
elif (args.bc == 'pulse'):
    ff_body = None; known_sol = False
    
    if (not curved_domain):
        uu_dir = lambda xx: uu_dir_pulse(xx,kh)
    else:
        uu_dir = lambda xx: uu_dir_pulse(param_map(xx),kh)
    
elif (args.bc == 'ones'):
    ff_body = None; known_sol = False
    
    ones_func = lambda xx: torch.ones(xx.shape[0],1)
    if (not curved_domain):
        uu_dir = lambda xx: ones_func(xx)
    else:
        uu_dir = lambda xx: ones_func(param_map(xx))  

elif (args.bc == 'mms'):
    
    if (args.pde == 'poisson'):
        assert kh == 0
        assert (not curved_domain)

        Lx = 4*np.pi; Ly = 1
        uu_dir  = lambda xx: uu_dir_func_mms(xx,Lx,Ly)
        ff_body = lambda xx: ff_body_func_mms(xx,Lx,Ly)
        known_sol = True
    else:
        raise ValueError
        
elif (args.bc == 'log_dist'):
    
    if (args.pde == 'poisson'):
        assert kh == 0
        assert (not curved_domain)

        uu_dir  = lambda xx: uu_dir_func_greens(xx,kh)
        ff_body = None
        known_sol = True
    else:
        raise ValueError
else:
    raise ValueError

if (args.solver == 'slabLU'):
    
    uu_sol,res, true_res,resloc_hps,toc_solve = dom.solve(uu_dir,ff_body,known_sol=known_sol)
    uu_sol,res, true_res,resloc_hps,toc_solve = dom.solve(uu_dir,ff_body,known_sol=known_sol)
    
    print("\t--Slab solver solved Ax=b residual %5.2e with known solution residual %5.2e and resloc_HPS %5.2e in time %5.2f s"\
          %(res,true_res,resloc_hps,toc_solve))
    solve_info['res_solve']        = res
    solve_info['trueres_solve']    = true_res
    solve_info['resloc_hps_solve'] = resloc_hps
    solve_info['toc_solve']        = toc_solve
    
elif (args.solver == 'superLU'):

    uu_sol,res, true_res,resloc_hps,toc_solve = dom.solve(uu_dir,ff_body,known_sol=known_sol)
    uu_sol,res, true_res,resloc_hps,toc_solve = dom.solve(uu_dir,ff_body,known_sol=known_sol)

    print("\t--SuperLU solved Ax=b residual %5.2e with known solution residual %5.2e and resloc_HPS %5.2e in time %5.2f s"\
          %(res,true_res,resloc_hps,toc_solve))
    solve_info['res_solve_superLU']            = res
    solve_info['trueres_solve_superLU']        = true_res
    solve_info['resloc_hps_solve_superLU']     = resloc_hps
    solve_info['toc_solve_superLU']            = toc_solve

else:

    uu_sol,res, true_res,resloc_hps,toc_solve = dom.solve(uu_dir,ff_body,known_sol=known_sol)
    uu_sol,res, true_res,resloc_hps,toc_solve = dom.solve(uu_dir,ff_body,known_sol=known_sol)

    print("\t--Builtin solver %s solved Ax=b residual %5.2e with known solution residual %5.2e and resloc_HPS %5.2e in time %5.2f s"\
          %(args.solver,res,true_res,resloc_hps,toc_solve))
    solve_info['res_solve_petsc']            = res
    solve_info['trueres_solve_petsc']        = true_res
    solve_info['resloc_hps_solve_petsc']     = resloc_hps
    solve_info['toc_solve_petsc']            = toc_solve
    
if (args.store_sol):
    print("\t--Storing solution")
    if (disc == 'fd'):
        XX = dom.fd.XX
    elif (disc == 'hps'):
        XX = dom.hps.xx_tot
    solve_info['xx']        = XX
    solve_info['sol']       = uu_sol
    
if (args.pickle is not None):
    file_loc = args.pickle
    print("Pickling results to file %s"% (file_loc))
    f = open(file_loc,"wb+")
    pickle.dump(build_info,f)
    pickle.dump(solve_info,f)
    f.close()