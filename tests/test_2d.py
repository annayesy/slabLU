import pickle
import os
import torch
os.environ['LANG']='en_US.UTF-8'

torch.set_default_dtype(torch.double)  # Ensure all torch tensors are double precision for accuracy

def run_test_via_argparse(domain, box_xlim=1.0, box_ylim=1.0, periodic_bc=False):
    assembly_type = 'reduced_cpu'
    pde    = 'bfield_constant'
    bc     = 'free_space'
    disc_n = 400
    ppw    = 40

    p      = 22
    solver = 'slabLU'

    pickle_loc = 'tmp_test_file'

    s = 'python argparse_driver.py --n %d --pde %s --bc %s --pickle %s' % (disc_n,pde,bc,pickle_loc)

    s += ' --disc hps --p %d' % (p)    
    s += ' --domain %s' % (domain)
    s += ' --ppw %d' % (ppw)

    s += ' --solver %s' % (solver)
    s += ' --sparse_assembly %s' % (assembly_type)

    s += ' --box_xlim %f' % box_xlim
    s += ' --box_ylim %f' % box_ylim

    s += ' --disable_cuda'
    if (periodic_bc):
        s += ' --periodic_bc'

    r = os.system(s)
    if (r == 0):
        f = open(pickle_loc,"rb")

        _ = pickle.load(f)
        d = pickle.load(f)

        assert d['trueres_solve'] < 2e-5
        os.system('rm %s' % pickle_loc)
    else:
        raise ValueError("test failed")

def test_helm_poisson():
    run_test_via_argparse('square')

def test_helm_poisson_annulus():
    run_test_via_argparse('annulus')

def test_helm_poisson_curvyannulus():
    run_test_via_argparse('curvy_annulus',box_xlim=6.0, periodic_bc=True)
