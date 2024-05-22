# SlabLU: Two-Level Sparse Direct Solver for Elliptic PDEs
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.11238664.svg)](https://doi.org/10.5281/zenodo.11238664)
[![License](https://img.shields.io/github/license/annayesy/slabLU)](./LICENSE.md)
[![Top language](https://img.shields.io/github/languages/top/annayesy/slabLU)](https://www.python.org)
![Code size](https://img.shields.io/github/languages/code-size/annayesy/slabLU)
[![Latest commit](https://img.shields.io/github/last-commit/annayesy/slabLU)](https://github.com/annayesy/slabLU/commits/master)

## Overview
SlabLU is a sparse direct solver designed for solving linear systems that arise from the discretization of elliptic partial differential equations (PDEs). By decomposing the domain into thin "slabs" and employing a two-level approach, SlabLU enhances parallelization, making it well-suited for modern multi-core architectures and GPUs.

The solver leverages the $\mathcal{H}^2$-matrix structures that emerge during factorization and incorporates randomized algorithms to efficiently handle these structures. Unlike traditional multi-level nested dissection schemes that use hierarchical matrix techniques across varying front sizes, SlabLU focuses on fronts of approximately equal size, streamlining its application and tuning for heterogeneous computing environments.

## Key Features
- **Two-Level Parallel Scheme:** Optimized for parallel execution, particularly on systems with GPU support.
- **Efficient Handling of $\mathcal{H}^2$-Matrices:** Utilizes efficient matrix algebra for efficient computational performance.
- **Flexible Discretization Compatibility:** Supports a wide range of local discretizations, including high-order multi-domain spectral collocation methods.
- **High Performance:** Demonstrated ability to solve significant problems, such as a Helmholtz problem on a domain of size $1000 \lambda \times 1000 \lambda$ with $N=100M$, within 15 minutes to six correct digits.

<p align="center">
    <img src="https://github.com/annayesy/slabLU/blob/main/figures/picture_crystal.png" width="75%"/>
</p>

<div style="display: flex; justify-content: center;">
    <p style="width: 75%; text-align: center; font-size: 90%;">
        Figure 1: Solutions of variable-coefficient Helmholtz problem on square domain $\Omega$ with Dirichlet data given by $u \equiv 1$ on $\partial \Omega$ for various wavenumbers $\kappa.$ The scattering field is $b_{\rm crystal}$, which is a photonic crystal with an extruded corner waveguide. The crystal is represented as a series of narrow Gaussian bumps with separation $s=0.04$ and is designed to filter wave frequencies that are roughly $1/s$.
    </p>
</div>
<p align="center" >
	<img src="https://github.com/annayesy/slabLU/blob/main/figures/picture_curvy_annulus.png" width="75%" /> 
</p>
<div style="display: flex; justify-content: center;">
    <p style="width: 75%; text-align: center; font-size: 90%;">
        Figure 2: Solutions of constant-coefficient Helmholtz problem on curved domain $\Phi$ with Dirichlet data given by $u \equiv 1$ on $\partial \Phi$ for various wavenumbers $\kappa$.
    </p>
</div>

## Usage

To install SlabLU, clone the repository and install the conda environment in `slablu.yml`.

To test the framework on a benchmark problem, run
```
python argparse_driver.py --n 1000 --disc hps --p 22 --ppw 10 --domain square --pde bfield_constant --bc free_space --solver slabLU
```
which solves the constant-coefficient Helmholtz equation of size $100 \lambda \times 100 \lambda$ on a unit square with $N \approx 1\rm M$ with local polynomial order $p=22$.

To solve a set of example PDEs on curved domains, run
```
python generate_pictures.py
```
which generates the figures in [1].


## Associated Papers
[1] Yesypenko, Anna, and Per-Gunnar Martinsson. "SlabLU: A Two-Level Sparse Direct Solver for Elliptic PDEs." arXiv preprint arXiv:2211.07572 (2022).

[2] Yesypenko, Anna, and Per-Gunnar Martinsson. "GPU Optimizations for the Hierarchical Poincaré-Steklov Scheme." International Conference on Domain Decomposition Methods. Cham: Springer Nature Switzerland, 2022.
