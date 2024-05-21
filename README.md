# SlabLU: Two-Level Sparse Direct Solver for Elliptic PDEs

## Overview
SlabLU is a sparse direct solver designed for solving linear systems that arise from the discretization of elliptic partial differential equations (PDEs). By decomposing the domain into thin "slabs" and employing a two-level approach, SlabLU enhances parallelization, making it well-suited for modern multi-core architectures and GPUs.

The solver leverages the $\mathcal{H}^2$-matrix structures that emerge during factorization and incorporates randomized algorithms to efficiently handle these structures. Unlike traditional multi-level nested dissection schemes that use hierarchical matrix techniques across varying front sizes, SlabLU focuses on fronts of approximately equal size, streamlining its application and tuning for heterogeneous computing environments.

## Key Features
- **Two-Level Parallel Scheme:** Optimized for parallel execution, particularly on systems with GPU support.
- **Efficient Handling of $\mathcal{H}^2$-Matrices:** Utilizes efficient matrix algebra for efficient computational performance.
- **Flexible Discretization Compatibility:** Supports a wide range of local discretizations, including high-order multi-domain spectral collocation methods.
- **High Performance:** Demonstrated ability to solve significant problems, such as a Helmholtz problem on a domain of size $1000 \lambda \times 1000 \lambda$ with $N=100M$, within 15 minutes to six correct digits.

## Usage

To install SlabLU:

1. Clone the repository
2. In a conda environment, install `scipy`, `numpy`, and `pytorch` with GPU compatibility (if a GPU is available).
3. Install `matplotlib` for plotting utilities.

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
