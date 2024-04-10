# Latent_ISP

This is an implementation of 'Solving Inverse Obstacle Scattering Problem with Latent Surface Representations" by Junqing Chen, Bangti Jin and Haibo Liu. See the paper [here](https://arxiv.org/abs/2311.07187).

# Citing Latent_ISP
If you use Latent_ISP in your research, please cite the [paper](https://arxiv.org/abs/2311.07187):
```
@misc{chen2023solving,
	title={Solving Inverse Obstacle Scattering Problem with Latent Surface Representations}, 
	author={Junqing Chen and Bangti Jin and Haibo Liu},
	year={2023},
	eprint={2311.07187},
	archivePrefix={arXiv},
	primaryClass={math.NA}
}
```

# How to Use Latent_ISP

This is the repro-pack for the latent-ISP using bempp-cl and exafmm-t. It has the following structure:

- `logs`: a.

Following the steps below to run each case:

### 1. Install bempp-cl and exafmm-t

We suggest to use `conda` as the package manager and create a conda environment for this application.
``` bash
conda create --yes -n bempp python=3.8
conda install -n bempp --yes numpy scipy matplotlib numba pytest jupyter plotly git pip mpi4py pyyaml
conda install -n bempp --yes -c conda-forge pocl pyopencl meshio
```
Then activate this environment: `conda activate bempp`.

Next, we need to install exafmm-t's dependencies: OpenBLAS and FFTW3. Follow these commands to install them on Ubuntu:
``` bash
apt-get update
apt-get -y install libopenblas-dev libfftw3-dev
```
Modify the commands accordingly if you are running other Linux distributions or using another package manager.

Finally, install `bempp-cl` and `exafmm-t`:
``` bash
pip install git+git://github.com/bempp/bempp-cl@v0.2.2
pip install git+git://github.com/exafmm/exafmm-t
```

The installation will take several minutes on a normal workstation.

### 2. Clone this repo

Next, clone this repo locally, change directory to this `repro-pack` folder:
``` bash
git clone https://github.com/barbagroup/bempp_exafmm_paper.git
cd bempp_exafmm_paper/repro-pack 


### 4. Run/Sumbit the script
``` bash
> python far_field_plane.py
```



# Team
Junqing Chen, Bangti Jin, Haibo Liu

# Acknowledgements

# License
