# np-match

Simulated total X-ray scattering from nanoparticles via the Debye equation.
This repository contains supporting code for the following publication:

```bibtex
@misc{frost_energetically_2026,
	title = {Energetically {Driven} {Structure} {Matching} for {Autonomous} {Total} {X}-ray {Scattering} {Experiments}},
	url = {http://arxiv.org/abs/2609.30852},
	doi = {10.48550/arXiv.2609.30852},
	publisher = {arXiv},
	author = {Frost, Emil J. P. and Karlsen, Martin A. and Jensen, Jonas H. and Moreno, Rodrigo and Lambertsen, Jørn and Quinson, Jonathan and Anker, Andy S. and Bagger, Alexander and Vegge, Tejs},
	month = sep,
	year = {2026},
}
```

The installable package (`src/np_match`) is divided into four parts:
1) **src/np_match/debye**: A forked minimal version of the [DebyeCalculator](https://github.com/FrederikLizakJohansen/DebyeCalculator). It is faster but lacks certain features. Importantly, only single-element structures are supported and, for the
BinnedSingleDebyeCalculator, gradients are not propagated to atomic positions.
2) **src/np_match/neighbors**: PyTorch-based neighbor list with CUDA support.
3) **src/np_match/mace**: ASE calculator with new neighbor list. Molecular dynamics
and geometry relaxation routines.
4) **src/np_match/workflow**: Functionality for structure mining workflows including
fitting experimental scattering data.

## Install

Nanoparticle construction, specifically [workflows/generate_particles/scripts/main.py](workflows/generate_particles/scripts/main.py), relies on a fork of the [WulffPack](https://gitlab.com/materials-modeling/wulffpack) module.
Install the WulffPack fork from the `size-scan` branch:

```bash
python -m pip install "git+https://gitlab.com/emilfrost/wulffpack.git@size-scan"
```


Install the package and core dependencies (from the directory of this README file).
```bash
python -m pip install .
```

To run the workflows, install optional dependencies.

```bash
python -m pip install ".[analysis,mace,workflow]"
```

## Workflows
The data generation (simulated scattering) is divided into four workflows.
1) **workflows/generate_particles**: All unique Wulff constructions and spherical 
nanoparticles in a size window are computed and saved. Takes less than a minute with a CUDA-accelerated GPU.
2) **workflows/ideal**: X-ray scattering from idealized particles from step 1)
is simulated and compared with experimental data.
3) **workflows/relax**: Idealized particles from step 1) are geometry relaxed
with a machine-learned interatomic potential. Scattering from the relaxed
structures is computed and compared with experimental data.
4) **workflows/md**: Relaxed particles from 3) are time-evolved with molecular
dynamics with a machine-learned interatomic potential. Scattering is averaged
from the final part of the trajectory and compared with experimental data.

Parts 2–4 use the [PerQueue workflow manager](https://gitlab.com/asm-dtu/perqueue)
and are intended to be submitted to CUDA-enabled GPUs on an HPC.

TOML configuration files are found under `workflows/{ideal,relax,md}` -- paths 
are relative to the config file. Configure HPC resources in these files. Workflows are intended to be submitted from
their `perqueue/` folder (output files end up here).
For example:

```bash
cd workflows/ideal/perqueue
python ../scripts/pq_driver.py
```

Analysis of the results is done in `workflows/{ideal,relax,md}/notebooks/analyse_output.ipynb`.

## Tests

```bash
python -m pip install ".[test]"
python -m pytest
```
