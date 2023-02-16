# MoReVis : A Visual Summary for Spatiotemporal Moving Regions

MoReVis is a visualization technique that summarizes spatiotemporal data that presents a region in space that evolves through time. In this repository is the code that implements the technique and the implementation of visualizations tools that make use of MoReVis.

## Prerequisites

- We developed MoReVis with [Anaconda](https://www.anaconda.com/) using Python. On a terminal with conda, create an env with the packages used in MoReVis with the `enviroment.yml` file.

```bash
conda env create -f enviroment.yml
```

- Then activate the environment to run Notebooks and scripts:

```bash
conda activate morevis
```

- MoReVis also made use of a solver; it can be [GUROBI](https://www.gurobi.com/downloads/end-user-license-agreement-academic/) or [MOSEK](https://www.mosek.com/products/academic-licenses/). You need to ask for a free academic license and install it in your conda env.

```bash
conda install -c gurobi gurobi
```

or 

```bash
conda install -c mosek mosek
``` 

- And then follow the steps on the site [GUROBI](https://www.gurobi.com/features/academic-named-user-license/)/[MOSEK](https://docs.mosek.com/latest/licensing/quickstart.html) to activate your license.

- The preprocessed data is available in `data/processed`, and the hurricane is available at `data/hurdat`; however, if you want to apply the preprocessing steps, download the WILDTRACK data in this [link](https://www.epfl.ch/labs/cvlab/data/data-wildtrack/) (download the annotated dataset) and extract it in the folder `data/wildtrack`.

## Using MoReVis

- Before running notebooks, create the folders `notebooks/metrics-results` and `notebooks/plots`. 
- For preprocessing the datasets, run the notebooks `wildtrack.ipynb`, `hurdat.ipynb`, and `synthetic_data.ipynb`.
- For running evaluations, run the notebooks `evaluate_optimization.ipynb`, `evaluate_projections.ipynb`, `motionlines_comparison.ipynb`.
- You can run the visualization interface. It is necessary first to download the WILDTRACK dataset and run the preprocessing. After, inside the file `app/` run the commands:

```bash
set FLASK_APP = application.py # windows

export FLASK_APP = application.py # ubuntu

python -m flask run
```

- And open your browser in the URL `http://127.0.0.1:5000/index`.

## LICENSE

Distributed under the GNU v3 License. See LICENSE for more information.