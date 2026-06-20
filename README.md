# Reproducing PSO with Dynamic Strategy for Multi-modal Multi-objective Location Optimization

## Project Overview
This repository contains an implementation of the Particle Swarm Optimization with Dynamic Strategy (PSO-DN) algorithm presented in the paper  [A Particle Swarm Optimization with Dynamic Strategy for Multi-modal Multi-objective Location Optimization Problem](https://ieeexplore.ieee.org/document/10294853). The project specifically reproduces the benchmark experiment proposed: solving a real-world map-based multi-modal multi-objective location optimization problem (MMOP). 

The primary goal of this reproduction is to visualize and verify the algorithm's ability to maintain population diversity and discover multiple disjointed Pareto optimal sets in a 2D decision space. Furthermore, this project expands by comparing the performance of PSO-DN against other classic multi-objective evolutionary algorithms, specifically NSGA-III and classic MOPSO, to evaluate solution quality and search efficiency.

## Current Implementation Status
Currently, the core architecture is established. The repository includes:
* A modular definition of the Location Optimization Problem (generating or loading map facilities and infrastructure).
* The foundational multi-objective utilities (Non-dominated sorting, Special Crowding Distance).
* A base optimizer framework to easily slot in different algorithms.
* The full implementation of the **PSO-DN** algorithm, including its dynamic radius scaling and sub-population niching strategies.
* Specialized archive structures for maintaining Personal Best (PBA) and Global Best (GBA) solutions.
* Visualization tools to plot the Pareto front directly onto the simulated 2D city map.

## Repository Structure

```text
.
├── data/
│   └── test_map.py             # Contains hardcoded benchmark map data (facilities, roads, bounds)
├── notebook/
│   └── pso-dn-for-mmop.ipynb   # Interactive Jupyter notebook for step-by-step execution of the algorithm
├── src/
│   ├── optimizers/
│   │   ├── base_optimizers.py  # abstract base class with shared PSO physics and validation logic
│   │   └── pso_dn.py           # specific implementation of the PSO-DN algorithm
│   ├── archives.py             # dataclasses for handling dominance-based solution archives (PBA & GBA)
│   ├── mo_utils.py             # multi-objective mathematical utilities (SCD, front sorting)
│   ├── plot_utils.py           # matplotlib wrappers for rendering the map and solution points
│   └── problem.py              # LocationProblem class and objective evaluation logic
├── main.py                     # main orchestrator script to initialize and run the optimization
├── README.md                   # you're here!
└── requirements.txt            # python dependencies

```

## Getting Started

### Prerequisites

* Python 3.8+

### Installation

1. Clone the repository to your local machine:
```bash
git clone https://github.com/silviacalabretta/pso-dn-mmop
cd pso-dn-mmop

```

2. Install the required dependencies using `pip`:
```bash
pip install -r requirements.txt

```


### Usage

To run the optimization using the terminal, simply execute the `main.py` orchestrator script from the root directory of the project:

```bash
python main.py

```

This will initialize the map data, instantiate the PSO-DN algorithm, run the optimization loop, print the progress, and finally display a Matplotlib plot showing the discovered Pareto optimal solutions on the city map.

Alternatively, you can open the `notebook/pso-dn-for-mmop.ipynb` file in Jupyter Notebook or VS Code to explore the algorithm interactively.
