# Reproducing PSO with Dynamic Strategy for Multi-modal Multi-objective Location Optimization

## Project Overview
This repository contains an implementation of the Particle Swarm Optimization with Dynamic Strategy (PSO-DN) algorithm presented in the paper  [A Particle Swarm Optimization with Dynamic Strategy for Multi-modal Multi-objective Location Optimization Problem](https://ieeexplore.ieee.org/document/10294853). The project specifically reproduces the benchmark experiment proposed: solving a real-world map-based multi-modal multi-objective location optimization problem (MMOP). 

The primary goal of this reproduction is to visualize and verify the algorithm's ability to maintain population diversity and discover multiple disjointed Pareto optimal sets in a 2D decision space. Furthermore, this project expands by comparing the performance of PSO-DN against other classic multi-objective evolutionary algorithms, specifically **NSGA-III** and classic **MOPSO**, to evaluate solution quality and search efficiency.

## Getting Started

### Prerequisites
* Python 3.8+ (or your chosen language)
* Required libraries: `numpy`, `matplotlib` (for decision space visualizations)

### Installation
1. Clone the repository:
   ```bash
   git clone [https://github.com/yourusername/reproducing-pso-dn.git](https://github.com/yourusername/reproducing-pso-dn.git)
