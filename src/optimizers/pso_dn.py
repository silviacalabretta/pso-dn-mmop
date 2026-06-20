import numpy as np
from tqdm import tqdm
from scipy.spatial import distance
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

import numpy.typing as npt
from typing import Union, Optional, List

from .base_optimizers import BasePSO
from src.archives import Archive 
from src.mo_utils import get_sorted_fronts_and_scd, scd_for_front


class PSODN(BasePSO):
    def __init__(self, 
                 pop_size: int = 50, 
                 n_iter: int = 100, 
                 R_l: float = 10.0,                                    
                 w: Union[float, npt.NDArray] = 0.8,             
                 c1: float = 1.49445,                            
                 c2: float = 1.49445,                            
                 v_bounds: Optional[npt.ArrayLike] = None):      
        
        # initialize the BasePSO parameters
        super().__init__(pop_size, n_iter, w, c1, c2, v_bounds)
        
        # PSO-DN specific parameter
        self.R_l = R_l

    def optimize(self, problem):
        """
        The main execution loop for PSO-DN.
        """
        D = problem.D
        x_bounds = problem.x_bounds

        self._validate_parameters(D, x_bounds)

        nds = NonDominatedSorting()

        # initialize positions and velocities
        positions = np.random.uniform(x_bounds[:,0], x_bounds[:,1], size=(self.pop_size, D))
        # signs = np.random.choice([-1, 1], size=(self.pop_size, D))
        # magnitudes = np.random.uniform(self.v_bounds[:, 0], self.v_bounds[:, 1], size=(self.pop_size, D))
        # velocities = signs * magnitudes    
        velocities = np.zeros((self.pop_size, D))

        # evaluate initial population
        fitnesses = problem.evaluate_population(positions)
        
        # initialise archives
        # personal best archives for each particle
        pba_archives = [
            Archive(
                positions=[positions[i].copy()], 
                fitnesses=[fitnesses[i].copy()]
            ) 
            for i in range(self.pop_size)
        ]

        # global best archive (all non-dominated solutions found so far)
        best_front = nds.do(fitnesses, only_non_dominated_front=True)
        gba_archive = Archive(
            positions = list(positions[best_front]),
            fitnesses = list(fitnesses[best_front])
        )
        
        hist = [positions.copy()]
        
        for _ in tqdm(range(self.n_iter), desc="Running PSO-DN"):
            
            sorted_fronts, _ = get_sorted_fronts_and_scd(positions, fitnesses, nds)

            # PSO-DN specific logic: SCD sorting, dynamic radius, getting nbest
            nbest = self._get_nbest(positions, sorted_fronts)
            pbest = self._get_pbest(pba_archives)

            # Call the shared physics from BasePSO
            velocities = self._update_velocity(positions, velocities, pbest, nbest)
            positions, velocities = self._update_position(positions, velocities, x_bounds)
            
            fitnesses = problem.evaluate_population(positions)
            
            self._update_archives(pba_archives, gba_archive, positions, fitnesses, nds)
            hist.append(positions.copy())
            
        return gba_archive, hist
        

    def _get_nbest(self, positions, sorted_fronts):
        """
        Implements the dynamic radius setting strategy and splits the 
        swarm into sub-populations using the connected components graph.
        
        Parameters
            positions: ndarray of shape (N, D) of particle positions
            R_l: float, the initial predefined radius
            sorted_fronts: list of 1D np arrays of indices, representing the fronts,
                        each sorted according to SCD
        Returns
            nbest: ndarray of shape (N,D) containing nbest_i for each individual
        """
        # pairwise distance matrix (N x N)
        dist_matrix = distance.cdist(positions, positions, metric='euclidean')
        
        current_radius = self._get_dynamic_radius(dist_matrix, sorted_fronts[0])
            
        _, nbest_idx = self._assign_subpopulations(dist_matrix, current_radius, sorted_fronts)
        
        return positions[nbest_idx]
    
    def _get_pbest(self, pba_archives: List[Archive]):
        """ 
        Parameters
            pba_archives: list[Archive] of the personal best positions
        Returns
            pbest: ndarray of shape (N,D) with positions of personal best
        """
        pbest = []

        for i in range(self.pop_size):

            # extract PBA_i
            pba_i = pba_archives[i]
            positions, fitnesses = pba_i.extract_arrays()

            # pbest is the individual with max scd
            scd_i = scd_for_front(positions, fitnesses)
            best_index = np.argmax(scd_i)
            pbest.append(positions[best_index])
            
        return np.array(pbest)
    
    def _update_archives(self, pba_archives: List[Archive], gba_archive: Archive, 
                         positions: np.ndarray, fitnesses: np.ndarray, 
                         nds: NonDominatedSorting) -> None:
        """
        Updates the personal and global best archives for the swarm.
        """
        N = len(positions)
        
        # update PBA_i for each individual
        for i in range(N):
            pba_archives[i].update_with_dominance(positions[i], fitnesses[i])

        # update GBA
        current_gba_pos, current_gba_fit = gba_archive.extract_arrays()
        combined_pos = np.vstack([current_gba_pos, positions])
        combined_fit = np.vstack([current_gba_fit, fitnesses])

        best_front = nds.do(combined_fit, only_non_dominated_front=True)

        gba_archive.positions = list(combined_pos[best_front])
        gba_archive.fitnesses = list(combined_fit[best_front])


    def _get_dynamic_radius(self, dist_matrix, non_dom_indices):
        """
        Implements the dynamic radius setting strategy
        
        Parameters
            dist_matrix: ndarray of shape (N, N) of euclidean distance among particles
            R_l: float, the initial predefined radius
            non_dom_indices: ndarray of shape (p,) of indices for Front 0
        Returns
            updated radius (float)
        """
        # number of neighbors within R_l for every individual (subtract 1 to exclude self)
        neighbors_count = np.sum(dist_matrix <= self.R_l, axis=1) - 1
            
        # find the two reference individuals
        idx_max = np.argmax(neighbors_count)
        nd_counts = neighbors_count[non_dom_indices]
        local_idx_min = np.argmin(nd_counts)
        idx_min_nd = non_dom_indices[local_idx_min]
        
        # temporarily replace diagonal for the two target particles with infinity
        dist_matrix[idx_max, idx_max] = np.inf
        dist_matrix[idx_min_nd, idx_min_nd] = np.inf
        
        # compute the closest distance
        d_h = np.min(dist_matrix[idx_max])
        d_s = np.min(dist_matrix[idx_min_nd])
        
        # restore the diagonal to 0.0
        dist_matrix[idx_max, idx_max] = 0.0
        dist_matrix[idx_min_nd, idx_min_nd] = 0.0
        
        # apply the logical condition
        if d_h < d_s:   # high density detected
            return dist_matrix[idx_max, idx_min_nd]
        else:
            return self.R_l
        
    def _assign_subpopulations(self, dist_matrix, current_radius, sorted_fronts):
        """
        Assigns particles to sub-populations using a greedy niching approach based on SCD.
        """
        N = dist_matrix.shape[0]
        sub_populations = np.full(N, -1)  # -1 means unassigned
        nbest_indices = np.full(N, -1)
        
        # Sort indices by SCD in descending order
        sorted_indices = np.concatenate(sorted_fronts)

        n_subpopulations = 0

        for idx in sorted_indices:
            if sub_populations[idx] == -1:  # If not yet assigned
        
                # find all particles within radius of this seed
                in_radius = dist_matrix[idx] <= current_radius
                unassigned = sub_populations == -1
                
                sub_populations[in_radius & unassigned] = n_subpopulations
                nbest_indices[in_radius & unassigned] = idx
                
                n_subpopulations += 1
        
        return sub_populations, nbest_indices