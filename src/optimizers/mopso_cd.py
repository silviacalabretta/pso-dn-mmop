import numpy as np
from tqdm import tqdm
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.operators.survival.rank_and_crowding.metrics import calc_crowding_distance

import numpy.typing as npt
from typing import Union, Optional

from .base_optimizers import BasePSO
from src.archives import Archive

class MOPSO_CD(BasePSO):
    def __init__(self, 
                 pop_size: int = 50, 
                 n_iter: int = 100, 
                 Q: int = 100,                  # maximum archive size
                 tourn_size: int = 2,           # tournament size for leader selection
                 w: Union[float, npt.NDArray] = 0.729,             
                 c1: float = 1.49445,                            
                 c2: float = 1.49445,                            
                 v_bounds: Optional[npt.ArrayLike] = None):      
        
        # initialize the BasePSO parameters
        super().__init__(pop_size, n_iter, w, c1, c2, v_bounds)
        
        # MOPSO specific parameters
        self.Q = Q
        self.tourn_size = tourn_size

    def optimize(self, problem):
        """
        The main execution loop for MOPSO_CD.
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
        
        # initialize personal bests
        pbest_pos = positions.copy()
        pbest_fit = fitnesses.copy()

        # initialize global repository
        best_front = nds.do(fitnesses, only_non_dominated_front=True)
        rep_archive = Archive(
            positions = list(positions[best_front]),
            fitnesses = list(fitnesses[best_front])
        )
        self._truncate_repository(rep_archive)
        
        hist = [positions.copy()]
        
        for _ in tqdm(range(self.n_iter), desc="Running MOPSO_CD"):
            
            # select leaders for each particle via tournament
            leaders = self._select_leaders(rep_archive)

            # update velocities and positions
            velocities = self._update_velocity(positions, velocities, pbest_pos, leaders)
            positions, velocities = self._update_position(positions, velocities, x_bounds)
            
            # evaluate new positions
            fitnesses = problem.evaluate_population(positions)
            
            # update personal bests
            self._update_pbests(pbest_pos, pbest_fit, positions, fitnesses)
            
            # update global repository
            for i in range(self.pop_size):
                rep_archive.update_with_dominance(positions[i], fitnesses[i])
            self._truncate_repository(rep_archive)
            
            hist.append(positions.copy())
            
        return rep_archive, hist

    def _select_leaders(self, rep_archive: Archive) -> np.ndarray:
        """
        Selects a leader for each particle using tournament selection over the repository.
        Particles in less crowded regions (higher CD) win the tournament.
        """
        rep_pos, rep_fit = rep_archive.extract_arrays()
        n_rep = len(rep_pos)
        
        # if repository only has 1 particle, it's the only choice
        if n_rep == 1:
            return np.tile(rep_pos[0], (self.pop_size, 1))
            
        # compute crowding distances for the current repository
        cd_scores = calc_crowding_distance(rep_fit)
        
        leaders = np.zeros((self.pop_size, rep_pos.shape[1]))
        
        for i in range(self.pop_size):
            # tournament selection: pick random indices from the repository
            tourn_indices = np.random.choice(n_rep, size=min(self.tourn_size, n_rep), replace=False)
            
            # the winner is the one with the highest crowding distance (most isolated)
            winner_idx = tourn_indices[np.argmax(cd_scores[tourn_indices])]
            leaders[i] = rep_pos[winner_idx]
            
        return leaders
    
    def _update_pbests(self, pbest_pos, pbest_fit, new_pos, new_fit):
        """
        Updates the personal best of each particle based on Pareto dominance.
        """
        for i in range(self.pop_size):
            current_best_fit = pbest_fit[i]
            current_new_fit = new_fit[i]
            
            dominates = np.all(current_new_fit <= current_best_fit) and np.any(current_new_fit < current_best_fit)
            is_dominated = np.all(current_best_fit <= current_new_fit) and np.any(current_best_fit < current_new_fit)
            
            # If new position dominates, replace. 
            # If mutually non-dominated, replace randomly to maintain search diversity.
            if dominates or (not is_dominated and np.random.rand() < 0.5):
                pbest_pos[i] = new_pos[i].copy()
                pbest_fit[i] = new_fit[i].copy()

    def _truncate_repository(self, rep_archive: Archive) -> None:
        """
        Enforces the limited size Q of the repository.
        If size > Q, removes particles with higher crowding distance.
        """
        rep_pos, rep_fit = rep_archive.extract_arrays()
        
        if len(rep_pos) <= self.Q:
            return
            
        # compute crowding distances
        cd_scores = calc_crowding_distance(rep_fit)
        
        # keep the least crowded
        keep_indices = np.argsort(cd_scores)[::-1][:self.Q]
        rep_archive.positions = [rep_pos[i] for i in keep_indices]
        rep_archive.fitnesses = [rep_fit[i] for i in keep_indices]