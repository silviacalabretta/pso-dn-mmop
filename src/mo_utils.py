import numpy as np
import numpy.typing as npt
from typing import List, Tuple
from pymoo.operators.survival.rank_and_crowding.metrics import calc_crowding_distance
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

def standard_cd(front: npt.NDArray) -> npt.NDArray:
    """
    Vectorized computation of crowding distance for a specific front of size F.
    Parameters
        front (ndarray of shape (F, D)): position matrix of the front particles
    Returns
        cd (ndarray of shape (F,)): crowding distances
    """
    F, D = front.shape
    cd = np.zeros(F)
    
    # if front has 2 or fewer items, they are boundaries, assign infinity
    if F <= 2:
        return np.full(F, np.inf)
        
    for i in range(D):
        # indices that would sort this dimension
        sorted_indices = np.argsort(front[:, i])
        sorted_vals = front[sorted_indices, i]
        
        # set boundaries to infinity
        cd[sorted_indices[0]] = np.inf
        cd[sorted_indices[-1]] = np.inf
        
        # compute the range (max - min) for normalization
        val_range = sorted_vals[-1] - sorted_vals[0]
        if val_range == 0:
            continue
            
        # update all intermediate points
        cd_updates = (sorted_vals[2:] - sorted_vals[:-2]) / val_range
        cd[sorted_indices[1:-1]] += cd_updates
        
    return cd


def scd_for_front(front_positions: npt.NDArray, front_fitnesses: npt.NDArray) -> npt.NDArray:
    """
    Computes SCD for a single front of size F.
    Parameters 
        front_positions (ndarray of shape (F,D))
        front_fitnesses (ndarray of shape (F,M))
    Returns
        scd (ndarray of shape (F,)): SCD values
    """
    # compute standard CD in both spaces (pymoo)
    CD_x = calc_crowding_distance(front_positions)
    CD_F = calc_crowding_distance(front_fitnesses)
    
    # CD_x = standard_cd(front_positions)
    # CD_F = standard_cd(front_fitnesses)
    
    # compute average CD (ignoring infinity boundaries and the case front size <= 2)
    finite_CD_x = CD_x[np.isfinite(CD_x)]
    finite_CD_F = CD_F[np.isfinite(CD_F)]
    
    CD_avg_x = np.mean(finite_CD_x) if len(finite_CD_x) > 0 else 0
    CD_avg_F = np.mean(finite_CD_F) if len(finite_CD_F) > 0 else 0
    
    # distinguish the case where one of the two CDs is above the average
    condition = (CD_x > CD_avg_x) | (CD_F > CD_avg_F)
    
    scd = np.where(condition, 
                   np.maximum(CD_x, CD_F), 
                   np.minimum(CD_x, CD_F))
     
    return scd


def get_sorted_fronts_and_scd(positions: npt.NDArray, fitnesses: npt.NDArray, nds: NonDominatedSorting) -> Tuple[List[npt.NDArray], npt.NDArray]:
    """
    Sorts the population into non-dominated fronts using pymoo, computes 
    the SCD for each front, and returns the sorted fronts and the SCD values.
    
    Parameters
        positions (ndarray of shape (N, D)): position matrix of the population
        fitnesses (ndarray of shape (N, M)): fitness matrix
        nds: an instantiated pymoo NonDominatedSorting object used to perform 
            the non-dominated sorting efficiently.
    Returns
        sorted_fronts (list of 1D ndarrays): each array represents a front, 
            decribed by integer indices in {0,..., N-1}; indices within each front 
            are sorted in descending order based on the SCD value
        global_scd_values (ndarray of shape (N,)): SCD values of each individual
    """
    N = len(positions)
    global_scd_values = np.zeros(N)
    
    # perform non-dominated sorting using pymoo
    fronts = nds.do(fitnesses) 
    
    sorted_fronts = []
    
    for front_indices in fronts:
        # extract the matrices for this specific front
        front_X = positions[front_indices]
        front_F = fitnesses[front_indices]
        
        # compute SCD for this front (using the function we wrote earlier)
        scd = scd_for_front(front_X, front_F)
        
        # map the calculated SCDs back to the global array
        global_scd_values[front_indices] = scd
        
        # sort the current front indices descending by their SCD value
        sorted_idx_by_scd = np.argsort(scd)[::-1]
        sorted_front = np.array(front_indices)[sorted_idx_by_scd]
        
        sorted_fronts.append(sorted_front)
        
    return sorted_fronts, global_scd_values