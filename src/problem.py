import numpy as np
import numpy.typing as npt
from scipy.spatial import distance
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Optional, Dict, List, Any
import copy


class LocationProblem():
    def __init__(self, 
                 D: int = 2, 
                 M: int = 4, 
                 K: Optional[List[int]] = None, 
                 x_bounds: Optional[npt.ArrayLike] = None,
                 facilities: Optional[Dict[str, npt.NDArray]] = None, 
                 infrastructure: Optional[Dict[str, npt.NDArray]] = None, 
                 map_styles: Optional[Dict[str, Any]] = None) -> None:
        """
        Initializes the Location Optimization Problem. 
        If 'facilities' is provided and not empty, D, M, and K are automatically inferred from the data.
        If 'facilities' is not provided, they are generated randomly based on D, M, K, and x_bounds.
        """
        self.x_bounds = np.asarray(x_bounds) if x_bounds is not None else np.array([[0, 100], [0, 100]])
        self.infrastructure = infrastructure if infrastructure is not None else {}
        self.map_styles = copy.deepcopy(map_styles) if map_styles is not None else {}

        if facilities:
            self.facilities = facilities
            self.M = len(facilities) 
            self.D = facilities[next(iter(facilities))].shape[1]
            self.K = [facilities[key].shape[0] for key in facilities]
        else:
            self.D = D
            self.M = M
            self.K = K if K is not None else [6, 3, 13, 3]
            self.facilities = self._generate_random_facilities()
        self._ensure_map_styles()

    def _generate_random_facilities(self)-> Dict[str, npt.NDArray]:
        """Helper method to randomly generate facility coordinates."""
        facilities = {}
        
        x_min = self.x_bounds[:, 0]
        x_max = self.x_bounds[:, 1]

        for i in range(self.M):
            # Fallback to the last K value if M > len(K)
            k_points = self.K[i] if i < len(self.K) else self.K[-1]
            
            key = f"Facility_{i+1}"
            
            facilities[key] = np.random.uniform(low=x_min, high=x_max, size=(k_points, self.D))
            
        return facilities
    
    def _ensure_map_styles(self):
        """
        Ensures every facility type has a corresponding entry in map_styles.
        Uses a discrete colormap to assign colors to dynamically generated facilities.
        """
        cmap = plt.colormaps.get_cmap('tab10') 
        
        # facility style generation
        facility_keys = list(self.facilities.keys())
        
        color_idx = 0
        
        for key in facility_keys:
            if key not in self.map_styles:
                rgba_color = cmap(color_idx % cmap.N)
                hex_color = mcolors.to_hex(rgba_color)
                
                self.map_styles[key] = {
                    'color': hex_color,
                    'marker': 'o',
                    'size': 80
                }
                color_idx += 1
                
        # infrastructure defaults
        if 'Roads' not in self.map_styles:
            self.map_styles['Roads'] = {'color': '#B6B5B5', 'linestyle': '-', 'linewidth': 7, 'alpha': 1}
        if 'Railways' not in self.map_styles:
            self.map_styles['Railways'] = {'color': '#000000', 'linestyle': '--', 'linewidth': 2, 'alpha': 0.8, 'smooth': True}



    # Fitness evaluation methods
    def single_fitness(self, x: npt.NDArray, key: str) -> float:
        """
        Single objective evaluation
    
        Parameters
            x (ndarray, shape (D,)): position of a single individual
            key (str): dictionary key representing the target facility type
        Returns
            (float): minimum Euclidean distance to the specified facility type
        """
        distances = np.linalg.norm(self.facilities[key] - x, axis=1)
        
        return float(np.min(distances))

    def multi_objective_fitness(self, position):
        """
        Computes the objective values across all facility types for a given position.
    
        Parameters
            position (ndarray, shape (D,)): position of a single individual
        Returns
            (ndarray, shape (M,)): minimum distance to each facility type
        """
        objectives = []
        
        # ensure the objectives order is always consistent
        facility_keys = sorted(self.facilities.keys())

        for key in facility_keys:
            
            # Euclidean distances from the current position to ALL stations of this type
            distances = np.linalg.norm(self.facilities[key] - position, axis=1)
            
            # the objective is the distance to the closest one
            min_distance = np.min(distances)
            objectives.append(min_distance)
            
        return np.array(objectives)

    def evaluate_population(self, pop: npt.NDArray) -> npt.NDArray:
        """
        Computes the objective values for an entire population via vectorization.
    
        Parameters
            pop (ndarray, shape (N, D)): position matrix of the population
        Returns
            (ndarray, shape (N, M)): fitness matrix, each row corresponds to an individual's objectives
        """
        fitness_scores = []
        
        # ensure the objectives order is always consistent
        facility_keys = sorted(self.facilities.keys())

        for key in facility_keys:
            # euclidean distance from every individual to every facility of this type
            dist_matrix = distance.cdist(pop, self.facilities[key], metric='euclidean')
            
            # the objective is the distance to the closest one
            nearest_distances = dist_matrix.min(axis=1)     # array of size N
            fitness_scores.append(nearest_distances)
            
        # fitness_score is a list of M arrays of size N
        # stack the list into an (N, M) matrix
        return np.column_stack(fitness_scores)