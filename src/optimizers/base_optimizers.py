from abc import ABC, abstractmethod
import numpy as np
import numpy.typing as npt
from typing import Optional, Union, Tuple, Any

class BaseOptimizer(ABC):
    def __init__(self, pop_size: int, n_iter: int):
        """
        Universal parameters shared by ALL optimizers (PSO, NSGA, etc.)
        """
        self.pop_size = pop_size        # population size
        self.n_iter = n_iter            # number of iterations to run

    @abstractmethod
    def optimize(self, problem) -> Tuple[Any, list]:
        """
        Every optimizer MUST implement this method. 
        It takes a Problem instance and returns the final Archive and historical positions.
        """
        pass


class BasePSO(BaseOptimizer):
    def __init__(self, pop_size: int, n_iter: int, 
                 w: Union[float, npt.NDArray], 
                 c1: float = 1.49445, 
                 c2: float = 1.49445, 
                 v_bounds: Optional[npt.ArrayLike] = None):
        
        # initialize the parent class
        super().__init__(pop_size, n_iter)
        
        # PSO-specific parameters
        self.w = w                  # intertia parameter (either a float or an array)
        self.c1 = c1                # cognitive parameter
        self.c2 = c2                # social parameter
        self._base_v_bounds = np.asarray(v_bounds) if v_bounds is not None else np.array([0.001, 1])

    def _update_velocity(self, positions, velocities, pbest, leader):
        """
        Standard velocity update equation shared by all PSO variants.
        'leader' can be gbest (for classic PSO) or nbest (for PSO-DN).
        """
        N, D = positions.shape

        r1 = np.random.random((N, D))
        r2 = np.random.random((N, D))
        
        inertia = self.w * velocities
        cognitive = self.c1 * r1 * (pbest - positions)
        social = self.c2 * r2 * (leader - positions)
        
        new_velocities = inertia + cognitive + social
        
        # Enforce velocity bounds
        v_min, v_max = self.v_bounds[:, 0], self.v_bounds[:, 1]
        magnitudes = np.abs(new_velocities)
        signs = np.sign(new_velocities)
        clipped_magnitudes = np.clip(magnitudes, v_min, v_max)
        
        return signs * clipped_magnitudes


    def _update_position(self, positions, velocities, x_bounds):
        """
        Standard position update and damping boundary enforcement.
        """
        new_positions = positions + velocities
        
        x_min, x_max = x_bounds[:, 0], x_bounds[:, 1]
        violations = (new_positions < x_min) | (new_positions > x_max)
        
        new_positions = np.clip(new_positions, x_min, x_max)
        
        # dampen velocities on boundaries
        damping_factors = -np.random.random(positions.shape)
        velocities[violations] *= damping_factors[violations]
        
        return new_positions, velocities
    
    
    def _validate_parameters(self, D: int, x_bounds: np.ndarray) -> None:
        """
        Validates that all hyperparameters and bounds match the problem dimension D,
        and enforces logical constraints (e.g., lower bound < upper bound).
        """
        
        # population and iterations
        if not isinstance(self.pop_size, int) or self.pop_size <= 0:
            raise ValueError(f"pop_size must be a positive integer, got {self.pop_size}")
        if not isinstance(self.n_iter, int) or self.n_iter <= 0:
            raise ValueError(f"n_iter must be a positive integer, got {self.n_iter}")

        # x_bounds (from the problem)
        if x_bounds.shape != (D, 2):
            raise ValueError(f"x_bounds must have shape ({D}, 2), but got {x_bounds.shape}")
        if np.any(x_bounds[:, 0] >= x_bounds[:, 1]):
            raise ValueError("In x_bounds, the lower bound must be strictly less than the upper bound.")

        # dynamically build v_bounds
        if self._base_v_bounds.ndim == 1 or self._base_v_bounds.shape[0] == 1:
            # if _base_v_bounds is a 1D array, tile it to be D x 2
            self.v_bounds = np.tile(self._base_v_bounds, (D, 1))
        else:
            self.v_bounds = self._base_v_bounds

        # v_bounds (from the optimizer)
        if self.v_bounds.shape != (D, 2):
            raise ValueError(f"v_bounds must have shape ({D}, 2), but got {self.v_bounds.shape}")
        if np.any(self.v_bounds[:, 0] >= self.v_bounds[:, 1]):
            raise ValueError("In v_bounds, the lower bound must be strictly less than the upper bound.")

        # inertia 'w'
        if isinstance(self.w, (float, int)):
            self.w = self.w * np.ones(D)
        else:
            self.w = np.asarray(self.w)
            if self.w.shape != (D,):
                raise ValueError(f"Inertia 'w' must be a scalar or an array of shape ({D},), but got {self.w.shape}")

        # learning factors
        if self.c1 < 0 or self.c2 < 0:
            raise ValueError("Cognitive (c1) and Social (c2) parameters must be non-negative.")