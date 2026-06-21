import numpy as np
from dataclasses import dataclass, field
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

@dataclass
class Archive:
    positions: list = field(default_factory=list)
    fitnesses: list = field(default_factory=list)

    def add_record(self, position: np.ndarray, fitness: np.ndarray) -> None:
        """Appends copies of the current position and fitness to the archive."""
        self.positions.append(position.copy())
        self.fitnesses.append(fitness.copy())

    def extract_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Converts the internal lists to NumPy arrays and returns them."""
        # This will return a shape of (N, D) for positions and (N, M) for fitnesses
        return np.array(self.positions), np.array(self.fitnesses)
    
    def update_with_dominance(self, new_positions: np.ndarray, new_fitnesses: np.ndarray, nds: NonDominatedSorting=None) -> None:
        """
        Updates the archive maintaining Pareto dominance.
        Accepts either a single solution (1D array) or a batch of solutions (2D array).
        Assumes a MINIMIZATION problem.
        """
        # normalize inputs so they are always 2D
        new_positions = np.atleast_2d(new_positions)
        new_fitnesses = np.atleast_2d(new_fitnesses)
        num_new = len(new_positions)

        ## single element evaluation
        if num_new == 1 and self.fitnesses:
            current_fit = np.array(self.fitnesses)
            new_fit = new_fitnesses[0]

            # Check if the new solution is dominated by any existing solution
            is_dominated = np.any(
                np.all(current_fit <= new_fit, axis=1) & np.any(current_fit < new_fit, axis=1)
            )
            if is_dominated:
                return 

            # Check if new solution dominates any existing solutions and remove them
            dominates = np.all(new_fit <= current_fit, axis=1) & np.any(new_fit < current_fit, axis=1)
            if np.any(dominates):
                keep_mask = ~dominates  
                current_pos = np.array(self.positions)
                
                self.positions = list(current_pos[keep_mask])
                self.fitnesses = list(current_fit[keep_mask])

            # append the new, non-dominated solution
            self.positions.append(new_positions[0].copy())
            self.fitnesses.append(new_fit.copy())
            return
        
        ## multiple elements (or empty archive)
        # combine with existing archive if it is not empty
        if len(self.positions) > 0:
            current_pos, current_fit = self.extract_arrays()
            combined_pos = np.vstack([current_pos, new_positions])
            combined_fit = np.vstack([current_fit, new_fitnesses])
        else:
            # if the archive is empty, we just evaluate the new batch
            combined_pos = new_positions
            combined_fit = new_fitnesses

        if not nds:
            nds = NonDominatedSorting()
        best_front = nds.do(combined_fit, only_non_dominated_front=True)

        self.positions = list(combined_pos[best_front])
        self.fitnesses = list(combined_fit[best_front])
        

    def remove_similar_solutions(self, decimals: int = 4) -> None:
        """
        Removes solutions that are geographically too close (similar positions)
        up to a specified number of decimal places.
        """
        if not self.positions:
            return

        positions_arr, _ = self.extract_arrays()
        pos_rounded = np.round(positions_arr, decimals)

        # find the indices of the unique positions
        _, unique_indices = np.unique(pos_rounded, axis=0, return_index=True)

        # sort indices to maintain the original insertion order (optional but clean)
        unique_indices = np.sort(unique_indices)

        self.positions = [self.positions[i] for i in unique_indices]
        self.fitnesses = [self.fitnesses[i] for i in unique_indices]
