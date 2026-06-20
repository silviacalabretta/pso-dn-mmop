import numpy as np
from dataclasses import dataclass, field

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
    
    def update_with_dominance(self, new_position: np.ndarray, new_fitness: np.ndarray) -> None:
        """
        Updates the archive with a new solution, maintaining Pareto dominance.
        Assumes a MINIMIZATION problem.
        """
        # if the archive is empty, just add the new record
        if not self.fitnesses:
            self.add_record(new_position, new_fitness)
            return

        _, current_fitnesses = self.extract_arrays()

        # Check if the new solution is dominated by any existing solution
        is_dominated = any(
            np.all(current_fitness <= new_fitness) and 
            np.any(current_fitness < new_fitness)
            for current_fitness in self.fitnesses
        )
        if is_dominated:
            return

        # Check if the new solution dominates any existing solutions and remove them
        new_positions = []
        new_fitnesses = []

        for pos, fit in zip(self.positions, self.fitnesses):
            
            # if the current solution is not dominated, keep it
            if not (np.all(new_fitness <= fit) and np.any(new_fitness < fit)):
                new_positions.append(pos)
                new_fitnesses.append(fit)

        self.positions = new_positions
        self.fitnesses = new_fitnesses

        self.add_record(new_position, new_fitness)