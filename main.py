import numpy as np
from src.problem import LocationProblem
from src.optimizers.pso_dn import PSODN
from src.optimizers.mopso_cd import MOPSO_CD
from src.plot_utils import plot_city_map_with_solutions
from data.test_map import facilities, infrastructure, map_styles, x_bounds, v_bounds

def main():
    # define general parameters
    D = 1
    M = 6
    K = [6, 3, 13, 3]  
    # x_bounds = [[0, 100], [0, 100]]
    # v_bounds = [[0.001, 1], [0.001, 1]]  

    # instantiate the problem
    print("Initializing the Location Optimization Problem...")
    problem = LocationProblem(
        D=D, 
        M=M, 
        K=K, 
        facilities=facilities,
        infrastructure=infrastructure,
        map_styles=map_styles,
        x_bounds=x_bounds
    )

    # instantiate the optimizer

    # print("Initializing the PSO-DN Optimizer...")
    # optimizer = PSODN(
    #     pop_size=100, 
    #     n_iter=200, 
    #     R_l=10.0,       
    #     w=0.729,          
    #     c1=1.49445, 
    #     c2=1.49445, 
    #     v_bounds=v_bounds
    # )

    print("Initializing the MOPSO_CD Optimizer...")
    optimizer = MOPSO_CD(
        pop_size=100, 
        n_iter=200, 
        Q=500,
        tourn_size = 2,       
        w=0.8,          
        c1=1.49445, 
        c2=1.49445, 
        v_bounds=v_bounds
    )

    # run the optimization
    print("Starting optimization loop...")
    gba_archive, history = optimizer.optimize(problem)

    # extract Results
    best_positions, best_fitnesses = gba_archive.extract_arrays()
    print(f"\nOptimization complete! Found {len(best_positions)} Pareto optimal solutions.")

    # plot the Final Results
    print("Generating Map...")
    plot_city_map_with_solutions(
        point_data=problem.facilities, 
        line_data=problem.infrastructure, 
        styles=problem.map_styles, 
        solutions=best_positions,
        label="Pareto Front (GBA)",
        color="#00ffff",  # Magenta stars so they pop against the map
        marker="o",
        size=20
    )

if __name__ == "__main__":
    main()