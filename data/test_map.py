import numpy as np

x_bounds = [[0, 100], [0, 100]]
v_bounds = [[0.001, 1], [0.001, 1]]

facilities = {
    'ES': np.array([[3, 37], [42, 96], [45, 60], [50, 25], [83, 72],[98, 38]]),
    'HS': np.array([[40, 20], [51, 60], [95, 51]]),
    'CS': np.array([[10, 55], [15, 15], [15, 78], [15, 88], [20, 23],[20, 70], [32, 42], [35, 60], [40, 76], [52, 78],[52, 96], [55, 33], [75, 27]]),
    'RS': np.array([[17.5, 82.5], [55.5, 82.5], [94.5, 6.5]])
}

infrastructure = {
    'Roads': np.array([
    # Major Horizontals
    [[0, 7], [100, 7]],
    [[0, 5.5], [100, 5.5]],
    [[0, 58], [100, 58]],
    [[0, 56.5], [100, 56.5]],
    [[0, 94], [100, 94]],
    [[0, 92.5], [100, 92.5]],
    
    [[19.6, 74], [100, 74]],

    # Minor Horizontals [interrupted by the center block]
    [[0, 40], [34, 40]],
    [[42, 31], [76, 31]],
    
    # Major Verticals
    [[17, 0], [17, 100]],
    [[18.5, 0], [18.5, 100]],
    [[77, 7], [77, 58]],
    
    # Minor Verticals [interrupted by the center block]
    [[42, 7], [42, 31]],
    [[34, 40], [34, 58]],
    [[55.5, 58], [55.5, 82.5]],
    
    # Diagonals branching off the center block
    [[40, 58], [30, 74]],
    [[34, 40], [42, 31]], # Up-right diagonal
    ]),
    
    'Railways': [
        [(0, 82.5), (100,82.5)],
        [(0, 82.5), (17.5, 82.5), (55.5, 82), (83, 68), (94.5, 35), (94.35,6.5), (94.5, 0)] 
    ]
}

map_styles = {
    'ES': {'color': '#ff4c4c', 'marker': 'o', 'size': 80},
    'HS':       {'color': '#4c72ff', 'marker': 'o', 'size': 80},
    'CS':           {'color': "#00ff00", 'marker': 'o', 'size': 80},
    'RS':           {'color': '#ffcc00', 'marker': 'o', 'size': 80},
    'Roads':                   {'color': "#B6B5B5", 'linestyle': '-', 'linewidth': 7, 'alpha': 1},
    'Railways':                {'color': '#000000', 'linestyle': '--', 'linewidth': 2, 'alpha': 0.8, 'smooth': True}
}

REGIONS = {
    "R1": (10.0, 20.0, 75.0, 85.0),
    "R2": (45.0, 60.0, 20.0, 40.0),
    "R3": (30.0, 60.0, 55.0, 100.0),
}