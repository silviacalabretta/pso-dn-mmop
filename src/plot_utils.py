import matplotlib.pyplot as plt
import copy


def smooth_path(path, iterations=4):
    """
    Smooths a path using Chaikin's corner-cutting algorithm.
    This strictly prevents 'wobbling' or overshooting.
    """
    if len(path) < 3:
        return path

    current_path = path
    # The more iterations, the smoother the curve. 4 or 5 is usually perfect.
    for _ in range(iterations):
        new_path = [current_path[0]] # Anchor the very first starting point
        
        for i in range(len(current_path) - 1):
            p0 = current_path[i]
            p1 = current_path[i+1]
            
            # Find the points 25% and 75% of the way along the line segment
            q = (0.75 * p0[0] + 0.25 * p1[0], 0.75 * p0[1] + 0.25 * p1[1])
            r = (0.25 * p0[0] + 0.75 * p1[0], 0.25 * p0[1] + 0.75 * p1[1])
            
            # Add those new points to shave off the corner
            new_path.extend([q, r])
            
        new_path.append(current_path[-1]) # Anchor the very last ending point
        current_path = new_path
        
    return current_path

def plot_city_map(point_data, line_data, styles, ax=None):
    """
    Plots a generic 2D map. Now supports automatic smoothing for lines.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
        standalone = True
    else:
        standalone = False

    # 1. Plot Lines (Roads, Railways)
    for category, paths in line_data.items():
        style = styles.get(category, {})
        color = style.get('color', 'black')
        linestyle = style.get('linestyle', '-')
        linewidth = style.get('linewidth', 2)
        alpha = style.get('alpha', 1.0)
        
        # Check if the style dictionary requests smoothing for this category
        smooth = style.get('smooth', False) 
        
        for i, path in enumerate(paths):
            if len(path) == 0:
                continue
            
            # Apply smoothing if requested
            plot_path = smooth_path(path) if smooth else path
            
            x_coords, y_coords = zip(*plot_path)
            label = category if i == 0 else ""
            
            ax.plot(x_coords, y_coords, color=color, linestyle=linestyle, 
                    linewidth=linewidth, alpha=alpha, label=label, zorder=1)

    # 2. Plot Points (Facilities)
    for category, points in point_data.items():
        if len(points) == 0:
            continue
            
        style = styles.get(category, {})
        color = style.get('color', 'black')
        marker = style.get('marker', 'o')
        size = style.get('size', 70)
        
        x_coords, y_coords = zip(*points)
        ax.scatter(x_coords, y_coords, color=color, marker=marker, s=size, 
                   edgecolors='black', linewidths=1.5, label=category, zorder=3)

    # 3. Environment & Styling
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    if line_data:
        ax.grid(False)
    else:
        ax.grid(True, linestyle=':', alpha=0.8, zorder=0)

    ax.set_aspect('equal')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title('Procedural City Infrastructure Map')
    
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)

    if standalone:
        plt.tight_layout()
        plt.show()


def plot_city_map_with_solutions(point_data, line_data, styles, solutions, label="Solutions", color="#ff00ff", marker="*", size=150, ax=None):
    """
    Wrapper around plot_city_map that temporarily injects algorithm solutions 
    (or population history) into the plotting dictionaries without mutating the originals.
    
    Parameters
        point_data: the facilities dictionary.
        line_data: the infrastructure dictionary.
        styles: The map_styles dictionary.
        solutions: ndarray of shape (N, D) representing the points to plot.
        label: The legend label for the new points.
        color, marker, size: Plotting style properties for the solutions.
    """
    # Create copies to prevent mutating the global dictionaries
    points_copy = copy.deepcopy(point_data)
    styles_copy = copy.deepcopy(styles)
    
    # Inject the new points and their styling
    points_copy[label] = solutions
    styles_copy[label] = {
        'color': color, 
        'marker': marker, 
        'size': size
    }
    
    # Call the existing plotting function
    plot_city_map(points_copy, line_data, styles_copy, ax=ax)