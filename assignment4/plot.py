import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def plot_Q(Q):
    # Create a grid
    x = np.arange(0, 5, 1)
    y = np.arange(0, 5, 1)

    # Create a meshgrid
    X, Y = np.meshgrid(x, y)

    # Flatten the meshgrid to create coordinates for the bars
    x_flat = X.flatten()
    y_flat = Y.flatten()
    z_flat = np.zeros_like(x_flat)  # Z-coordinates for the base of the bars

    # Width, depth, and height of the bars
    dx = dy = 0.2  # Adjust the width to your preference

    # Create a 3D bar plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Color map for different actions
    colors = ['b', 'g', 'r', 'c']

    # Legend entries for different actions
    actions = ['up', 'down', 'left', 'right']

    for yi in reversed(range(5)):
        for xi in range(5):
            for ai in range(4):
                dz = Q[xi, yi, ai]
                ax.bar3d(xi + 0.2*ai, yi, z_flat[0], dx, dy, dz, shade=True, color=colors[ai] if dz > 0 else 'white')

    legend_handles = []
    for i in range(4):
        legend_handles.append(Line2D([0], [0], color=colors[i], label=actions[i]))

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Q-values')
    ax.set_title('Q-values for different actions')
    ax.legend(handles=legend_handles)

    plt.show()

if __name__ == '__main__':
    # Sample Q-values (replace with your actual Q-values)
    Q = np.random.rand(5, 5, 4)
    plot_Q(Q)