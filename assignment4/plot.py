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
    ax.set_xticklabels(reversed(range(7)))
    ax.set_ylabel('Y')
    ax.set_zlabel('Q-values')
    ax.set_title('Q-values for different actions')
    ax.legend(handles=legend_handles)

    plt.show()


def plot_policy(Q):
    V = np.max(Q, axis=2)
    policy = np.argmax(Q, axis=2)

    # Create a grid
    x = np.arange(0, 5, 1)
    y = np.arange(0, 5, 1)

    # Create meshgrid
    X, Y = np.meshgrid(x, y)

    # Flatten meshgrid to create coordinates for arrows
    x_flat = X.flatten()
    y_flat = Y.flatten()

    # Width and height of the arrows
    dx = dy = 0.8

    # Create a plot
    fig, ax = plt.subplots(figsize=(8, 8))

    # Color the cells based on V-values
    V_max = np.max(V)
    V_min = np.min(V)
    norm = plt.Normalize(V_min, V_max)
    colors = plt.cm.viridis(norm(V))

    for yi in range(5):
        for xi in range(5):
            # Plot colored cells
            rect = plt.Rectangle((xi, yi), 1, 1, fill=True, color=colors[xi][yi])
            ax.add_patch(rect)


    # Plot arrows based on the policy
    for yi in range(5):
        for xi in range(5):
            action = policy[xi, yi]
            if action == 0:
                ax.quiver(xi + 0.5, yi + 0.125, 0, dy, angles='xy', scale_units='xy', scale=1, color='blue')
            elif action == 1:
                ax.quiver(xi + 0.5, yi + 1 - 0.125, 0, -dy, angles='xy', scale_units='xy', scale=1, color='blue')
            elif action == 2:
                ax.quiver(xi + 1 - 0.125, yi + 0.5, -dx, 0, angles='xy', scale_units='xy', scale=1, color='blue')
            elif action == 3:
                ax.quiver(xi + 0.125, yi + 0.5, dx, 0, angles='xy', scale_units='xy', scale=1, color='blue')


    ax.set_xlim(0, 5.001)
    ax.set_ylim(0, 5.001)
    ax.set_xticks(np.arange(0, 5, 1))
    ax.set_yticks(np.arange(0, 5, 1))
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    # plt.gca().invert_xaxis()  # Invert y-axis to match the grid representation

    plt.show()


if __name__ == '__main__':
    # Sample Q-values (replace with your actual Q-values)
    Q = np.random.rand(5, 5, 4)
    # plot_Q(Q)
    plot_policy(Q)