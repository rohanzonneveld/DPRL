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
    size = Q.shape[0]
    Q = np.rot90(Q, k=-1)
    V = np.max(Q, axis=2)
    policy = np.argmax(Q, axis=2)

    # Create a grid
    x = np.arange(0, size, 1)
    y = np.arange(0, size, 1)

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

    for yi in range(size):
        for xi in range(size):
            # Plot colored cells
            rect = plt.Rectangle((xi, yi), 1, 1, fill=True, color=colors[xi][yi])
            ax.add_patch(rect)


    # Plot arrows based on the policy
    for yi in range(size):
        for xi in range(size):
            action = policy[xi,yi]
            if action == 0:
                ax.quiver(xi + 0.5, yi + 0.125, 0, dy, angles='xy', scale_units='xy', scale=1, color='blue')
            elif action == 1:
                ax.quiver(xi + 0.5, yi + 1 - 0.125, 0, -dy, angles='xy', scale_units='xy', scale=1, color='blue')
            elif action == 2:
                ax.quiver(xi + 1 - 0.125, yi + 0.5, -dx, 0, angles='xy', scale_units='xy', scale=1, color='blue')
            elif action == 3:
                ax.quiver(xi + 0.125, yi + 0.5, dx, 0, angles='xy', scale_units='xy', scale=1, color='blue')


    ax.set_xlim(0, size + .001)
    ax.set_ylim(0, size + .001)
    ax.set_xticks(np.arange(0, size, 1))
    ax.set_yticks(np.arange(0, size, 1))
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    # plt.gca().invert_xaxis()  # Invert y-axis to match the grid representation

    plt.show()


if __name__ == '__main__':
    # Sample Q-values (replace with your actual Q-values)
    Q = np.array([[[ 0.07928495, -0.00489993,  0.13166167,  0.0733027 ],
        [ 0.08144136,  0.00388729,  0.13053668,  0.12134992],
        [ 0.07908121,  0.01306732,  0.12674724,  0.18632397],
        [ 0.07630678,  0.02297886,  0.12256768,  0.25101408],
        [ 0.07422139,  0.03444649,  0.11733194,  0.31147823]],

       [[ 0.07774519, -0.0035111 ,  0.1372105 ,  0.06403764],
        [ 0.0777008 , -0.0004918 ,  0.1334628 ,  0.06477413],
        [ 0.07748173,  0.00409822,  0.12958902,  0.06882215],
        [ 0.08073296,  0.01129997,  0.12951404,  0.08669795],
        [ 0.08096177,  0.02071451,  0.12776169,  0.14377822]],

       [[ 0.07824614,  0.00048413,  0.13876179,  0.06717221],
        [ 0.07766737,  0.00177081,  0.13599399,  0.06685109],
        [ 0.07773858,  0.00487702,  0.13269368,  0.06716148],
        [ 0.07738124,  0.00854503,  0.13002887,  0.06736344],
        [ 0.0767551 ,  0.0130818 ,  0.12749283,  0.06922897]],

       [[ 0.0787591 ,  0.00505221,  0.13986942,  0.07080293],
        [ 0.07777861,  0.00539472,  0.13772947,  0.07025304],
        [ 0.07794333,  0.00825682,  0.13419671,  0.07040598],
        [ 0.07761408,  0.01186022,  0.13141142,  0.07063185],
        [ 0.07721502,  0.0155503 ,  0.12871589,  0.070843  ]],

       [[ 0.07927206,  0.00962029,  0.14097704,  0.07443364],
        [ 0.07788503,  0.00904427,  0.13948838,  0.07368119],
        [ 0.07808589,  0.01171383,  0.13577974,  0.07363736],
        [ 0.07788865,  0.01515335,  0.13282467,  0.07389109],
        [ 0.07748959,  0.01884342,  0.13012916,  0.07410223]]])

    # plot_Q(Q)
    plot_policy(Q)