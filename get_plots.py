import json
from balloon_learning_environment.env.balloon.standard_atmosphere import Atmosphere
import jax
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from mpl_toolkits.mplot3d import Axes3D
import os


SEED = 22
REWARDS = []
ATMOSPHERE = Atmosphere(key=jax.random.key(seed=0))

def get_flight_path(filepath, seed=SEED):
    runs = json.loads(open(filepath).read())
    for run in runs:
        if run['seed'] == seed:
            # this way of tracking rewards is bad but i forgot to add it originally
            REWARDS.append((get_agent_name(filepath), run['cumulative_reward']))
            flight_path = []
            for data_point in run['flight_path']:
                flight_path.append([ 
                    data_point['x'], 
                    data_point['y'], 
                    ATMOSPHERE.at_pressure(data_point['pressure']).height.km ])
                
            flight_path = np.array(flight_path)
            return flight_path

    print('didnt find flight path')
    return np.array([])

def get_agent_name(filepath):
    return os.path.splitext(os.path.basename(filepath))[0]

def get_3d_fig_ax():
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Add labels
    rewards = ', '.join([f"{x}={y:.1f}" for x, y in REWARDS])
    fig.text(0.5, 0.01, f'Rewards: {rewards}', ha='center')
    ax.set_title(f"Agent Flight Path (seed={SEED})", fontsize=14)
    ax.set_xlabel("X Displacement (km)", fontsize=12)
    ax.set_ylabel("Y Displacement (km)", fontsize=12)
    ax.set_zlabel("Altitude (km)", fontsize=12)

    theta = np.linspace(0, 2 * np.pi, 100)
    x = 50 * np.cos(theta)
    y = 50 * np.sin(theta)
    z = np.full_like(x, 0)  # z-coordinates remain constant

    # Plot the circle
    ax.plot(x, y, z, color='red', linewidth=2, label='Station Radius')

    return fig, ax

def plot_flight_path(agent_name, flight_path, ax):
    # Example data (replace these with your actual (960, 3) arrays)
    X = flight_path[:, 0]  # Replace with your X-coordinates
    Y = flight_path[:, 1]  # Replace with your Y-coordinates
    Z = flight_path[:, 2]  # Replace with your Z-coordinates

    # Plot the trajectory
    ax.plot(X, Y, Z, label=f'{agent_name} Path', linewidth=2)


def animate_flight_path(named_flight_paths, fig, ax):

    colors = ['red', 'green', 'blue']
    i = 0

    # Create an empty line object
    line_map = []
    for agent_name, _ in named_flight_paths:
        line_pair = ax.plot3D([], [], [], linewidth=2,color=colors[i])[0], ax.plot3D([], [], [], linewidth=2, color=colors[i])[0]
        i= (i+1)%len(colors)
        line_map.append(line_pair)
    
    # line = ax.plot3D([], [], [], linewidth=2)[0]
    # line_proj = ax.plot3D([], [], [], linewidth=2)[0]

    # Function to update the plot for each frame
    def update(frame, named_flight_paths, line_map):
        for (_, flight_path), (line, line_proj) in zip(named_flight_paths, line_map):
            segment = flight_path[:frame]
            line.set_data(segment[:, 0], segment[:, 1])
            line.set_3d_properties(segment[:, 2])

            line_proj.set_data(segment[:, 0], segment[:, 1])
            line_proj.set_3d_properties(np.zeros_like(segment[:, 0]))
        return [ line, line_proj ]

    # Create the animation object
    ani = animation.FuncAnimation(
        fig, update, frames=len(named_flight_paths[0][1]),fargs=(named_flight_paths, line_map), interval=10, blit=True
    )
    
    # ax.legend(handles=[line for line_pair in line_map for line in line_pair], labels=[f'{agent_name} Path' for agent_name, _ in named_flight_paths])
    ax.legend(handles=[line_pair[0] for line_pair in line_map], labels=[f'{agent_name} Path' for agent_name, _ in named_flight_paths])

    return ani



filepaths = [
    'eval/new_eval/mpc.json',
    'eval/new_eval/perciatelli44.json',
    'eval/new_eval/station_seeker.json'
]

named_flight_paths = [ (get_agent_name(filepath), get_flight_path(filepath)) for filepath in filepaths ]

task = 'ANIMATE' # 'PLOT'

if task == 'ANIMATE':
    fig,ax = get_3d_fig_ax()
    ax.set_xlim([-200,200])
    ax.set_ylim([-200,200])
    ax.set_zlim([-1,23])
    # agent_name, flight_path = named_flight_paths[0]
    # for agent_name, flight_path in named_flight_paths:
    ani = animate_flight_path(named_flight_paths, fig, ax)
    # ax.legend(loc='best')
    ani.save('flight_paths.gif', fps=240) 
    plt.show()
elif task == 'PLOT':
    fig,ax = get_3d_fig_ax()
    for agent_name, flight_path in named_flight_paths:
        plot_flight_path(agent_name, flight_path, ax)
    ax.legend(loc='best')
    plt.show()

