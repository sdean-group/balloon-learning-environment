import json
import numpy as np
import matplotlib.pyplot as plt

datapath = "diagnostics/MPC4Agent-1741295398840.json"
agent = 'mpc4_agent'
diagnostics = json.load(open(datapath, 'r'))

divergence_threshold = 0.1
highlight_interval = 48

for seed, result in diagnostics.items():
    agent_winds = np.array(result['rollout'][agent]['wind'])
    simulation_winds = np.array(result['rollout']['simulator']['wind'])
    
    timesteps = range(len(agent_winds))

    divergence = np.linalg.norm(agent_winds - simulation_winds, axis=1)
    diverged_timesteps = timesteps[divergence > divergence_threshold]
    
    if diverged_timesteps.size > 0:
        first_divergence_timestep = diverged_timesteps[0]
        highlighted_ticks = np.arange(first_divergence_timestep, len(agent_winds), highlight_interval)
        print(f"Seed {seed} first divergence at timestep {first_divergence_timestep}")
        print(f"Highlighting every {highlight_interval} timesteps from {first_divergence_timestep}: {highlighted_ticks}")
    else:
        first_divergence_timestep = None
        highlighted_ticks = []
        
    magnitude_agent = np.linalg.norm(agent_winds, axis=1)
    magnitude_simulation = np.linalg.norm(simulation_winds, axis=1)
    
    plt.plot(timesteps, magnitude_agent, label='Agent Wind')
    plt.plot(timesteps, magnitude_simulation, label='Simulation Wind')
    
    if first_divergence_timestep is not None:
        plt.scatter(highlighted_ticks, magnitude_agent[highlighted_ticks], color='red', marker='D', label="Highlighted Points")
        for x in highlighted_ticks:
            plt.text(x, magnitude_agent[x], str(x), fontsize=10, verticalalignment='bottom', horizontalalignment='right')
            
    plt.xlabel('Timesteps')
    plt.ylabel('Wind Magnitude')
    # plt.xscale()
    plt.legend()
    plt.show()
    
    theta_agent = np.arctan2(agent_winds[:, 1], agent_winds[:, 0])
    theta_simulation = np.arctan2(simulation_winds[:, 1], simulation_winds[:, 0])
    
    plt.plot(timesteps, theta_agent, label='Agent Wind')
    plt.plot(timesteps, theta_simulation, label='Simulation Wind')

    if first_divergence_timestep is not None:
        plt.scatter(highlighted_ticks, theta_agent[highlighted_ticks], color='red', marker='D', label="Highlighted Points")
        for x in highlighted_ticks:
            plt.text(x, theta_agent[x], str(x), fontsize=10, verticalalignment='bottom', horizontalalignment='right')
            
    plt.xlabel('Timesteps')
    plt.ylabel('Wind Angle')
    plt.legend()
    plt.show()
    
