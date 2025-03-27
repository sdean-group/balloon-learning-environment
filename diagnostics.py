import json
import numpy as np
import matplotlib.pyplot as plt

# datapaths = [
#             "diagnostics/used_in_report/mpcagent-replanned-fixed-wind-field-no-noise.json",
#             # "diagnostics/used_in_report/mpcagent-replanned-fixed-wind-field-no-noise-initializations-up-to-19km.json",
#             "diagnostics/used_in_report/mpcagent-replanned-fixed-wind-field-with-noise.json"]

datapaths = [
            "diagnostics/mpc4agent-replanned-no-noise.json",
            "diagnostics/MPC4Agent-1743049127887.json",
            "diagnostics/used_in_report/mpcagent-replanned-fixed-wind-field-no-noise.json"]

agent = 'mpc_agent'

diagnostics = [ json.load(open(datapath, 'r')) for datapath in datapaths ]

# print('Seed | No Noise | With Noise | ∆')
for seed in diagnostics[0]:
    # noise_twr = diagnostics[0][seed]['rollout'][agent]['twr']
    # wind_twr = diagnostics[1][seed]['rollout'][agent]['twr']

    for i, diag in enumerate(diagnostics):
        if (diag[seed]['steps'] != 960):
            print(f'seed {seed} ran out of power')

    seed_int = int(seed)
    digits = (int(np.log10(seed_int)) if seed_int != 0 else 0)
    print(f"{seed_int}{' ' * (4 - digits)}", end="")


    for i, diag in enumerate(diagnostics):
        twr = diag[seed]['twr']
        print(f"| {twr:0.3f}    ", end="")

    print()

# exit()

prior_results = {
    0: {"MPC": (751.47, 0.699), "Perciatelli44": (667.76, 0.569), "StationSeeker": (657.21, 0.551)},
    2: {"MPC": (307.57, 0.193), "Perciatelli44": (383.92, 0.306), "StationSeeker": (338.72, 0.263)},
    3: {"MPC": (839.27, 0.802), "Perciatelli44": (959.21, 1.000), "StationSeeker": (957.76, 1.000)},
    4: {"MPC": (794.57, 0.814), "Perciatelli44": (958.93, 1.000), "StationSeeker": (897.54, 0.923)},
    5: {"MPC": (767.24, 0.724), "Perciatelli44": (672.77, 0.624), "StationSeeker": (713.11, 0.680)},
    8: {"MPC": (204.91, 0.124), "Perciatelli44": (952.58, 0.992), "StationSeeker": (952.56, 1.000)},
    20: {"MPC": (607.60, 0.484), "Perciatelli44": (809.70, 0.779), "StationSeeker": (659.90, 0.534)},
    21: {"MPC": (673.90, 0.613), "Perciatelli44": (907.40, 0.922), "StationSeeker": (747.31, 0.704)},
    22: {"MPC": (852.31, 0.851), "Perciatelli44": (805.74, 0.784), "StationSeeker": (767.27, 0.713)},
}
# diagnostics/MPCAgent-1742156446638.json
# diagnostics/MPCAgent-1742156748804.json
# datapath = "diagnostics/MPCAgent-1742157269044.json"
# datapath = "diagnostics/MPCAgent-1742158470954.json"
# datapath = "diagnostics/MPCAgent-1742159194286.json"
# datapath = "diagnostics/MPCAgent-1742160106991.json"
# datapath = "diagnostics/MPCAgent-1742161918509.json"
# datapath = "diagnostics/used_in_report/mpc4agent-no-replan-fixed-wind-field-no-wind-noise-240-steps.json"
datapath = "diagnostics/used_in_report/mpcagent-no-replan-fixed-wind-field-no-wind-noise-240-steps.json"
# datapath = "diagnostics/used_in_report/mpcagent-replanned-fixed-wind-field-no-noise-initializations-up-to-19km.json"
datapath = "diagnostics/MPC4Agent-1742759266647.json"
datapath = "diagnostics/MPC4Agent-1742761566816.json"
datapath = "diagnostics/MPCAgent-1742762525802.json"
datapath = "diagnostics/MPC4Agent-1742762127491.json"
# datapath = "diagnostics/used_in_report/mpcagent-replanned-fixed-wind-field-no-noise.json"
datapath = "diagnostics/MPC4Agent-1743049127887.json"
agent = 'mpc4_agent'
diagnostics = json.load(open(datapath, 'r'))

# perciatelli_datapath = "/Users/myles/Programming/sdean/balloon-learning-environment/Perciatelli44-1740594371922.json"
# perciatelli_diagnostics = json.load(open(perciatelli_datapath, 'r'))

fidelities = []
twrs = []

# for result, perciatelli_result in zip(diagnostics, perciatelli_diagnostics):
#     mpc_agent_z = np.array(result['diagnostic']['mpc_agent']['z'])
#     simulation_z = np.array(result['diagnostic']['simulator']['z'])

#     fidelity = np.linalg.norm(mpc_agent_z - simulation_z)
#     seed = result['seed']

#     twr_score = result['twr']/perciatelli_result['twr'] if perciatelli_result['twr'] != 0 else 1.0
#     if perciatelli_result['twr'] ==0: continue
#         # print(result['twr'], perciatelli_result['twr'])
#     reward_score = result['reward']/perciatelli_result['reward']
    
#     fidelities.append(fidelity)
#     twrs.append(twr_score)

#     # plt.plot(range(result['steps']+1), mpc_agent_z, label='mpc')
#     # plt.plot(range(result['steps']+1), simulation_z, label='simulator')
#     # plt.legend()
#     # plt.title(f'seed={seed}, twr_score={twr_score}, fidelity={fidelity}')
#     plt.show()

#     print(f"seed={seed}, reward_score={reward_score:.5}, twr_score={twr_score:.3}, fidelity={fidelity}")

for seed, result in diagnostics.items():
    if False:
        agent_x = np.array(result['rollout'][agent]['x'])
        agent_y = np.array(result['rollout'][agent]['y'])
        agent_z = np.array(result['rollout'][agent]['z'])

        agent_rollout = np.column_stack((agent_x, agent_y, agent_x))
        print(agent_rollout[0])

        simulation_x = np.array(result['rollout']['simulator']['x'])
        simulation_y = np.array(result['rollout']['simulator']['y'])
        simulation_z = np.array(result['rollout']['simulator']['z'])

        simulation_rollout = np.column_stack((simulation_x, simulation_y, simulation_z))

        # fix this calculation
        path_rollout_fidelity = np.sum(np.linalg.norm(agent_rollout - simulation_rollout,axis=1))
        path_xy_fidelity = np.sum(np.linalg.norm(np.column_stack((agent_x, agent_y)) - np.column_stack((simulation_x, simulation_y)), axis=1))
        path_z_fidelity = np.sum(np.linalg.norm(agent_z - simulation_z, axis=0))

        print('rollout fidelity', path_rollout_fidelity)
        print('xy fidelity', path_xy_fidelity)
        print('z fidelity', path_z_fidelity)
        input()
        continue 

    TIME = list(range(result['steps'] + 2))

    agent_z = np.array(result['rollout'][agent]['z'])
    simulation_z = np.array(result['rollout']['simulator']['z'])

    agent_x = np.array(result['rollout'][agent]['x'])
    simulation_x = np.array(result['rollout']['simulator']['x'])

    agent_y = np.array(result['rollout'][agent]['y'])
    simulation_y = np.array(result['rollout']['simulator']['y'])

    # if False:
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # First subplot (X dynamics)
    axs[0].plot(TIME, agent_x, label="Agent X", linestyle="--")
    axs[0].plot(TIME, simulation_x, label="Simulation X")
    axs[0].set_title(f'{seed=} Approximate Dynamics X vs Simulation X')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Position X')
    axs[0].legend()
    axs[0].grid(True)

    # Second subplot (Y dynamics)
    axs[1].plot(TIME, agent_y, label="Agent Y", linestyle="--")
    axs[1].plot(TIME, simulation_y, label="Simulation Y")
    axs[1].set_title(f'{seed=} Approximate Dynamics Y vs Simulation Y')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Position Y')
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()  # Adjust layout for better spacing
    plt.show()
    # plt.plot(range(result['steps']+1), mpc_agent_y, label='mpc x')
    # plt.plot(range(result['steps']+1), simulation_y, label='simulation x')
    # plt.legend()
    # plt.title('mpc-rollout y<->sim-rollout y')
    # plt.show()

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(agent_x, agent_y, agent_z, label='mpc rollout')
    ax.plot(simulation_x, simulation_y, simulation_z, label='sim rollout')
    ax.legend()
    plt.title(f'{seed=} rollouts')
    plt.show()

    plt.plot(agent_x, agent_y, label='mpc xy rollout')
    plt.plot(simulation_x, simulation_y, label='simulation xy rollout')
    plt.legend()
    plt.show()

    fidelity = np.linalg.norm(agent_z - simulation_z)
    seed = result['seed']
    perciatelli_result = prior_results[seed]['Perciatelli44']

    twr_score = result['twr']/perciatelli_result[1]# if perciatelli_result['twr'] != 0 else 1.0
    #if perciatelli_result['twr'] ==0: continue
        # print(result['twr'], perciatelli_result['twr'])
    reward_score = result['reward']/perciatelli_result[0]
    
    fidelities.append(fidelity)
    twrs.append(twr_score)

    if agent == 'mpc_agent':
        mpc_agent_plan = np.insert(mpc_agent_plan, 0, mpc_agent_plan[0])
        plt.plot(range(result['steps']+2), mpc_agent_plan, label='mpc plan')

    plt.plot(TIME, agent_z, label='mpc dynamics rollout')
    plt.plot(TIME, simulation_z, label='simulator rollout')
    plt.legend()
    # plt.title(f'seed={seed}, twr_score={twr_score}, fidelity={fidelity}')
    plt.show()

    if agent == 'mpc4_agent':
        plt.plot(range(result['steps']+1), np.array(result['rollout'][agent]['plan']), label='mpc plan')
        plt.title('mpc plan (-1 <-> +1)')
        plt.show()

    # print(f"seed={seed}, reward_score={reward_score:.5}, twr_score={twr_score:.3}, fidelity={fidelity}")




fidelities = np.array(fidelities)
twrs = np.array(twrs)

indices = np.argsort(fidelities)
plt.scatter(fidelities[indices], twrs[indices])
plt.plot([0,100], [1,1], 'r')
plt.yticks(np.linspace(0,10,20))
plt.show()