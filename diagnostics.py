import json
import numpy as np
import matplotlib.pyplot as plt

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


# datapath = "/Users/myles/Programming/sdean/balloon-learning-environment/MPCAgent-1740593147380.json"
# datapath = "/Users/myles/Programming/sdean/balloon-learning-environment/MPCAgent-1740595451265.json"
datapath = "/Users/myles/Programming/sdean/balloon-learning-environment/MPCAgent-1740595696619.json"
# datapath = "/Users/myles/Programming/sdean/balloon-learning-environment/MPCAgent-1740604720348.json"
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

for result in diagnostics:
    
    mpc_agent_z = np.array(result['diagnostic']['mpc_agent']['z'])
    simulation_z = np.array(result['diagnostic']['simulator']['z'])

    fidelity = np.linalg.norm(mpc_agent_z - simulation_z)
    seed = result['seed']
    perciatelli_result = prior_results[seed]['Perciatelli44']

    twr_score = result['twr']/perciatelli_result[1]# if perciatelli_result['twr'] != 0 else 1.0
    #if perciatelli_result['twr'] ==0: continue
        # print(result['twr'], perciatelli_result['twr'])
    reward_score = result['reward']/perciatelli_result[0]
    
    fidelities.append(fidelity)
    twrs.append(twr_score)

    plt.plot(range(result['steps']+1), mpc_agent_z, label='mpc')
    plt.plot(range(result['steps']+1), simulation_z, label='simulator')
    plt.legend()
    plt.title(f'seed={seed}, twr_score={twr_score}, fidelity={fidelity}')
    plt.show()

    print(f"seed={seed}, reward_score={reward_score:.5}, twr_score={twr_score:.3}, fidelity={fidelity}")




fidelities = np.array(fidelities)
twrs = np.array(twrs)

indices = np.argsort(fidelities)
plt.scatter(fidelities[indices], twrs[indices])
plt.plot([0,100], [1,1], 'r')
plt.yticks(np.linspace(0,10,20))
plt.show()