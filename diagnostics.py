import json
import numpy as np
import matplotlib.pyplot as plt

datapaths = [
            "diagnostics/used_in_report/mpcagent-replanned-fixed-wind-field-no-noise.json",
            # "diagnostics/used_in_report/mpcagent-replanned-fixed-wind-field-no-noise-initializations-up-to-19km.json",
            "diagnostics/used_in_report/mpcagent-replanned-fixed-wind-field-with-noise.json"]

agent = 'mpc_agent'

diagnostics = [ json.load(open(datapath, 'r')) for datapath in datapaths ]

print('Seed | No Noise | With Noise | ∆')
for seed in diagnostics[0]:
    # noise_twr = diagnostics[0][seed]['rollout'][agent]['twr']
    # wind_twr = diagnostics[1][seed]['rollout'][agent]['twr']

    noise_twr = diagnostics[0][seed]['twr']
    wind_twr = diagnostics[1][seed]['twr']

    seed = int(seed)
    digits = (int(np.log10(seed)) if seed != 0 else 0)
    print(f"{seed}{' ' * (4 - digits)}| {noise_twr:0.3f}    | {wind_twr:0.3f}      | {noise_twr-wind_twr:0.3f}")

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
# datapath = "diagnostics/mpcagent-replanning.json"
# datapath = "diagnostics/mpcagent-no-plan.json"
# datapath = "diagnostics/mpcagent-no-plan-with-noise.json"
# datapath = "diagnostics/mpcagent-no-plan-with-more-noise.json"
# datapath = "diagnostics/MPCAgent-1740776475969.json"
# datapath = "diagnostics/mpcagent-no-plan-no-replan.json"
# datapath = "diagnostics/mpcagent-no-plan-no-replan.json"
# datapath = "diagnostics/mpcagent-replanning-agressive-control.json"
# datapath = "diagnostics/mpc4-debug.json"
# datapath = "diagnostics/mpcagent-wind-error-reset-every-23.json"
# datapath = "diagnostics/mpcagent-wind-error.json"
# datapath = "diagnostics/mpcagent-no-wind-noise-deadreckon-reset-every-23.json"
# datapath = "diagnostics/deadreckon-with-correct-wind.json"
# datapath = "diagnostics/mpcagent-no-wind-noise-deadreckon-FIXED.json"
# datapath = "diagnostics/MPCAgent-1740947274453.json"
# datapath = "diagnostics/MPCAgent-1740947490813.json"
# datapath = "diagnostics/MPCAgent-1740947597406.json"
# datapath = "diagnostics/MPCAgent-1740948517234.json"
# datapath = "diagnostics/MPC4Agent-1740951597843.json"
# datapath = "diagnostics/mpcagent-replanned-fixed-wind-field-with-noise.json"
# datapath = "diagnostics/MPC4FollowerAgent-1741027696110.json"
# datapath = "diagnostics/MPCAgent-1741148859614.json"
# datapath = "diagnostics/MPCAgent-1741151447192.json"
# datapath = datapaths[1]#"diagnostics/MPCAgent-1741152861243.json"
# datapath = "diagnostics/MPC4Agent-1741198787338.json"
# datapath = "diagnostics/used_in_report/mpc4agent-no-replan-fixed-wind-full-horizon.json"
# datapath = "diagnostics/MPC4Agent-1741294093372.json"
# datapath = "diagnostics/MPC4Agent-1741294180645.json"
# datapath = "diagnostics/used_in_report/mpcagenet-no-replan-fixed-wind-full-horizon.json"
# datapath = "diagnostics/MPC4Agent-1741294476000.json" 
# datapath = "diagnostics/MPC4Agent-1741295136940.json"
# datapath = "diagnostics/MPC4Agent-1741297098842.json"
datapath = "diagnostics/used_in_report/mpc4agent-replanned-fixed-wind-with-noise.json"
# datapath = "diagnostics/MPC4Agent-1741297195697.json"
# datapath = "diagnostics/MPCAgent-1741297366179.json"
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

    TIME = list(range(result['steps'] + 1))

    mpc_agent_plan = np.array(result['rollout'][agent]['plan'])
    agent_z = np.array(result['rollout'][agent]['z'])
    simulation_z = np.array(result['rollout']['simulator']['z'])

    print('plan<->mpc-rollout fidelity:', np.linalg.norm(mpc_agent_plan - agent_z))
    print('plan<->sim-rollout fidelity:', np.linalg.norm(mpc_agent_plan - simulation_z))
    print('mpc-rollout<->sim-rollout fidelity:', np.linalg.norm(simulation_z - agent_z))

    agent_x = np.array(result['rollout'][agent]['x'])
    simulation_x = np.array(result['rollout']['simulator']['x'])
    print('mpc-rollout x<->sim-rollout x fidelity:', np.linalg.norm(agent_x - simulation_x))


    # plt.plot(range(result['steps']+1), mpc_agent_x, label='mpc x')
    # plt.plot(range(result['steps']+1), simulation_x, label='simulation x')
    # plt.legend()
    # plt.title('mpc-rollout x<->sim-rollout x')
    # plt.show()


    agent_y = np.array(result['rollout'][agent]['y'])
    simulation_y = np.array(result['rollout']['simulator']['y'])
    print('mpc-rollout x<->sim-rollout y fidelity:', np.linalg.norm(agent_y - simulation_y))
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
        plt.plot(range(result['steps']+1), mpc_agent_plan, label='mpc plan')

    plt.plot(range(result['steps']+1), agent_z, label='mpc dynamics rollout')
    plt.plot(range(result['steps']+1), simulation_z, label='simulator rollout')
    plt.legend()
    plt.title(f'seed={seed}, twr_score={twr_score}, fidelity={fidelity}')
    plt.show()

    if agent == 'mpc4_agent':
        plt.plot(range(result['steps']+1), mpc_agent_plan, label='mpc plan')
        plt.title('mpc plan (-1 <-> +1)')
        plt.show()

    print(f"seed={seed}, reward_score={reward_score:.5}, twr_score={twr_score:.3}, fidelity={fidelity}")




fidelities = np.array(fidelities)
twrs = np.array(twrs)

indices = np.argsort(fidelities)
plt.scatter(fidelities[indices], twrs[indices])
plt.plot([0,100], [1,1], 'r')
plt.yticks(np.linspace(0,10,20))
plt.show()