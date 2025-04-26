import json
import numpy as np
import matplotlib.pyplot as plt
import re

def get_perciatelli_rewards(use_twr=True):
    perciatelli_path = "diagnostics/perciatelli-100-seeds.json"
    perciatelli_diagnostics = json.load(open(perciatelli_path, 'r'))
    perciatelli_rewards = {}
    for result in perciatelli_diagnostics:
        perciatelli_rewards[result['seed']] = result['twr' if use_twr else 'reward']
    
    return perciatelli_rewards

def get_mpc4_rewards(use_twr=True):
    mpc4_path = "diagnostics/mpc4agent-replanned-redistributed-initial-plans.json"
    mpc4_diagnostics = json.load(open(mpc4_path, 'r'))
    mpc4_rewards = {}
    for seed, result in mpc4_diagnostics.items():
        value = result['twr' if use_twr else 'reward']
        if result['steps'] != 960:
                print(f"WARNING -- {seed} ran out of power", end=' -- ')
                if use_twr:
                    print('Approximating twr by placing balloon out of radius')
                    value = (result['steps'] * value) / 960
                else:
                    print('Cannot approximate remaining reward, ignoring this seed')
                    continue

        mpc4_rewards[int(seed)] = value
    return mpc4_rewards

def get_mpc4_small_fix_rewards(use_twr=True):
    pattern_twr = r"seed=(\d+).*?time_within_radius=([-\d.]+).*?final_timestep=(\d+)"
    pattern_reward = r"seed=(\d+).*?cumulative_reward=([-\d.]+).*?final_timestep=(\d+)"

    pattern = pattern_twr if use_twr else pattern_reward

    rewards = {}

    mpc4_path = "logfile_modified_initialization.out"
    # mpc4_path = "logfile.out"
    mpc4_diagnostics = open(mpc4_path, 'r')
    for line in mpc4_diagnostics:
        regex_match = re.search(pattern, line)
        if regex_match:

            seed = int(regex_match.group(1))
            twr = float(regex_match.group(2))
            final_timestep = int(regex_match.group(3))
            
            if final_timestep != 960:
                print(f"WARNING -- {seed} ran out of power", end=' -- ')
                if use_twr:
                    print('Approximating twr by placing balloon out of radius')
                    twr = (final_timestep * twr) / 960
                else:
                    print('Cannot approximate remaining reward, ignoring this seed')
                    continue
            
            rewards[int(seed)] = float(twr)
            # print(line)
            # print(f"seed={seed}, twr={twr}")
    
    return rewards

def compare_perciatelli_mpc4_rewards():
    use_twr = True
    seeds_to_test = set(range(46))

    mpc4_rewards = { key:value for key, value in get_mpc4_small_fix_rewards(use_twr).items() if key in seeds_to_test}
    perciatelli_rewards = { key:value for key, value in get_perciatelli_rewards(use_twr).items() if key in seeds_to_test}

    print('mpc4 average: ', sum(mpc4_rewards.values()) / len(mpc4_rewards))
    print('perciatelli average: ', sum(perciatelli_rewards.values()) / len(perciatelli_rewards))

    perciatelli_non_zero_count = 0
    mpc4_non_zero_count = 0

    reward_scores = []

    for seed in seeds_to_test:
        if seed in perciatelli_rewards and perciatelli_rewards[seed] > 0:
            perciatelli_non_zero_count += 1

        if seed in mpc4_rewards and mpc4_rewards[seed] > 0:
            mpc4_non_zero_count += 1

        if seed in mpc4_rewards and seed in perciatelli_rewards:        
            perciatelli_reward = perciatelli_rewards[seed]
            mpc4_reward = mpc4_rewards[seed]

            seed_repr = str(seed) + (3 - len(str(seed))) * ' '

            if perciatelli_reward == 0:
                reward_score = -1.0 - mpc4_reward
                reward_scores.append(reward_score)
                print(f"seed={seed_repr}, prci={perciatelli_reward:.5f}, mpc4={mpc4_reward:.5f}, score={mpc4_reward:.5f}/0.0")
            else:
                reward_score  = mpc4_reward / perciatelli_reward
                reward_scores.append(reward_score)
                print(f"seed={seed_repr}, prci={perciatelli_reward:.5f}, mpc4={mpc4_reward:.5f}, score={reward_score:.5f}")
        else:
            print('skipping seed', seed)

    print('nonzero on', perciatelli_non_zero_count, 'seeds, perciatelli average:', sum(perciatelli_rewards.values()) / perciatelli_non_zero_count)
    print('nonzero on', mpc4_non_zero_count, 'seeds, mpc4 average:', sum(mpc4_rewards.values()) / mpc4_non_zero_count)

    # Separate reward scores into three groups
    scores_below_one = [score for score in reward_scores if 0.0 <= score <= 1.0]
    scores_above_one = [score for score in reward_scores if score > 1.0]
    negative_scores = [score for score in reward_scores if score < 0.0]

    print(negative_scores)

    # Plot scores below or equal to 1.0
    plt.figure(figsize=(20, 7))
    bins_below_one = np.linspace(0.0, 1.0, 20)
    plt.hist(scores_below_one, bins=bins_below_one, edgecolor='black')
    plt.xticks(np.linspace(0.0, 1.0, 11), rotation=45)
    plt.yticks(range(0, max(np.histogram(scores_below_one, bins=bins_below_one)[0]) + 1))
    plt.title(f"Reward Scores <= 1.0 (Count: {len(scores_below_one)})")
    plt.xlabel("Reward Score")
    plt.ylabel("Frequency")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

    # Plot scores above 1.0
    plt.figure(figsize=(20, 7))
    bins_above_one = np.arange(1.0, max(scores_above_one) + 2, 1.0)
    plt.hist(scores_above_one, bins=bins_above_one, edgecolor='black')
    plt.xticks(np.arange(1.0, max(scores_above_one) + 2, 1.0), rotation=45)
    plt.yticks(range(0, max(np.histogram(scores_above_one, bins=bins_above_one)[0]) + 1))
    plt.title(f"Reward Scores > 1.0 (Count: {len(scores_above_one)})")
    plt.xlabel("Reward Score")
    plt.ylabel("Frequency")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

    # Plot negative scores (original mpc4 rewards)
    plt.figure(figsize=(20, 7))
    negative_scores = -(np.array(negative_scores) + 1)
    bins_negative = np.linspace(0.0, 1.1, 22) - 1.1/22
    plt.hist(negative_scores, bins=bins_negative, edgecolor='black')
    plt.xticks(bins_negative)
    plt.yticks(range(0, max(np.histogram(negative_scores, bins=bins_negative)[0]) + 1))
    plt.title(f"Scores when Perciatelli gets 0 (Count: {len(negative_scores)})")
    plt.xlabel("Original MPC4 Reward")
    plt.ylabel("Frequency")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

# def get_histogram_results

compare_perciatelli_mpc4_rewards()

exit()

# diagnostics/mpc4agent-no-replan-no-noise-240-steps.json
# diagnostics/mpc4agent-no-replan-no-noise-redistributed-initial-plans-with-noise.json

mpc4_diagnostic_paths = [
    "diagnostics/mpc4agent-replanned-no-noise-redistributed-initial-plans-with-no-noise.json",
    "diagnostics/mpc4agent-replanned-no-noise-redistributed-initial-plans-with-noise-500-random-plans.json",
    "diagnostics/mpc4agent-replanned-no-noise-redistributed-initial-plans-with-noise-no-discount-warm-start.json",
    "diagnostics/mpc4agent-replanned-no-noise-redistributed-initial-plans-with-noise-no-discount.json",
    "diagnostics/mpc4agent-replanned-no-noise.json",
]

mpc4_diagnostics = { path: json.load(open(path, 'r')) for path in mpc4_diagnostic_paths}

seeds = mpc4_diagnostics[mpc4_diagnostic_paths[0]].keys()

print(seeds)

for i, path in enumerate(mpc4_diagnostic_paths):
    print(i, '->', path)

print()

stats = { path: 0.0 for path in mpc4_diagnostic_paths}

for seed in seeds:
    print(seed + (3 - len(seed)) * ' ', end=' | ')
    for i, path in enumerate(mpc4_diagnostic_paths):
        twr, steps = mpc4_diagnostics[path][seed]['twr'], mpc4_diagnostics[path][seed]['steps']

        if steps != 960:
            print('WARNING -- path ended early')

        stats[path] += twr
        print(f'{twr:0.3f}', end=' | ')

    print()

# Print all the average twr scores by printing the 
print()
for i in range(len(mpc4_diagnostic_paths)):
    path = mpc4_diagnostic_paths[i]
    avg_twr = stats[path] / len(seeds)
    print(f'{i} -> {avg_twr:.3f}')

# Display the paths of each algorithm for each seed in a plot with 5 subplots and there is a new plot for every seed
for seed in seeds:
    plt.figure(figsize=(15, 10))
    plt.suptitle(f'Seed {seed}')
    plt.xlabel('Time')
    plt.ylabel('Z')

    for i, path in enumerate(mpc4_diagnostic_paths):
        steps = list(range(mpc4_diagnostics[path][seed]['steps']+2))
        agent_z = np.array(mpc4_diagnostics[path][seed]['rollout']['mpc4_agent']['z'])
        plt.plot(steps, agent_z, label=i)
    
    plt.legend()
    plt.show()

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111)
    ax.legend()
    plt.title(f'Seed {seed}')

    for i, path in enumerate(mpc4_diagnostic_paths):
        steps = list(range(mpc4_diagnostics[path][seed]['steps']+2))

        agent_x = np.array(mpc4_diagnostics[path][seed]['rollout']['mpc4_agent']['x'])
        agent_y = np.array(mpc4_diagnostics[path][seed]['rollout']['mpc4_agent']['y'])
        agent_z = np.array(mpc4_diagnostics[path][seed]['rollout']['mpc4_agent']['z'])

        ax.plot(agent_x, agent_y, label=i)

    plt.legend()
    plt.show()

