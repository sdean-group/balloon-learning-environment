import heapq
import math
from typing import List

import numpy as np
from dataclasses import dataclass

from balloon_learning_environment.env.wind_field import JaxWindField
from balloon_learning_environment.agents import agent

"""  DOWN = 0
  STAY = 1
  UP = 2
"""


class ExplorerState:
    def __init__(self, x, y, pressure, time):
        self.x = x
        self.y = y
        self.pressure = pressure
        self.time = time

    def next_state(self, action: int, wind_vector, dt):
        """ """
        delta_pressure = 0
        if action == 0:     delta_pressure = +100
        elif action == 2:   delta_pressure = -100

        return ExplorerState(
            self.x + wind_vector[0] * dt, 
            self.y + wind_vector[1] * dt, 
            self.pressure + delta_pressure * dt, 
            self.time + dt)


class Node:
    def __init__(self, state, cumulative_reward, action_sequence, depth):
        self.state: ExplorerState = state                    # Balloon state (instance of Balloon)
        self.cumulative_reward = cumulative_reward
        self.action_sequence = action_sequence  # List of actions taken from the root to reach this node
        self.depth = depth                    # Number of time steps taken so far
        self.optimistic_value = None          # Upper bound on total reward from this node onward

    def __str__(self):
        return (f"Node(depth={self.depth}, cum_reward={self.cumulative_reward}, "
                f"opt_val={self.optimistic_value}, actions={self.action_sequence})")
    
@dataclass
class ExplorerOptions:
    budget: int
    planning_horizon: int
    delta_time: int

@dataclass
class PlanOptions:
    delta_time: int

def run_opd_search(start: ExplorerState, wind_field: JaxWindField, action_space: List[int], options: ExplorerOptions):
    queue = []
    node_counter = 0  # A counter to break ties in the heap
    
    def push_node(node: Node):
        nonlocal node_counter
        # Calculate the remaining steps.
        remaining_steps = options.planning_horizon - node.depth
        # Assume that at most you can get a reward of 1 per step.
        max_future_reward = remaining_steps
        # Optimistic value: cumulative reward so far + maximum possible future reward.
        node.optimistic_value = node.cumulative_reward + max_future_reward
        # Push into the heap (using negative value for max-first behavior).
        heapq.heappush(queue, (-node.optimistic_value, node_counter, node))
        node_counter += 1
    
    root = Node(state=start, cumulative_reward=0, action_sequence=[], depth=0)
    push_node(root)

    # best_nodes={}
    best_node_early = root
    best_node = root
    iterations = 0

    while len(queue) > 0 and iterations < options.budget:
        node: Node = heapq.heappop(queue)[2]
        if node.depth > options.planning_horizon: 
            continue

        wind_vector = wind_field.get_forecast(node.state.x/1000, node.state.y/1000, node.state.pressure, node.state.time)

        for action in action_space:
            next_state = node.state.next_state(action, wind_vector, options.delta_time)
            
            distance = math.sqrt((next_state.x/1000)**2 + (next_state.y/1000)**2)
            reward = 1 if distance <= 50.0 else 0

            next_node = Node(
                state=next_state, 
                cumulative_reward=node.cumulative_reward + reward,
                action_sequence=node.action_sequence + [action],
                depth=node.depth + 1)
            push_node(next_node)

            # if next_node.depth in best_nodes:
            #     if next_node.cumulative_reward > best_nodes[next_node.depth].cumulative_reward:
            #         best_nodes[next_node.depth] = next_node
            # else:
            #     best_nodes[next_node.depth] = next_node

            if next_node.cumulative_reward > best_node_early.cumulative_reward:
                best_node_early = next_node

            if next_node.cumulative_reward >= best_node.cumulative_reward and next_node.depth >= best_node.depth:
                best_node = next_node

        iterations += 1

    return best_node, best_node_early

def get_plan_from_opd_node(node: Node, search_delta_time: int, plan_delta_time: int):
    plan_size = (node.depth * search_delta_time) // plan_delta_time
    plan = np.zeros(plan_size)
    i = 0
    for action in node.action_sequence:
        for j in range(search_delta_time//plan_delta_time):
            plan_size[i] = action
            i+=1
    
    return plan

def get_best_plan(start: ExplorerState, wind_field: JaxWindField, action_space: List[int], opd_options: ExplorerOptions, plan_options: PlanOptions):
    best_node, _ = run_opd_search(start, wind_field, action_space, opd_options)
    return get_plan_from_opd_node(best_node, opd_options.delta_time, plan_options.delta_time)