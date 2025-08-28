import copy
import random
import gym
import numpy as np
from typing import Union, Tuple
import jax
import jax.numpy as jnp
import optax
import equinox as eqx
import datetime as dt

# BLE stuff
from balloon_learning_environment.env.balloon_env import BalloonEnv
from balloon_learning_environment.utils import units
from balloon_learning_environment.env.balloon_arena import BalloonArena
from balloon_learning_environment.env import generative_wind_field
from balloon_learning_environment.env.features import FeatureConstructor
from balloon_learning_environment.env import wind_field, simulator_data
from balloon_learning_environment.env.balloon import standard_atmosphere
from balloon_learning_environment.env.balloon.balloon import BalloonState, Balloon

# Jax BLE stuff
from balloon_learning_environment.env.balloon.jax_balloon import JaxBalloon, JaxBalloonState
from balloon_learning_environment.env.wind_field import JaxWindField
from balloon_learning_environment.env.balloon.standard_atmosphere import JaxAtmosphere
from balloon_learning_environment.agents.mpc4_agent import get_initial_plan, grad_descent_optimizer, TerminalCost, get_dplan


from .utils import get_balloon_arena
### Tunable Parameters ###

"""
🎯 Concrete starting point for your case
Param	Value
Z	23 (update every replan) or 46 (every other replan)
G	10 (frequent updates) to 25 (less frequent updates)
H	24 (aim to reduce counterfactual tail issues)
N	12 (focus value function training on near-future that’s better predicted)

🚀 Pro tip: monitor ensemble disagreement + reward variance
If you see high disagreement growing at H’s tail → shorten H or downweight terminal cost

If value targets have high variance → shorten N
"""

# MPC settings

time_delta = 3*60
stride = 10

mpc_plan_horizon = 64 
mpc_replan_frequency = 16

mpc_initialization_num_plans = 100

# POLO Settings

polo_ensemble_size = 8
polo_value_update_frequency = 16

polo_max_episode_length = 240

# Training parameters

training_seeds = list(range(10_000, 11_000))
testing_seeds = list(range(11_000, 11_100))

training_num_episodes = 1000
training_max_episode_length = 960

training_num_gradient_steps = 25 # because we are replanning less frequently
training_batch_size = 128

### Helper Functions ###

def get_optimized_plan(balloon_state: JaxBalloonState, forecast: JaxWindField, atmosphere: JaxAtmosphere, terminal_cost_fn: TerminalCost, previous_plan=None, return_cost=False):
    """ Helper function to perform a trajectory optimization of a balloon in a wind field """
    balloon = JaxBalloon(balloon_state)
    
    initial_plan = get_initial_plan(
        balloon,
        mpc_initialization_num_plans, 
        forecast, 
        atmosphere, 
        terminal_cost_fn,
        mpc_plan_horizon, 
        time_delta, 
        stride, 
        previous_plan)

    results: Union[np.ndarray, Tuple[np.ndarray, float]] = grad_descent_optimizer(
        initial_plan,
        get_dplan,
        balloon,
        forecast,
        atmosphere,
        terminal_cost_fn,
        time_delta,
        stride,
        return_cost)

    return results

def evaluate_plan_in_arena(plan: np.ndarray, balloon_state: BalloonState, arena: BalloonArena):
    """ 
    Helper function to evaluate a plan in a balloon arena WITH wind noise 
    
    balloon_state is the state you want to evaluate the plan from
    arena is used to get the real wind 

    Note: does not modify the arena object
    Note: should return a list of states and then the caller can define the cost function? 
        or pass in a cost function that takes in a list of states
    Returns: final_balloon, total_steps_within_radius, total_steps, distances 
    """    
    balloon = Balloon(copy.deepcopy(balloon_state)) # NOTE: needs a copy of the balloon state because otherwise this method modifies the reference

    steps_within_radius = 0
    steps_completed = len(plan)

    distances = []

    for action in plan:
        wind_vector = arena._wind_field.get_ground_truth(
            balloon.state.x,
            balloon.state.y,
            balloon.state.pressure,
            balloon.state.elapsed_time)

        balloon.simulate_step(
            wind_vector, 
            arena._atmosphere, 
            action, 
            dt.timedelta(seconds=time_delta), 
            dt.timedelta(seconds=stride))

        if balloon.state.x.km**2 + balloon.state.y.km**2 < 50**2:
            steps_within_radius += 1

        distances.append(balloon.state.x.km**2 + balloon.state.y.km**2)

    return balloon.state, steps_within_radius, steps_completed, distances

def evaluate_mpc_performance(arena: BalloonArena, jax_forecast: JaxWindField, jax_atmosphere: JaxAtmosphere, terminal_cost_fn: TerminalCost):
    """ Helper function to evaluate MPC's performance in a balloon arena (e.g. how it would actually perform with wind noise) """
    
    # Initialize loop variables
    previous_plan = None
    balloon_state = None

    # Performance
    total_steps_within_radius = 0
    total_steps_completed = 0

    total_distance = 0
    
    while total_steps_completed < training_max_episode_length:
        balloon_state = arena.get_balloon_state()
        plan = get_optimized_plan(
            JaxBalloonState.from_ble_state(balloon_state), 
            jax_forecast, 
            jax_atmosphere, 
            terminal_cost_fn,
            previous_plan)

        balloon_state, steps_within_radius, steps_completed, distances = evaluate_plan_in_arena(plan[:mpc_replan_frequency], arena)
        
        which_steps = range(total_steps_completed, total_steps_completed + steps_completed)
        total_distance += sum(distance * 0.99**step for distance, step in zip(distances, which_steps))
        
        total_steps_within_radius += steps_within_radius
        total_steps_completed += steps_completed
    
    return total_distance, total_steps_within_radius / total_steps_completed


### Model definition ###
from .value_network import ValueNetwork, ValueNetworkTerminalCost, EnsembleValueNetworkTerminalCost
from .value_network_feature import ValueNetworkFeature, BasicValueNetworkFeature

class SpatialAveragingFeature(ValueNetworkFeature):
    @property
    def num_input_dimensions(self):
        raise NotImplementedError('Implement num_input_dimensions')

    def compute(self, balloon: JaxBalloon, wind_forecast: JaxWindField):
        raise NotImplementedError('Implement computing')

### Training loop ###


if __name__ == "__main__":
    vn_feature = BasicValueNetworkFeature()
    ensemble = [ 
        ValueNetwork.create(key=jax.random.key(seed=seed), input_dim=vn_feature.num_input_dimensions) 
        for seed 
        in range(polo_ensemble_size) ]
    D: list[JaxBalloonState] = []

    for episode in range(training_num_episodes):
        print("Episode", episode)
        
        # Evaluate performance of MPC with ensemble terminal cost on testing seeds
        if episode % 10 == 0:
            print(f"Evaluating performance on testing seeds at episode {episode}")
            ensemble_value_fn = EnsembleValueNetworkTerminalCost(vn_feature, ensemble)

            for seed in testing_seeds:
                arena = get_balloon_arena(seed)
                jax_forecast, jax_atmosphere = arena._wind_field.to_jax_wind_field(), arena._atmosphere.to_jax_atmosphere()

                twr, cost = evaluate_mpc_performance(arena, jax_forecast, jax_atmosphere, ensemble_value_fn)

        # Sample seed randomly from the training seeds
        seed = random.choice(training_seeds)
        print(f"Training episode {episode + 1}/{training_num_episodes} with seed {seed}")

        # Create balloon and wind field from seed
        arena = get_balloon_arena(seed)
        jax_forecast, jax_atmosphere = arena._wind_field.to_jax_wind_field(), arena._atmosphere.to_jax_atmosphere()
        # NOTE: starting position of balloon will be the same for a seed, so the balloon will always start at the same position given a seed
        # NOTE: call arena.reset to get initial observation

        plan = None # Initialize plan variable, potentially re-use across iterations based on replan frequency
        plan_idx = 0

        for t in range(training_max_episode_length):
            jax_balloon_state = JaxBalloonState.from_ble_state(arena.get_balloon_state())
            ensemble_value_fn = EnsembleValueNetworkTerminalCost(vn_feature, ensemble) # NOTE: this is recreated every time so it should just be a light wrapper around ensemble to weighted softmax for terminal cost

            # Generate a new MPC plan if it's the first step or if we need to replan
            if t % mpc_replan_frequency == 0:
                plan = get_optimized_plan(jax_balloon_state, jax_forecast, jax_atmosphere, ensemble_value_fn, plan)
                plan_idx = 0

            s_next = JaxBalloonState.from_ble_state(arena.step(plan[plan_idx])) # NOTE: we sample real transitions here (arena.step calculated the real wind)
            plan_idx += 1
            D.append(s_next)

            if t % polo_value_update_frequency == 0:
                for g in range(training_num_gradient_steps):
                    state_batch: list[JaxBalloonState] = random.sample(D, min(len(D), training_batch_size))
                    feature_batch = [vn_feature.compute(state, jax_forecast) for state in state_batch]

                    for k in range(len(ensemble)):
                        targets = []

                        for s_i in state_batch:
                            plan_i = get_optimized_plan(s_i, jax_forecast, jax_atmosphere, ValueNetworkTerminalCost(vn_feature, ensemble[k]), previous_plan=None)
                            
                        
                        ensemble[k].update(feature_batch, targets)


                    