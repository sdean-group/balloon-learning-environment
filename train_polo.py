import random
import gym
import numpy as np
from typing import Union, Tuple

# BLE stuff
from balloon_learning_environment.env.balloon_env import BalloonEnv
from balloon_learning_environment.env.balloon_arena import BalloonArena
from balloon_learning_environment.env import generative_wind_field
from balloon_learning_environment.env.features import FeatureConstructor
from balloon_learning_environment.env import wind_field, simulator_data
from balloon_learning_environment.env.balloon import standard_atmosphere
from balloon_learning_environment.env.balloon.balloon import BalloonState

# Jax BLE stuff
from balloon_learning_environment.env.balloon.jax_balloon import JaxBalloon, JaxBalloonState
from balloon_learning_environment.env.wind_field import JaxWindField
from balloon_learning_environment.env.balloon.standard_atmosphere import JaxAtmosphere
from balloon_learning_environment.agents.mpc4_agent import get_initial_plan, grad_descent_optimizer, TerminalCost, get_dplan, jax_plan_cost

# Neural Network Parameters
import equinox as eqx
import optax

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

# Training parameters

training_seeds = list(range(10_000, 11_000))
validation_seeds = list(range(11_000, 11_100))

training_num_episodes = 1000
training_max_episode_length = 960

training_num_gradient_steps = 25 # because we are replanning less frequently
training_batch_size = 128

### Helper Functions ###

class StateAsFeature(FeatureConstructor):
    def __init__(self,
                forecast: wind_field.WindField,
                atmosphere: standard_atmosphere.Atmosphere):
        self.observation = None

    def observe(self, observation: simulator_data.SimulatorObservation):
        self.observation = observation

    def get_features(self) -> BalloonState: # This function usually returns a numpy array, but we return BalloonState directly
        return self.observation.balloon_observation

    def observation_space(self) -> gym.Space:
        return gym.Space() # This function isn't called so this baseclass can be used


def get_optimized_plan(balloon_state: JaxBalloonState, forecast: JaxWindField, atmosphere: JaxAtmosphere, terminal_cost_fn: TerminalCost, previous_plan=None, return_cost=False):
    balloon = JaxBalloon()
    
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

### Model definition ###


def compute_feature_vector(balloon: JaxBalloon, wind_forecast: JaxWindField):
    # NOTE: needs to be jax compatible
    pass

class ValueNetwork:
    # NOTE: must be callable within a jax jit/grad block of code
    # NOTE: implement with equinox?
    pass

class EnsembleTerminalCost(TerminalCost):
    """ Uses an ensemble of value networks to calculate terminal cost """
    def __init__(self, ensemble: list[ValueNetwork]):
        pass

    def __call__(self, balloon: JaxBalloon, wind_forecast: JaxWindField):
        # NOTE: call compute feature vector here, then combine the ensemble networks
        pass

class ValueTerminalCost(TerminalCost):
    """ Use a single value network as a terminal cost """
    def __init__(self, network: ValueNetwork):
        pass

    def __call__(self, balloon: JaxBalloon, wind_forecast: JaxWindField):
        # NOTE: call compute_feature_vector here
        pass

### Training loop ###

ensemble = [ ValueNetwork() for _ in range(polo_ensemble_size) ]
D: list[JaxBalloonState] = []

for episode in range(training_num_episodes):
    # Sample seed randomly from the training seeds
    seed = random.choice(training_seeds)
    print(f"Training episode {episode + 1}/{training_num_episodes} with seed {seed}")

    # Create balloon and wind field from seed
    feature_constructor_factory = lambda forecast, atmosphere: StateAsFeature(forecast, atmosphere) 
    wind_field_factory = generative_wind_field.generative_wind_field_factory
    arena = BalloonArena(feature_constructor_factory, wind_field_factory(), seed=seed)
    jax_forecast, jax_atmosphere = arena._wind_field.to_jax_wind_field(), arena._atmosphere.to_jax_atmosphere()
    # NOTE: starting position of balloon will be the same for a seed, so the balloon will always start at the same position given a seed
    # NOTE: call arena.reset to get initial observation

    plan = None # Initialize plan variable, potentially re-use across iterations based on replan frequency
    plan_idx = 0

    for t in range(training_max_episode_length):
        jax_balloon_state = JaxBalloonState.from_ble_state(arena.get_balloon_state())
        ensemble_value_fn = EnsembleTerminalCost(ensemble) # NOTE: this is recreated every time so it should just be a light wrapper around ensemble to weighted softmax for terminal cost

        # Generate a new MPC plan if it's the first step or if we need to replan
        if t % mpc_replan_frequency == 0:
            plan = get_optimized_plan(jax_balloon_state, jax_forecast, jax_atmosphere, ensemble_value_fn, plan)
            plan_idx = 0

        s_next = JaxBalloonState.from_ble_state(arena.step(plan[plan_idx])) # NOTE: we sample real transitions here (arena.step calculated the real wind)
        plan_idx += 1
        D.append(s_next)

        if t % polo_value_update_frequency == 0:
            for g in range(training_num_gradient_steps):
                batch: list[JaxBalloonState] = random.sample(D, min(len(D), training_batch_size))

                for k in range(len(ensemble)):
                    targets = []

                    for s_i in batch:
                        
                        plan_i, cost_i = get_optimized_plan(s_i, jax_forecast, jax_atmosphere, ValueTerminalCost(ensemble[k]), previous_plan=None, return_cost=True)
                        targets.append(cost_i)
                    
                    # TODO: calculate loss function and update ensemble[k]
