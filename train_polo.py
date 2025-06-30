import random
import gym
import numpy as np

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
from balloon_learning_environment.agents.mpc4_agent import get_initial_plan, grad_descent_optimizer, TerminalCost, get_dplan

### Tunable Parameters ###

# MPC settings

time_delta = 3*60
stride = 10

mpc_plan_horizon = 64 
mpc_replan_frequency = 16

mpc_initialization_num_plans = 100

# POLO Settings

ensemble_size = 8

# Training parameters

training_seeds = list(range(10_000, 11_000))
validation_seeds = list(range(11_000, 11_100))

training_num_episodes = 1000
training_max_episode_length = 1000

training_gradient_steps = 25 # because we are replanning less frequently

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


def get_optimized_plan(balloon: JaxBalloon, forecast: JaxWindField, atmosphere: JaxAtmosphere, terminal_cost_fn: TerminalCost, previous_plan=None):
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

    plan = grad_descent_optimizer(
        initial_plan,
        get_dplan,
        balloon,
        forecast,
        atmosphere,
        terminal_cost_fn,
        time_delta,
        stride)

    return plan

### Model definition ###

class ValueNetwork:
    pass

class EnsembleTerminalCost(TerminalCost):
    def __init__(self, ensemble: list[ValueNetwork]):
        pass

### Training loop ###

ensemble = [ ValueNetwork() for _ in range(ensemble_size) ]

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

    for t in range(training_max_episode_length):
        jax_balloon = JaxBalloon(JaxBalloonState.from_ble_state(arena.get_balloon_state()))
        ensemble_value_fn = EnsembleTerminalCost(ensemble) # NOTE: this is recreated every time so it should just be a light wrapper around ensemble to weighted softmax for terminal cost

        # Generate a new MPC plan if it's the first step or if we need to replan
        if t % mpc_replan_frequency == 0:
            plan = get_optimized_plan(jax_balloon, jax_forecast, jax_atmosphere, ensemble_value_fn, plan)

        # TODO: collect next state and store in buffer
        # TODO: collect rollouts from start states and train value functions

