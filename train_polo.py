import random
from balloon_learning_environment.env.balloon_env import BalloonEnv
from balloon_learning_environment.env.balloon_arena import BalloonArena
from balloon_learning_environment.env import generative_wind_field
from balloon_learning_environment.env.features import FeatureConstructor
from balloon_learning_environment.env import wind_field, simulator_data
from balloon_learning_environment.env.balloon import standard_atmosphere
from balloon_learning_environment.env.balloon.balloon import BalloonState
import gym
import numpy as np

training_seeds = list(range(10_000, 11_000))
validation_seeds = list(range(11_000, 11_100))

num_training_episodes = 1000
max_episode_length = 1000

horizon_length = 64 
replan_frequency = 16
gradient_steps = 25 # because we are replanning less frequently


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

for episode in range(num_training_episodes):
    # Sample seed randomly from the training seeds
    seed = random.choice(training_seeds)
    print(f"Training episode {episode + 1}/{num_training_episodes} with seed {seed}")

    # Create balloon and wind field from seed
    feature_constructor_factory = lambda forecast, atmosphere: StateAsFeature(forecast, atmosphere) 
    wind_field_factory = generative_wind_field.generative_wind_field_factory
    arena = BalloonArena(feature_constructor_factory, wind_field_factory(), seed=seed)
    # NOTE: starting position of balloon will be the same for a seed, so the balloon will always start at the same position given a seed

    for t in range(max_episode_length):
        plan = MPC_plan(arena, t, seed)  # Replace with your MPC plan function


