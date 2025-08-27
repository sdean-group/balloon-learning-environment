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


class ArenaStateAsFeature(FeatureConstructor):
    """ this is a BLE class used to get the balloon environment to return balloon state state (as opposed to vector states) """

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


def get_balloon_arena(seed: int) -> BalloonArena:
    feature_constructor_factory = lambda forecast, atmosphere: ArenaStateAsFeature(forecast, atmosphere) 
    wind_field_factory = generative_wind_field.generative_wind_field_factory
    arena = BalloonArena(feature_constructor_factory, wind_field_factory(), seed=seed)
    return arena