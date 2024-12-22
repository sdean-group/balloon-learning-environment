# from memory_profiler import profile
import atmosnav.utils
from balloon_learning_environment.agents import agent
from balloon_learning_environment.env.wind_field import JaxWindField
from balloon_learning_environment.models import models
from balloon_learning_environment.utils import units
# from balloon_learning_environment.env.balloon.standard_atmosphere import
import numpy as np
from typing import Optional, Sequence, Union
import jax.numpy as jnp
import datetime as dt
from atmosnav import *
import atmosnav as atm
from scipy.optimize import minimize
import json
import atmosnav

def convert_plan_to_actions(plan, observation, i, follower):
    i %= len(plan)
    _, _, _, pressure = observation
    altitude = follower.atmosphere.at_pressure(pressure).height.km
    
    # Change the following to test: 
    # pressure = follower.atmosphere.at_height(units.Distance(km=altitude)).pressure
    # pressure = jax.jit(atmosnav.utils.alt2p)(altitude)
    pressure = follower.jax_atmosphere.at_height(altitude*1000).pressure

    if pressure < plan[i]:
        return 0 # UP
    
    return 2

class Follower(agent.Agent):
    """An agent that takes uniform random actions."""

    def __init__(self, num_actions: int, observation_shape: Sequence[int]):
        super(Follower, self).__init__(num_actions, observation_shape)
        self.i = 0
        
        filepath = '/tmp/ble/eval/micro_eval/perciatelli44.json'

        rawjson = open(filepath).read()
        data = json.loads(rawjson)
        self.plan = list(entry['pressure'] for entry in data[0]['flight_path'])

    def begin_episode(self, observation: np.ndarray) -> int:
        return convert_plan_to_actions(self.plan, observation, self.i, self)

    def step(self, reward: float, observation: np.ndarray) -> int:
        self.i += 1
        return convert_plan_to_actions(self.plan, observation, self.i, self)
 
    def end_episode(self, reward: float, terminal: bool = True) -> None:
        self.i = 0 

    def update_forecast(self, forecast: agent.WindField): 
        self.forecast = forecast.to_jax_wind_field()

    def update_atmosphere(self, atmosphere: agent.standard_atmosphere.Atmosphere): 
        self.atmosphere = atmosphere
        self.jax_atmosphere = atmosphere.to_jax_atmopshere()

