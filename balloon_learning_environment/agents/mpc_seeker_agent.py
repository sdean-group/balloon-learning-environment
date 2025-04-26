# from memory_profiler import profile
from balloon_learning_environment.agents import agent, station_seeker_agent, mpc4_agent
from balloon_learning_environment.env.wind_field import WindField, JaxWindField
from balloon_learning_environment.env import features
from balloon_learning_environment.env.balloon import control
from balloon_learning_environment.env.balloon.jax_balloon import JaxBalloon, JaxBalloonState
from balloon_learning_environment.env.balloon.standard_atmosphere import Atmosphere
from balloon_learning_environment.models import models
from balloon_learning_environment.env import simulator_data
from balloon_learning_environment.env.balloon.balloon import Balloon
from balloon_learning_environment.utils import units
import numpy as np
from typing import Optional, Sequence, Union
import jax.numpy as jnp
import datetime as dt
from atmosnav import *
import atmosnav as atm
from functools import partial
import copy

#  DOWN = 0
#  STAY = 1
#  UP = 2

# Use the actions decided by the 
def action_to_value(action):
    if action == control.AltitudeControlCommand.UP:     
        return 0.99
    elif action == control.AltitudeControlCommand.DOWN: 
        return -0.99
    else:
        return 0.0

def get_seeker_plan(
        observation: simulator_data.SimulatorObservation, 
        atmosphere: Atmosphere, 
        forecast: WindField, 
        station_seeker: station_seeker_agent.StationSeekerAgent, 
        feature_constructor: features.PerciatelliFeatureConstructor, 
        plan_steps: int, time_delta: dt.timedelta, stride: dt.timedelta):
    plan = []
    # observation is latest observation
    # feature_constructor already updated with observation
    balloon = Balloon(copy.deepcopy(observation.balloon_observation))

    for i in range(plan_steps):
        try:
            action = station_seeker.pick_action(feature_constructor.get_features())
            # Plans ahead 
            wind_vector = observation.wind_at_balloon if i == 0 else forecast.get_forecast(balloon.state.x, balloon.state.y, balloon.state.pressure, balloon.state.time_elapsed)
            balloon.simulate_step(wind_vector, atmosphere, action, time_delta, stride)
            plan.append(action)
        except:
            return plan

    return plan

class MPCSeekerAgent(agent.Agent):
    """An agent that takes uniform random actions."""

    def __init__(self, num_actions: int, observation_shape: Sequence[int]):
        super(MPCSeekerAgent, self).__init__(num_actions, observation_shape)
        self.forecast = None
        self.atmosphere = None
        
        self.jax_forecast = None
        self.jax_atmosphere = None

        self.perciatelli_feature_constructor = None

        self.station_seeker = station_seeker_agent.StationSeekerAgent(num_actions, observation_shape)

        self.plan_steps = 240
        self.time_delta = dt.timedelta(seconds=3*60)
        self.stride = dt.timedelta(seconds=10)

        self.i = 0

        self.get_dplan = jax.grad(mpc4_agent.jax_plan_cost, argnums=0)

    def _observe(self, observation):
        if self.perciatelli_feature_constructor is None:
            self.perciatelli_feature_constructor = features.PerciatelliFeatureConstructor(self.forecast, self.atmosphere)
        self.perciatelli_feature_constructor.observe(observation)


    def begin_episode(self, observation: np.ndarray) -> int:
        observation: simulator_data.SimulatorObservation = observation

        # Always update feature constructor
        self._observe(observation)
        
        # self.i = 0
        # return self.station_seeker.pick_action(self.perciatelli_feature_constructor.get_features())

        discrete_seeker_plan = get_seeker_plan(
            observation, 
            self.atmosphere, 
            self.forecast, 
            self.station_seeker, self.perciatelli_feature_constructor,
            self.plan_steps, self.time_delta, self.stride)

        continuous_seeker_plan = np.array([ action_to_value(action) for action in discrete_seeker_plan ])
        
        jax_balloon = JaxBalloon(JaxBalloonState.from_ble_state(observation.balloon_observation))
        optimizable_continuous_seeker_plan = mpc4_agent.inverse_sigmoid(continuous_seeker_plan)
        
        # seeker_plan = mpc4_agent.sigmoid(mpc4_agent.grad_descent_optimizer(
        #     optimizable_continuous_seeker_plan, 
        #     self.get_dplan,
        #     jax_balloon, 
        #     self.jax_forecast, 
        #     self.jax_atmosphere, 
        #     self.time_delta.seconds, 
        #     self.stride.seconds))

        seeker_plan = mpc4_agent.sigmoid(optimizable_continuous_seeker_plan)
        
        self.plan = seeker_plan

        # print('replanned')
        self.i = 0
        action = self.plan[self.i]
        self.i += 1
        return action

    def step(self, reward, observation):
        self._observe(observation)
        # return self.begin_episode(observation)

        # 0 - NEVER replan
        # 1 - ALWAYS replan
        replan_every = 24 # self.plan_steps
        if replan_every <= 0 or self.i%replan_every != 0:
            # print('not replanning')
            action = self.plan[self.i]
            self.i += 1
            return action
        else:
            return self.begin_episode(observation)

    def end_episode(self, reward: float, terminal: bool = True) -> None:
        self.i = 0 

    def update_forecast(self, forecast: agent.WindField): 
        self.forecast = forecast
        self.jax_forecast = forecast.to_jax_wind_field()

    def update_atmosphere(self, atmosphere: agent.standard_atmosphere.Atmosphere): 
        self.atmosphere = atmosphere
        self.jax_atmosphere = atmosphere.to_jax_atmosphere() 

