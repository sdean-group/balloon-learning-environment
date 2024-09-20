from balloon_learning_environment.agents import agent
from balloon_learning_environment.models import models
import numpy as np
from typing import Optional, Sequence, Union
import jax.numpy as jnp

from atmosnav import *


def make_weather_balloon(init_lat, init_lon, start_time, waypoint_time_step, integration_time_step, seed):
    return Airborne(
        jnp.array([ init_lat, init_lon, 0.0, 0.0 ]),
        PlanToWaypointController(start_time=start_time, waypoint_time_step=waypoint_time_step),
        AltitudeModel(integration_time_step=integration_time_step, key=jax.random.key(seed)))

WAYPOINT_TIME_STEP = 60*10
INTEGRATION_TIME_STEP = 60*60*3

def cost_at(start_time, balloon, plan, wind):
    # jax.debug.print("{start_time}, {balloon}, {plan}, {wind}", start_time=start_time, balloon=balloon, plan=plan, wind=wind)
    N = ((len(plan)-1)*WAYPOINT_TIME_STEP)//INTEGRATION_TIME_STEP
    def inner_run(i, time_balloon):
        time, balloon = time_balloon
        # step the agent in time
        next_balloon, _ =balloon.step(time, plan, wind.get_direction(time, balloon.state))

        # jump dt
        next_time = time + INTEGRATION_TIME_STEP
        return next_time, next_balloon

    final_time, final_balloon = jax.lax.fori_loop(0, N, inner_run, init_val=(start_time, balloon))
    return final_balloon.state[1]

gradient_at = jax.grad(cost_at, argnums=2)

WAYPOINT_COUNT = 40 #  Total sim time = Waypoint Count * Waypoint Time Step = 40 * 3 hours = 5 days
uppers = 10 + jnp.sin(2*np.pi*np.arange(WAYPOINT_COUNT)/10)
lowers = uppers - 3
plan = np.vstack([lowers,uppers]).T


# Idea: use observations to improve forecast (like perciatelli feature uses WindGP)

class MPCAgent(agent.Agent):
    """An agent that takes uniform random actions."""

    def __init__(self, num_actions: int, observation_shape: Sequence[int]):
        super(MPCAgent, self).__init__(num_actions, observation_shape)
        self.forecast = None

        self.plan = [ np.random.randint(0, 3) for _ in range(50) ]
        self.i = 0

    def begin_episode(self, observation: np.ndarray) -> int:
        print(observation[0].km, observation[1].km)
        self.i += 1
        return self.plan[0]

    def step(self, reward: float, observation: np.ndarray) -> int:
        action = self.plan[self.i%len(self.plan)]
        print(observation[0].km, observation[1].km)
        self.i+=1
        return action
 
    def end_episode(self, reward: float, terminal: bool = True) -> None:
        self.i = 0 

    def update_forecast(self, forecast: agent.WindField):
        self.forecast = forecast


