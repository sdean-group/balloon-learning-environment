from balloon_learning_environment.agents import agent
from balloon_learning_environment.models import models
from balloon_learning_environment.utils import units
import numpy as np
from typing import Optional, Sequence, Union
import jax.numpy as jnp
import datetime as dt
from atmosnav import *
import atmosnav as atm

class DeterministicAltitudeModel(Dynamics):

    def __init__(self, integration_time_step):
        self.dt = integration_time_step
        self.vlim = 1.7

    def control_input_to_delta_state(self, time: jnp.float32, state: Array, control_input: Array, wind_vector: Array):
        h = self.update(state, control_input[0])
        return jnp.array([ wind_vector[0], wind_vector[1], h - state[2], 0.0]), self
    
    def update(self, state, waypoint):
        return lax.cond(jnp.abs(waypoint-state[2]) > self.vlim / 3600.0 * self.dt,
                        lambda op: op[0][2] + self.vlim / 3600.0 * self.dt * jnp.sign(op[1]-op[0][2]),
                        lambda op: op[1],
                        operand = (state, waypoint))

    def tree_flatten(self):
        children = ()  # arrays / dynamic values
        aux_data = {'dt':self.dt, 'vlim':self.vlim}  # static values
        return (children, aux_data)
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return DeterministicAltitudeModel(aux_data['dt'])

def make_weather_balloon(init_lat, init_lon, init_pressure, start_time):
    return Airborne(
        jnp.array([ init_lat, init_lon, atm.utils.p2alt(init_pressure), 0.0 ]),
        # [ init_lat, init_lon, atm.utils.p2alt(init_pressure), 0.0 ],
        PlanToWaypointController(start_time=start_time, waypoint_time_step=3*60),
        DeterministicAltitudeModel(integration_time_step=3*60))

def cost_at(balloon, plan, wind):
    # jax.debug.print("{start_time}, {balloon}, {plan}, {wind}", start_time=start_time, balloon=balloon, plan=plan, wind=wind)
    N = ((len(plan)-1))
    def inner_run(i, balloon):
        # step the agent in time
        elapsed_time, x, y, altitude = balloon.state

        # if x < 0: x = -x

        # x=units.Distance(m=x)
        # y=units.Distance(m=y)
        # pressure = atm.utils.alt2p(altitude)
        # elapsed_time = dt.timedelta(seconds=elapsed_time)

        # wind_vector = wind.get_forecast(x, y, pressure, elapsed_time)

        next_balloon, _ = balloon.step(elapsed_time, plan, jnp.array([ wind_vector.u, wind_vector.v ]))

        return next_balloon

    final_balloon = jax.lax.fori_loop(0, N, inner_run, init_val=balloon)
    return final_balloon.state[0]**2 + final_balloon.state[1]**2.

gradient_at = jax.grad(cost_at, argnums=2)

# Plan helper functions
def make_plan(num_plans, num_steps, balloon, wind):
        
    plans = [np.zeros((num_steps, 1))]

    for _ in range(num_plans):
        plan = 22*np.random.rand(1) + jnp.sin(2*np.pi*np.random.rand(1)*np.arange(num_steps)/10)
        plans.append(np.reshape(plan, (num_steps, 1)))

    best_plan = -1
    best_cost = np.inf
    for i, plan in enumerate(plans):
        
        cost = cost_at(balloon, plan, wind)
        if cost < best_cost:
            best_plan = i
            best_cost = cost

    plan = plans[best_plan]
    return plan

def convert_plan_to_actions(plan, observation, i):
    _, _, _, pressure = observation
    if plan[i] > pressure:
        return 2
    elif plan[i] < pressure: 
        return 0
    else:
        return 1




#  DOWN = 0
#  STAY = 1
#  UP = 2

# Idea: use observations to improve forecast (like perciatelli feature uses WindGP)

class MPCAgent(agent.Agent):
    """An agent that takes uniform random actions."""

    def __init__(self, num_actions: int, observation_shape: Sequence[int]):
        super(MPCAgent, self).__init__(num_actions, observation_shape)
        self.forecast = None
        self.atmosphere = None
        
        self.plan_size = 50
        self.plan = None
        self.i = 0
        self.waypoint_time_step = 3*60
        self.integration_time_step = 3*60

    def begin_episode(self, observation: np.ndarray) -> int:
        x = observation[1].km
        y = observation[2].km
        pressure = observation[3]
        t = observation[0].seconds

        # t, x, y, pressure = observation
        balloon = make_weather_balloon(x, y, pressure, t)
        self.plan = make_plan(5000, 1000, balloon, self.forecast)
        self.i += 1
        return self.plan[0]

    def step(self, reward: float, observation: np.ndarray) -> int:
        # t, x, y, pressure = observation
        action = self.plan[self.i%len(self.plan)]
        self.i += 1
        return action
 
    def end_episode(self, reward: float, terminal: bool = True) -> None:
        self.i = 0 

    def update_forecast(self, forecast: agent.WindField): 
        self.forecast = forecast

    def update_atmosphere(self, atmosphere: agent.standard_atmosphere.Atmosphere): 
        self.atmosphere = atmosphere


    # def _check_forecast_present(self):


