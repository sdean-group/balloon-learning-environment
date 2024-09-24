from memory_profiler import profile
from balloon_learning_environment.agents import agent
from balloon_learning_environment.models import models
from balloon_learning_environment.utils import units
from atmosnav import JaxTree
import numpy as np
from typing import Optional, Sequence, Union
import jax.numpy as jnp
import datetime as dt
from atmosnav import *
import atmosnav as atm
from scipy.optimize import minimize

class DeltaTime(JaxTree):
    def __init__(self, seconds):
        self.seconds = seconds

    def total_seconds(self):
        return self.seconds

    def tree_flatten(self):
        return (self.seconds, ), {}

    @classmethod
    def tree_unflatten(cls, aux_data, children): 
        return DeltaTime(seconds=children[0])


class DeterministicAltitudeModel(Dynamics):

    def __init__(self, integration_time_step):
        self.dt = integration_time_step
        self.vlim = 1.7

    @profile
    def control_input_to_delta_state(self, time: jnp.float32, state: Array, control_input: Array, wind_vector: Array):
        h = self.update(state, control_input[0])
        return jnp.array([ wind_vector[0], wind_vector[1], h - state[2], 0.0]), self
    
    @profile
    def update(self, state, waypoint):
        return jax.lax.cond(jnp.abs(waypoint-state[2]) > self.vlim / 3600.0 * self.dt,
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

@profile
def make_weather_balloon(init_lat, init_lon, init_pressure, start_time):
    return Airborne(
        jnp.array([ init_lat, init_lon, atm.utils.p2alt(init_pressure), 0.0 ]),
        # [ init_lat, init_lon, atm.utils.p2alt(init_pressure), 0.0 ],
        PlanToWaypointController(start_time=start_time, waypoint_time_step=3*60),
        DeterministicAltitudeModel(integration_time_step=3*60))

@profile
def cost_at(start_time, dt, balloon, plan, wind):
    # jax.debug.print("{start_time}, {balloon}, {plan}, {wind}", start_time=start_time, balloon=balloon, plan=plan, wind=wind)
    N = ((len(plan)-1))
    def inner_run(i, time_and_balloon):
        time, balloon = time_and_balloon
        # step the agent in time
        x, y, altitude, _ = balloon.state
        pressure = atm.utils.alt2p(altitude)
        wind_vector = wind.get_forecast(x, y, pressure, time)
        next_balloon, _ = balloon.step(time, plan, wind_vector)
        return time + dt, next_balloon

    _, final_balloon = jax.lax.fori_loop(0, N, inner_run, init_val=(start_time, balloon))
    return final_balloon.state[0]**2 + final_balloon.state[1]**2.

# gradient_at = jax.jit(jax.grad(cost_at, argnums=3))
gradient_at = jax.grad(cost_at, argnums=3)

# Plan helper functions
@profile
def make_plan(start_time, dt, num_plans, num_steps, balloon, wind):
        
    plans = [jnp.zeros((num_steps, 1))]

    for _ in range(num_plans):
        plan = 22*np.random.rand(1) + jnp.sin(2*np.pi*np.random.rand(1)*np.arange(num_steps)/10)
        plans.append(jnp.reshape(plan, (num_steps, 1)))

    best_plan = -1
    best_cost = np.inf
    for i, plan in enumerate(plans):
        
        cost = cost_at(start_time, dt, balloon, plan, wind)
        if cost < best_cost:
            best_plan = i
            best_cost = cost

    plan = plans[best_plan]
    return jnp.array(plan)

@profile
def convert_plan_to_actions(plan, observation, i):
    i %= len(plan)
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
# TODO: use atmopshere class to do conversions between altitude and pressure

@profile
def cost(plan, observation, forecast, atmosphere, stride):
    vlim = 1.7
    t_i, x_i, y_i, p_i = observation
    cost = 0.0
    for a_target in plan: # target pressure
        wind_vector = forecast.get_forecast(x_i, y_i, p_i, t_i)
        # print(wind_vector)

        x_i += wind_vector.u * stride
        y_i += wind_vector.v * stride


        p_target = atmosphere.at_height(units.Distance(km=a_target)).pressure
        if p_i > p_target:
            a_i = atmosphere.at_pressure(p_i).height.km
            
            if abs(a_target-a_i) > vlim / 3600.0 * stride.seconds:
                a_i += vlim / 3600.0 * stride.seconds * np.sign(a_target-a_i)
            else:
                a_i = a_target
            p_i = atmosphere.at_height(a_i).pressure
        
        t_i += stride

        cost += -(x_i.meters)**2# + (y_i.meters**2)
    return cost


    

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

    @profile
    def begin_episode(self, observation: np.ndarray) -> int:
        # Failed scipy attempt
        # initial_plan = np.full((self.plan_size, ), 5.0)
        # self.plan = minimize(cost, initial_plan, args=(observation, self.forecast, self.atmosphere, dt.timedelta(minutes=3)))
        # print(self.plan)
        
        # atmosnav optimizer:
        x = observation[1].km
        y = observation[2].km
        pressure = observation[3]
        t = observation[0].seconds

        # t, x, y, pressure = observation
        balloon = make_weather_balloon(x, y, pressure, t)
        self.plan = make_plan(t, self.integration_time_step, 1, 1000, balloon, self.forecast)
        action = convert_plan_to_actions(self.plan, observation, self.i)
        self.i += 1
        return action

    @profile
    def step(self, reward: float, observation: np.ndarray) -> int:
        # t, x, y, pressure = observation

        action = convert_plan_to_actions(self.plan, observation, self.i)
        self.i += 1
        return action
 
    def end_episode(self, reward: float, terminal: bool = True) -> None:
        self.i = 0 

    @profile
    def update_forecast(self, forecast: agent.WindField): 
        self.forecast = forecast.to_jax_wind_field()

    @profile
    def update_atmosphere(self, atmosphere: agent.standard_atmosphere.Atmosphere): 
        self.atmosphere = atmosphere


    # def _check_forecast_present(self):


