# from memory_profiler import profile
from balloon_learning_environment.agents import agent
from balloon_learning_environment.env.wind_field import JaxWindField
from balloon_learning_environment.models import models
from balloon_learning_environment.utils import units
import numpy as np
from typing import Optional, Sequence, Union
import jax.numpy as jnp
import datetime as dt
from atmosnav import *
import atmosnav as atm
from functools import partial

class DeterministicAltitudeModel(Dynamics):

    def __init__(self, integration_time_step):
        self.dt = integration_time_step
        self.vlim = 1.7

    def control_input_to_delta_state(self, time: jnp.float32, state: Array, control_input: Array, wind_vector: Array):
        h = self.update(state, control_input[0])
        return jnp.array([ wind_vector[0], wind_vector[1], h - state[2], 0.0]), self

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

def make_weather_balloon(init_lat, init_lon, init_pressure, start_time, atmosphere, waypoint_time_step, integration_time_step):
    return Airborne(
        jnp.array([ init_lat, init_lon, atmosphere.at_pressure(init_pressure).height.km, 0.0 ]),
        PlanToWaypointController(start_time, waypoint_time_step),
        DeterministicAltitudeModel(integration_time_step))

@partial(jax.jit, static_argnums=(5, 6))
def cost_at(start_time, balloon, plan, wind, atmosphere, waypoint_time_step, integration_time_step):
    N = (waypoint_time_step * (len(plan)-1)) // integration_time_step
    factor = 1.0
    cost = 0.0
    plan_change_cost = 0.0
    
    def inner_run(i, time_balloon_cost):
        time, balloon, cost, factor, plan_change_cost = time_balloon_cost
        x, y, altitude, _ = balloon.state
        
        cost += factor * (balloon.state[0]**2 + balloon.state[1]**2)
        factor *= 0.99
        
        # plan_change_penalty = 50 * jax.lax.cond(i > 0, 
        #                                    lambda: (plan[i,0] - plan[i-1,0]) ** 2, 
        #                                    lambda: 0.0)
        # plan_change_cost += plan_change_penalty
        
        pressure = atmosphere.at_height(height_meters=altitude * 1000.0).pressure
        wind_vector = integration_time_step * wind.get_forecast(x, y, pressure, time) / 1000.0
        next_balloon, _ = balloon.step(time, plan, wind_vector)
        
        return time + integration_time_step, next_balloon, cost, factor, plan_change_cost
    
    t, final_balloon, cost, factor, plan_change_cost = jax.lax.fori_loop(0, N, inner_run, 
                                                                         init_val=(start_time, balloon, cost, factor, plan_change_cost))
    
    total_cost = cost + plan_change_cost
    return total_cost

gradient_at = jax.jit(jax.grad(cost_at, argnums=2), static_argnums=(5, 6))

np.random.seed(42)

def generate_fourier_plan(num_steps, num_frequencies):
    T = num_steps
    t = np.arange(T)
    
    a_k = np.random.uniform(-1.0, 1.0, num_frequencies)
    b_k = np.random.uniform(-1.0, 1.0, num_frequencies)
    
    plan = np.zeros(T)
    for k in range(num_frequencies):
        plan += a_k[k] * np.sin(2*np.pi*k*t/T) + b_k[k] * np.cos(2*np.pi*k*t/T)
    
    plan = 14 + 6*plan/np.max(np.abs(plan))
    return plan.reshape(-1, 1)
    
def make_plan(start_time, num_plans, num_steps, balloon, wind, atmosphere, waypoint_time_step, integration_time_step):
        
    best_plan = -1
    best_cost = +np.inf

    for _ in range(num_plans):
        plan = 13 + 9*np.random.rand(1)
        plan = np.full((num_steps, 1), plan)

        # plan = generate_fourier_plan(num_steps, 10)
        cost = cost_at(start_time, balloon, plan, wind, atmosphere, waypoint_time_step, integration_time_step)
        if cost < best_cost:
            best_plan = plan
            best_cost = cost

    return jnp.array(best_plan), best_cost

#@profile
def convert_plan_to_actions(plan, observation, i, atmosphere):
    i %= len(plan)
    _, _, _, pressure = observation
    height = atmosphere.at_pressure(pressure).height.km
    if abs(height - plan[i]) < 0.5:
        return 1 #STAY

    if height < plan[i]:
        return 2 # UP
    
    return 0

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
        
        self.plan_size = 240
        self.num_initializations=50
        self.plan = None
        self.i = 0
        self.waypoint_time_step = 3*60 # seconds, Equivalent to time_delta in BalloonArena

        self.integration_time_step = 10 # seconds, Equivalent to stride


    def begin_episode(self, observation: np.ndarray) -> int:
        # atmosnav optimizer:
        x = observation[1].km
        y = observation[2].km
        pressure = observation[3]
        t = observation[0].seconds

        # # t, x, y, pressure = observation
        balloon = make_weather_balloon(x, y, pressure, t, self.atmosphere, self.waypoint_time_step, self.integration_time_step)
        self.plan, best_cost = make_plan(t, self.num_initializations, self.plan_size, balloon, self.forecast, self.atmosphere, self.waypoint_time_step, self.integration_time_step)
        for i in range(100):
            dplan = gradient_at(t, balloon, self.plan, self.forecast, self.atmosphere, self.waypoint_time_step, self.integration_time_step)
            if abs(jnp.linalg.norm(dplan)) < 1e-7:
                break
            self.plan -= dplan / (np.linalg.norm(dplan) + 0.0001)
        
        self.i = 0
        action = convert_plan_to_actions(self.plan, observation, self.i, self.atmosphere)
        return action

    def step(self, reward, observation):
        REPLANNING = True
        if REPLANNING:
            N = 23
            if N==0 or (self.i > 0 and self.i%N == 0):
                return self.begin_episode(observation)
            else:
                self.i += 1
                action = convert_plan_to_actions(self.plan, observation, self.i, self.atmosphere)
                return action
        else:
            self.i += 1
            action = convert_plan_to_actions(self.plan, observation, self.i, self.atmosphere)
            return action

    def write_diagnostics(self, diagnostics):
        if 'mpc_agent' not in diagnostics:
            diagnostics['mpc_agent'] = {'z':[]}
        
        height = self.plan[self.i].item()

        diagnostics['mpc_agent']['z'].append(height)


    def end_episode(self, reward: float, terminal: bool = True) -> None:
        self.i = 0 

    def update_forecast(self, forecast: agent.WindField): 
        # self.forecast = SimpleJaxWindField()
        self.forecast = forecast.to_jax_wind_field()

    def update_atmosphere(self, atmosphere: agent.standard_atmosphere.Atmosphere): 
        self.atmosphere = atmosphere.to_jax_atmopshere() 

