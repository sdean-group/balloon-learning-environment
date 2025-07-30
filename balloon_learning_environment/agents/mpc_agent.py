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
import time

# imports jaxballoon file:
from balloon_learning_environment.env.balloon.jax_balloon import JaxBalloon, JaxBalloonState

import optax 

MIN_ALT, MAX_ALT = 15.1, 19.1

class DeterministicAltitudeModel(Dynamics):

    def __init__(self, integration_time_step):
        self.dt = integration_time_step
        self.vlim = 0.9
        # For descending, the maximum change is half that for ascending
        self.vlim_down = 0.35

    def control_input_to_delta_state(self, time, state, control_input, wind_vector):
        h = self.update(state, control_input)
        return jnp.array([wind_vector[0], wind_vector[1], h - state[2], 0.0]), self

    def update(self, state, waypoint):
        # Calculate the difference between target and current altitude.
        delta = waypoint - state[2]
        # Choose the appropriate velocity limit based on whether we are ascending or descending.
        # Ascending: use self.vlim; descending: use self.vlim_down.
        current_vlim = jnp.where(delta >= 0, self.vlim, self.vlim_down)
        # Compute the maximum allowed change for this time step (convert per hour rate to dt seconds).
        limit = current_vlim / 3600.0 * self.dt
        
        # Update altitude using jax.lax.cond:
        # If the absolute difference is larger than 'limit', move by 'limit' in the correct direction.
        # Otherwise, set altitude directly to the waypoint.
        h = jax.lax.cond(jnp.abs(delta) > limit,
            lambda op: op[0][2] + limit * jnp.sign(op[1] - op[0][2]),
            lambda op: op[1],
            operand=(state, waypoint))
        
        h = jnp.clip(h, MIN_ALT, MAX_ALT)
        return h

    def tree_flatten(self):
        children = ()  # arrays / dynamic values
        aux_data = {'dt': self.dt, 'vlim': self.vlim, 'vlim_down': self.vlim_down}  # static values
        return (children, aux_data)
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        # Reconstruct the model and restore all static attributes.
        model = DeterministicAltitudeModel(aux_data['dt'])
        model.vlim = aux_data['vlim']
        model.vlim_down = aux_data['vlim_down']
        return model


def make_weather_balloon(init_lat, init_lon, init_pressure, start_time, atmosphere, waypoint_time_step, integration_time_step):
    return Airborne(
        jnp.array([ init_lat, init_lon, atmosphere.at_pressure(init_pressure).height.km, 0.0 ]),
        PlanToWaypointController(start_time, waypoint_time_step),
        DeterministicAltitudeModel(integration_time_step))

@partial(jax.jit, static_argnums=(6, 7))
def cost_at(start_time, balloon, plan_0, plan, wind, atmosphere, waypoint_time_step, integration_time_step, ):
    N = (waypoint_time_step * (len(plan)-1)) // integration_time_step
    factor = 1.0
    cost = 0.0
    plan_change_cost = 0.0

    plan = jnp.concatenate([jnp.array([plan_0]), plan])

    def inner_run(i, time_balloon_cost):
        time, balloon, cost, factor, plan_change_cost = time_balloon_cost
        x, y, altitude, _ = balloon.state
        
        cost += factor * (balloon.state[0]**2 + balloon.state[1]**2)
        factor *= 0.99
        
        # plan_change_penalty = 50 * jax.lax.cond(i > 0, 
        #                                    lambda: (plan[i] - plan[i-1]) ** 2, 
        #                                    lambda: 0.0)
        # plan_change_cost += plan_change_penalty
        
        pressure = atmosphere.at_height(height_meters=altitude * 1000.0).pressure
        wind_vector = integration_time_step * wind.get_forecast(x, y, pressure, time) / 1000.0
        next_balloon, _ = balloon.step(time, plan, wind_vector)
        
        return time + integration_time_step, next_balloon, cost, factor, plan_change_cost
    
    # t, balloon, _, factor, _ = inner_run(0, (start_time, balloon, 0.0, factor, 0.0))

    t, final_balloon, cost, factor, plan_change_cost = jax.lax.fori_loop(0, N, inner_run, 
                                                                         init_val=(start_time, balloon, cost, factor, plan_change_cost))
    
    total_cost = cost + plan_change_cost
    return total_cost

gradient_at = jax.jit(jax.grad(cost_at, argnums=3), static_argnums=(6, 7))

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
    is_awesome = True

    best_plan = -1
    best_cost = +np.inf

    for _ in range(num_plans):

        if is_awesome:
            x, y, altitude, _ = balloon.state

            target = MIN_ALT + (MAX_ALT - MIN_ALT)*np.random.rand()
            delta = (target - altitude)
            vlim = 0.9 if delta > 0 else 0.35
            limit = vlim / 3600.0 * waypoint_time_step
            steps_to_reach = int(abs(delta)/limit + 1)
            
            # print(steps_to_reach>=240)

            start_plan = np.linspace(altitude, target, steps_to_reach)
            end_plan = np.full((num_steps - steps_to_reach), target)

            # Concatenate them along axis 0 to form a (num_steps, 1) array
            plan = np.concatenate([start_plan, end_plan], axis=0)
        else:
            plan = MIN_ALT + (MAX_ALT - MIN_ALT)*np.random.rand(1)
            plan = np.full(num_steps, plan)

        # plan = generate_fourier_plan(num_steps, 10)

        cost = cost_at(start_time, balloon, plan[0], plan[1:], wind, atmosphere, waypoint_time_step, integration_time_step)
        if cost < best_cost:
            best_plan = plan
            best_cost = cost

    return best_plan[0], best_plan[1:], best_cost


#@profile
def convert_plan_to_actions(plan, observation, i, atmosphere):
    i %= len(plan)
    _, _, _, pressure = observation
    height = atmosphere.at_pressure(pressure).height.km
    if abs(height - plan[i]) < 0.10:
        return 1 #STAY

    if height < plan[i]:
        return 2 # UP
    
    return 0

#  DOWN = 0
#  STAY = 1
#  UP = 2

# Idea: use observations to improve forecast (like perciatelli feature uses WindGP)    

@partial(jax.jit, static_argnames=("waypoint_time_step", "integration_time_step"))
def _deadreckon_jax(balloon, time, plan, forecast, atmosphere, waypoint_time_step, integration_time_step):
    # Number of integration steps per waypoint.
    N = waypoint_time_step // integration_time_step

    def body_fun(i, state):
        t, balloon = state
        # Extract state components (assumed to be [x, y, altitude, ...])
        x, y, altitude, _ = balloon.state

        # Compute the atmospheric pressure at the current altitude.
        pressure = atmosphere.at_height(height_meters=altitude * 1000.0).pressure

        # Compute the wind vector scaled by the integration time step and convert from m/s to km.
        wind_vector = integration_time_step * forecast.get_forecast(x, y, pressure, t) / 1000.0

        # Advance the balloon state using its step method.
        new_balloon, _ = balloon.step(t, plan, wind_vector)
        new_t = t + integration_time_step

        return (new_t, new_balloon)

    final_time, final_balloon = jax.lax.fori_loop(0, N, body_fun, (time, balloon))
    return final_balloon, final_time

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
        self.avg_iters = 0
        self.avg_opt_time = 0
        self.waypoint_time_step = 3*60 # seconds, Equivalent to time_delta in BalloonArena

        self.integration_time_step = 10 # seconds, Equivalent to stride


        self.balloon = None
        self.time = None
        self.steps_within_radius = 0
        self.j = 0

    def _deadreckon(self):
        # Call the jitted dead reckoning function.
        final_balloon, final_time = _deadreckon_jax(
            self.balloon,
            self.time,
            self.plan,
            self.forecast,
            self.atmosphere,
            self.waypoint_time_step,
            self.integration_time_step
        )
        
        # Update the instance's balloon and time with the results.
        self.balloon = final_balloon
        self.time = final_time

        # Check if the balloon is within a radius of 50.0 (using x^2+y^2)
        x, y, _, _ = self.balloon.state
        if (x**2 + y**2) <= (50.0)**2:
            self.steps_within_radius += 1



    def begin_episode(self, observation: np.ndarray) -> int:
        x = observation[1].km
        y = observation[2].km
        pressure = observation[3]
        self.time = int(observation[0].total_seconds()) 
        # print('time given', self.time/3600)

        # # t, x, y, pressure = observation
        self.balloon = make_weather_balloon(
            # x if self.balloon is None or self.j%(1000) == 0 else self.balloon.state[0],
            # y if self.balloon is None or self.j%(1000) == 0 else self.balloon.state[1], 
            x,
            y,
            pressure, 
            self.time, 
            self.atmosphere, 
            self.waypoint_time_step, 
            self.integration_time_step)
        
        self.j+=1 
        # path_noise = np.random.uniform(-1, 1, size=(self.plan_size, 1))
        # self.plan = np.full((self.plan_size, 1), fill_value=self.atmosphere.at_pressure(pressure).height.km.item())
        
        plan_0, plan, best_cost = make_plan(self.time, self.num_initializations, self.plan_size, self.balloon, self.forecast, self.atmosphere, self.waypoint_time_step, self.integration_time_step)

        if False:
            learning_rate = 0.01
            optimizer = optax.adam(learning_rate)
            
            opt_state = optimizer.init(plan)

            # Optimization step function
            @jax.jit 
            def optimization_step(plan, opt_state):
                dplan = gradient_at(
                    self.time, self.balloon, plan_0, plan,
                    self.forecast, self.atmosphere,
                    self.waypoint_time_step, self.integration_time_step
                )
                
                # Compute updates
                updates, opt_state = optimizer.update(dplan, opt_state, plan)
                
                # Apply updates to the plan
                plan = optax.apply_updates(plan, updates)

                return plan, opt_state, jnp.linalg.norm(dplan)

            # Run optimization loop
            for iters in range(500):
                plan, opt_state, grad_norm = optimization_step(plan, opt_state)
                if grad_norm < 1e-7:
                    break
            print('Took', (iters + 1) ,'iterations')
        else:
            start = time.time()
            for iters in range(500):
                dplan = gradient_at(self.time, self.balloon, plan_0, plan, self.forecast, self.atmosphere, self.waypoint_time_step, self.integration_time_step)
                if abs(jnp.linalg.norm(dplan)) < 1e-7:
                    break
                plan -= 0.01 * dplan / jnp.linalg.norm(dplan)
            end = time.time()
            print('Took', (iters + 1) ,'iterations')
            self.avg_iters += iters + 1
            self.avg_opt_time += end-start
            # pass

        self.plan = np.concatenate([np.array([plan_0]), np.array(plan)])
        self.i = 0
        action = convert_plan_to_actions(self.plan, observation, self.i, self.atmosphere)

        self._deadreckon()
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
                self._deadreckon()
                return action
        else:
            self.i += 1
            action = convert_plan_to_actions(self.plan, observation, self.i, self.atmosphere)
            self._deadreckon()
            return action

    def write_diagnostics_start(self, observation, diagnostics):
        if 'mpc_agent' not in diagnostics:
            diagnostics['mpc_agent'] = {'x': [], 'y': [], 'z':[], 'wind':[], 'plan':[]}

        x = observation[1].km
        y = observation[2].km
        pressure = observation[3]
        time = int(observation[0].total_seconds()) 

        balloon = make_weather_balloon(
            x,
            y,
            pressure, 
            time, 
            self.atmosphere, 
            self.waypoint_time_step, 
            self.integration_time_step)


        diagnostics['mpc_agent']['x'].append(balloon.state[0].item())
        diagnostics['mpc_agent']['y'].append(balloon.state[1].item())
        diagnostics['mpc_agent']['z'].append(balloon.state[2].item())
        # diagnostics['mpc4_agent']['plan'].append(0.0)

        wind_vector = self.forecast.get_forecast(
            balloon.state[0]/1000, 
            balloon.state[1]/1000, 
            self.atmosphere.at_height(balloon.state[2]*1000).pressure, 
            time)
        diagnostics['mpc_agent']['wind'].append([wind_vector[0].item(), wind_vector[1].item()])


    def write_diagnostics(self, diagnostics):
        if 'mpc_agent' not in diagnostics:
            diagnostics['mpc_agent'] = {'x': [], 'y': [], 'z': [], 'wind':[], 'plan':[]}
        
        plan_i = self.plan[min(self.i, len(self.plan) -1)].item()

        diagnostics['mpc_agent']['x'].append(self.balloon.state[0].item())
        diagnostics['mpc_agent']['y'].append(self.balloon.state[1].item())
        diagnostics['mpc_agent']['z'].append(self.balloon.state[2].item())
        
        wind_vector = self.forecast.get_forecast(
            self.balloon.state[0], 
            self.balloon.state[1], 
            self.atmosphere.at_height(self.balloon.state[2]*1000).pressure, 
            self.time)
        diagnostics['mpc_agent']['wind'].append([wind_vector[0].item(), wind_vector[1].item()])

        wind_vector = None

        diagnostics['mpc_agent']['plan'].append(plan_i)

    def write_diagnostics_end(self, diagnostics):
        if 'mpc_agent' not in diagnostics:
            diagnostics['mpc_agent'] = {'x': [], 'y': [], 'z': [], 'plan':[]}
        
        X = diagnostics['mpc_agent']['x']
        if len(X) != 0:
            diagnostics['mpc_agent']['twr'] = self.steps_within_radius/len(X)
        else:
            diagnostics['mpc_agent']['twr'] = 0 
        


    def end_episode(self, reward: float, terminal: bool = True) -> None:
        self.i = 0 
        self.steps_within_radius = 0
        self.balloon = None
        self.j = 0
        self.avg_iters = 0
        self.avg_opt_time = 0

    def update_forecast(self, forecast: agent.WindField): 
        # self.forecast = SimpleJaxWindField()
        self.forecast = forecast.to_jax_wind_field()

    def update_atmosphere(self, atmosphere: agent.standard_atmosphere.Atmosphere): 
        self.atmosphere = atmosphere.to_jax_atmosphere() 



class Deadreckon(agent.Agent):
    """An agent that takes uniform random actions."""

    def __init__(self, num_actions: int, observation_shape: Sequence[int]):
        super(Deadreckon, self).__init__(num_actions, observation_shape)
        self.forecast = None
        self.atmosphere = None
        
        self.plan_size = 240
        self.num_initializations=50
        self.plan = None
        self.i = 0
        self.waypoint_time_step = 3*60 # seconds, Equivalent to time_delta in BalloonArena

        self.integration_time_step = 10 # seconds, Equivalent to stride


        self.balloon = None
        self.time = None
        self.steps_within_radius = 0
        self.j = 0

    def _deadreckon(self):
        for _ in range(self.waypoint_time_step//self.integration_time_step):
            wind_vector = self.forecast.get_forecast(self.balloon[0], self.balloon[1], self.balloon[2], self.time)
            self.balloon[0] += wind_vector.u * dt.timedelta(seconds=self.integration_time_step)
            self.balloon[1] += wind_vector.v * dt.timedelta(seconds=self.integration_time_step)
            # self.balloon[2] += 0.0
            self.time += dt.timedelta(seconds=self.integration_time_step)
        
        x,y,_ = self.balloon
        if (x.km**2 + y.km**2) <= (50.0)**2:
            self.steps_within_radius += 1

    def begin_episode(self, observation: np.ndarray) -> int:
        x = observation[1]
        y = observation[2]
        pressure = observation[3]
        self.time = observation[0]

        
        # # t, x, y, pressure = observation
        self.balloon =[ x, y, pressure ]
        
        self.j+=1 
        self.plan = np.full((self.plan_size, 1), fill_value=self.atmosphere.at_pressure(pressure).height.km)
        
        self.i = 0
        action = 1 # stay

        self._deadreckon()
        return action

    def step(self, reward, observation):
        return self.begin_episode(observation)

    def write_diagnostics(self, diagnostics):
        if 'deadreckon' not in diagnostics:
            diagnostics['deadreckon'] = {'x': [], 'y': [], 'z': [], 'wind':[], 'plan':[]}
        
        plan_i = self.plan[min(self.i, len(self.plan) -1), 0]

        diagnostics['deadreckon']['x'].append(self.balloon[0].km)
        diagnostics['deadreckon']['y'].append(self.balloon[1].km)
        diagnostics['deadreckon']['z'].append(self.atmosphere.at_pressure(self.balloon[2]).height.km)

        wind_vector= self.forecast.get_forecast(self.balloon[0], self.balloon[1], self.balloon[2], self.time)
        diagnostics['deadreckon']['wind'].append([wind_vector.u.meters_per_second, wind_vector.v.meters_per_second])
        
        diagnostics['deadreckon']['plan'].append(plan_i)

    def write_diagnostics_end(self, diagnostics):
        if 'deadreckon' not in diagnostics:
            diagnostics['deadreckon'] = {'x': [], 'y': [], 'z': [], 'plan':[]}
        
        X = diagnostics['deadreckon']['x']
        if len(X) != 0:
            diagnostics['deadreckon']['twr'] = self.steps_within_radius/len(X)
        else:
            diagnostics['deadreckon']['twr'] = 0 
        
    def end_episode(self, reward: float, terminal: bool = True) -> None:
        self.i = 0 
        self.steps_within_radius = 0
        self.balloon = None
        self.j = 0

    def update_forecast(self, forecast: agent.WindField): 
        # self.forecast = SimpleJaxWindField()
        self.forecast = forecast

    def update_atmosphere(self, atmosphere: agent.standard_atmosphere.Atmosphere): 
        self.atmosphere = atmosphere 

