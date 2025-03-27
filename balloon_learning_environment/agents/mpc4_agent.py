import jax.scipy.optimize
import scipy.optimize
from balloon_learning_environment.agents import agent, opd
from balloon_learning_environment.env.balloon.jax_balloon import JaxBalloon, JaxBalloonState
from balloon_learning_environment.env.wind_field import JaxWindField
from balloon_learning_environment.utils import units
from balloon_learning_environment.env.balloon.standard_atmosphere import JaxAtmosphere
import numpy as np
import jax
import datetime as dt
import jax.numpy as jnp
import scipy
from functools import partial
import time
import json

def inverse_sigmoid(x):
    return jnp.log((x+1)/(1-x))

def sigmoid(x):
    return 2 / (1 + jnp.exp(-x)) - 1

def jax_balloon_cost(balloon: JaxBalloon):
    return (balloon.state.x/1000)**2 + (balloon.state.y/1000)**2

@partial(jax.jit, static_argnums=(-2, -1))
def jax_plan_cost(plan, balloon: JaxBalloon, wind_field: JaxWindField, atmosphere: JaxAtmosphere, time_delta: 'int, seconds', stride: 'int, seconds'):
    cost = 0.0
    discount_factor = 0.99 # 1.00

    plan = sigmoid(plan)
    
    def update_step(i, balloon_and_cost: tuple[JaxBalloon, float]):
        balloon, cost = balloon_and_cost

        wind_vector = wind_field.get_forecast(balloon.state.x/1000, balloon.state.y/1000, balloon.state.pressure, balloon.state.time_elapsed)
        
        next_balloon = balloon.simulate_step_continuous(wind_vector, atmosphere, plan[i], time_delta, stride)
        
        cost += (discount_factor**i) * jax_balloon_cost(next_balloon)

        return next_balloon, cost

    final_balloon, final_cost = jax.lax.fori_loop(0, len(plan), update_step, init_val=(balloon, cost))
    return final_cost

def jax_plan_reward(balloon: JaxBalloon):
    dist_km = (balloon.state.x/1000)**2  + (balloon.state.y/1000)**2
    shift = 50
    return (-(dist_km * 4) + shift**2)/(shift**2)

@partial(jax.jit, static_argnums=(-2, -1))
def jax_plan_reward_with_V_function(plan, balloon: JaxBalloon, wind_field: JaxWindField, atmosphere: JaxAtmosphere, v_function, time_delta: 'int, seconds', stride: 'int, seconds'):
    reward = 0.0
    discount_factor = 0.99

    plan = sigmoid(plan)
    def update_step(i, balloon_and_reward: tuple[JaxBalloon, float]):
        balloon, reward = balloon_and_reward
        wind_vector = wind_field.get_forecast(balloon.state.x/1000, balloon.state.y/1000, balloon.state.pressure, balloon.state.time_elapsed)
        next_balloon = balloon.simulate_step_continuous(wind_vector, atmosphere, plan[i], time_delta, stride)
        reward += (discount_factor**i) * jax_plan_reward(next_balloon)
        return next_balloon, reward

    final_balloon, final_cost = jax.lax.fori_loop(0, len(plan), update_step, init_val=(balloon, reward))
    return final_cost


def grad_descent_optimizer(initial_plan, dcost_dplan, balloon, forecast, atmosphere, time_delta, stride):
    start_cost = jax_plan_cost(initial_plan, balloon, forecast, atmosphere, time_delta, stride)
    plan = initial_plan
    for gradient_steps in range(100):
        dplan = dcost_dplan(plan, balloon, forecast, atmosphere, time_delta, stride)
        if  np.isnan(dplan).any() or abs(jnp.linalg.norm(dplan)) < 1e-7:
            # print('Exiting early, |∂plan| =',abs(jnp.linalg.norm(dplan)))
            break
        # print("A", gradient_steps, abs(jnp.linalg.norm(dplan)))
        plan -= dplan / jnp.linalg.norm(dplan)

    after_cost = jax_plan_cost(plan, balloon, forecast, atmosphere, time_delta, stride)
    print("GD", gradient_steps, f"∆cost = {after_cost} - {start_cost} = {after_cost - start_cost}")
    return plan

np.random.seed(seed=42)
def get_initial_plans(balloon: JaxBalloon, num_plans, forecast: JaxWindField, atmosphere: JaxAtmosphere, plan_steps, time_delta, stride):
    time_to_top = 0
    max_km_to_explore = 19.1
    # print('a')
    up_balloon = balloon
    while time_to_top < plan_steps and atmosphere.at_pressure(up_balloon.state.pressure).height.km < max_km_to_explore:
        wind_vector = forecast.get_forecast(up_balloon.state.x/1000, up_balloon.state.y/1000, up_balloon.state.pressure, up_balloon.state.time_elapsed)
        up_balloon = up_balloon.simulate_step_continuous(wind_vector, atmosphere, 1.0, time_delta, stride)
        time_to_top += 1

    time_to_bottom = 0
    min_km_to_explore = 15.4
    # print('b')
    down_balloon = balloon
    while time_to_bottom < plan_steps and atmosphere.at_pressure(down_balloon.state.pressure).height.km > min_km_to_explore:
        wind_vector = forecast.get_forecast(down_balloon.state.x/1000, down_balloon.state.y/1000, down_balloon.state.pressure, down_balloon.state.time_elapsed)
        down_balloon = down_balloon.simulate_step_continuous(wind_vector, atmosphere, -1.0, time_delta, time_delta)
        time_to_bottom += 1
    # print('c')

    plans = []

    for i in range(num_plans//2):
        up_plan = np.zeros((plan_steps, ))
        up_time = np.random.randint(0, max(1, time_to_top))
        up_plan[:up_time] = 0.99
        up_plan[up_time:] += np.random.uniform(-0.3, 0.3, plan_steps - up_time)

        plans.append(up_plan)

        down_plan = np.zeros((plan_steps, ))
        down_time = np.random.randint(0, max(1, time_to_bottom))
        down_plan[:down_time] = -0.99
        down_plan[down_time:] += np.random.uniform(-0.3, 0.3, plan_steps - down_time)

        plans.append(down_plan)
    
    # print('d')
    
    return inverse_sigmoid(jnp.array(plans))


class MPC4Agent(agent.Agent):
        
    def __init__(self, num_actions: int, observation_shape): # Sequence[int]
        super(MPC4Agent, self).__init__(num_actions, observation_shape)
        self.forecast = None # WindField
        self.ble_atmosphere = None 
        self.atmosphere = None # Atmosphere

        self.get_dplan = jax.jit(jax.grad(jax_plan_cost, argnums=0), static_argnums=(-2, -1))

        self.plan_time = 2*24*60*60
        self.time_delta = 3*60
        self.stride = 10

        self.plan_steps = 240 # (self.plan_time // self.time_delta) // 3

        self.plan = None # jnp.full((self.plan_steps, ), fill_value=1.0/3.0)
        self.i = 0

        self.key = jax.random.key(seed=0)

        self.balloon = None
        self.time = None
        self.steps_within_radius = 0

    def _deadreckon(self):
        # wind_vector = self.ble_forecast.get_forecast(
        #     units.Distance(meters=self.balloon.state.x),
        #     units.Distance(meters=self.balloon.state.y), 
        #     self.balloon.state.pressure,
        #     dt.datetime())
        
        # wind_vector = wind_vector.u.meters_per_second, wind_vector.v.meters_per_second
        
        wind_vector = self.forecast.get_forecast(
            self.balloon.state.x/1000, 
            self.balloon.state.y/1000, 
            self.balloon.state.pressure, 
            self.balloon.state.time_elapsed)
    
        # print(self.balloon.state.time_elapsed/3600.0)

        # print(self.balloon.state.time_elapsed)
        self.balloon = self.balloon.simulate_step_continuous(
            wind_vector, 
            self.atmosphere, 
            self.plan[self.i], 
            self.time_delta, 
            self.stride)
        
        if (self.balloon.state.x/1000)**2 + (self.balloon.state.y/1000)**2 <= (50.0)**2:
            self.steps_within_radius += 1

    def begin_episode(self, observation: np.ndarray) -> int:
        # TODO: actually convert observation into an ndarray (it is a JaxBalloonState, see features.py)
        # balloon = JaxBalloon(jax_balloon_state_from_observation(observation))

        observation: JaxBalloonState = observation
        # if self.balloon is not None:
        #     observation.x = self.balloon.state.x
        #     observation.y = self.balloon.state.y
        self.balloon = JaxBalloon(observation)

        # current_plan_cost = jax_plan_cost(self.plan, balloon, self.forecast, self.atmosphere, self.time_delta, self.stride)
        #if current_plan_cost < best_random_cost:
        #    initial_plan = self.plan

        # TODO: is it necessary to pass in forecast when just trying to get to a height?
        
        initialization_type = 'best_altitude'
        print('USING ' + initialization_type + ' INITIALIZATION')

        if initialization_type == 'opd':
            start = opd.ExplorerState(
                self.balloon.state.x,
                self.balloon.state.y,
                self.balloon.state.pressure,
                self.balloon.state.time_elapsed)

            search_delta_time = 60*60
            best_node, best_node_early = opd.run_opd_search(start, self.forecast, [0, 1, 2], opd.ExplorerOptions(budget=25_000, planning_horizon=240, delta_time=search_delta_time))
            initial_plan =  opd.get_plan_from_opd_node(best_node, search_delta_time=search_delta_time, plan_delta_time=self.time_delta)

        elif initialization_type == 'best_altitude':
            initial_plans = get_initial_plans(self.balloon, 100, self.forecast, self.atmosphere, self.plan_steps, self.time_delta, self.stride)
            batched_cost = []
            for i in range(len(initial_plans)):
                batched_cost.append(jax_plan_cost(jnp.array(initial_plans[i]), self.balloon, self.forecast, self.atmosphere, self.time_delta, self.stride))
            initial_plan = initial_plans[np.argmin(batched_cost)]

            # print(np.min(batched_cost))
            initial_plan = initial_plans[np.argmin(batched_cost)]
            # print(time.time() - b4, 's to get minimum cost plan')
        elif initialization_type == 'random':
            initial_plan = np.random.uniform(-1.0, 1.0, size=(self.plan_steps, ))
        else:
            initial_plan = np.zeros((self.plan_steps, ))

        optimizing_on = True
        if optimizing_on:
            b4 = time.time()
            self.plan = grad_descent_optimizer(
                initial_plan, 
                self.get_dplan, 
                self.balloon, 
                self.forecast, 
                self.atmosphere,
                self.time_delta, 
                self.stride)
            print(time.time() - b4, 's to get optimized plan')
            self.plan = sigmoid(self.plan)
            print(time.time() - b4, 's to get optimized plan')
        else:
            self.plan = initial_plan

        self.i = 0

        b4 = time.time()
        self._deadreckon()
        # print(time.time() - b4, 's to deadreckon ballooon')

        action = self.plan[self.i]
        # print('action', action)
        return action.item()

    def step(self, reward: float, observation: np.ndarray) -> int:
        REPLANNING = True
        observation: JaxBalloonState = observation
        self.i+=1
        # self._deadreckon()
        # print(observation.battery_charge/observation.battery_capacity)
        if not REPLANNING:
            self._deadreckon()
            action = self.plan[self.i]
            return action.item()
        else:
            N = min(len(self.plan), 23)
            if self.i>0 and self.i%N==0:
                # self.plan = jnp.vstack((self.plan[N:], jax.random.uniform(self.key, (N, ))))
                return self.begin_episode(observation)
            else:
                # print('not replanning')
                self._deadreckon()
                action = self.plan[self.i]
                # print('action', action)
                return action.item()
            
    def write_diagnostics_start(self, observation, diagnostics):
        if 'mpc4_agent' not in diagnostics:
            diagnostics['mpc4_agent'] = {'x': [], 'y': [], 'z':[], 'wind':[], 'plan':[]}

        observation: JaxBalloonState = observation
        balloon = JaxBalloon(observation)
        
        height = self.atmosphere.at_pressure(balloon.state.pressure).height.km.item()

        diagnostics['mpc4_agent']['x'].append(balloon.state.x/1000)
        diagnostics['mpc4_agent']['y'].append(balloon.state.y/1000)
        diagnostics['mpc4_agent']['z'].append(height)
        # diagnostics['mpc4_agent']['plan'].append(0.0)

        wind_vector = self.forecast.get_forecast(
            balloon.state.x/1000, 
            balloon.state.y/1000, 
            balloon.state.pressure, 
            balloon.state.time_elapsed)
        diagnostics['mpc4_agent']['wind'].append([wind_vector[0].item(), wind_vector[1].item()])

    
    def write_diagnostics(self, diagnostics):
        if 'mpc4_agent' not in diagnostics:
            diagnostics['mpc4_agent'] = {'x': [], 'y': [], 'z':[], 'wind':[], 'plan':[]}
        
        height = self.atmosphere.at_pressure(self.balloon.state.pressure).height.km.item()
        # height = self.ble_atmosphere.at_pressure(self.balloon.state.pressure).height.km.item()

        diagnostics['mpc4_agent']['x'].append(self.balloon.state.x.item()/1000)
        diagnostics['mpc4_agent']['y'].append(self.balloon.state.y.item()/1000)
        diagnostics['mpc4_agent']['z'].append(height)
        diagnostics['mpc4_agent']['plan'].append(self.plan[self.i].item())

        wind_vector = self.forecast.get_forecast(
            self.balloon.state.x/1000, 
            self.balloon.state.y/1000, 
            self.balloon.state.pressure, 
            self.balloon.state.time_elapsed)
        diagnostics['mpc4_agent']['wind'].append([wind_vector[0].item(), wind_vector[1].item()])


    def write_diagnostics_end(self, diagnostics):
        if 'mpc4_agent' not in diagnostics:
            diagnostics['mpc4_agent'] = {'x': [], 'y': [], 'z':[], 'wind':[], 'plan':[]}
        
        X = diagnostics['mpc4_agent']['x']
        if len(X) != 0:
            diagnostics['mpc4_agent']['twr'] = self.steps_within_radius/len(X)
        else:
            diagnostics['mpc4_agent']['twr'] = 0 
 
    def end_episode(self, reward: float, terminal: bool = True) -> None:
        self.i = 0
        self.steps_within_radius = 0
        self.balloon = None

    def update_forecast(self, forecast: agent.WindField): 
        self.ble_forecast = forecast
        self.forecast = forecast.to_jax_wind_field()

    def update_atmosphere(self, atmosphere: agent.standard_atmosphere.Atmosphere): 
        self.ble_atmosphere = atmosphere
        self.atmosphere = atmosphere.to_jax_atmopshere() 


class MPC4FollowerAgent(agent.Agent):
        
    def __init__(self, num_actions: int, observation_shape): # Sequence[int]
        super(MPC4FollowerAgent, self).__init__(num_actions, observation_shape)
        self.i = 0

        datapath = "diagnostics/MPC4Agent-1740952765564.json"
        agent_name = 'mpc4_agent'
        diagnostics = json.load(open(datapath, 'r'))

        self.plan = diagnostics["0"]['rollout'][agent_name]['plan']

    def begin_episode(self, observation: np.ndarray) -> int:
        action = self.plan[self.i]
        self.i += 1
        return action

    def step(self, reward: float, observation: np.ndarray) -> int:
        return self.begin_episode(observation)

    def end_episode(self, reward: float, terminal: bool = True) -> None:
        self.i = 0
