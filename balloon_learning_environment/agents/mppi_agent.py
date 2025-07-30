import scipy.interpolate
from balloon_learning_environment.agents import agent, opd
import scipy.signal
from balloon_learning_environment.env.balloon.jax_balloon import JaxBalloon, JaxBalloonState
from balloon_learning_environment.env.wind_field import JaxWindField
from balloon_learning_environment.env.balloon.standard_atmosphere import JaxAtmosphere
from balloon_learning_environment.models import jax_perciatelli
import numpy as np
import jax
import jax.numpy as jnp
import scipy
from functools import partial
import time
import json

def inverse_sigmoid(x):
    return jnp.log((x+1)/(1-x))
"""Returns a sigmoid function with range [-1, 1] instead of [0,1]"""
def sigmoid(x):
    return 2 / (1 + jnp.exp(-x)) - 1

def jax_balloon_cost(balloon: JaxBalloon):
    r_2 = (balloon.state.x/1000)**2 + (balloon.state.y/1000)**2
    
    soc = balloon.state.battery_charge / balloon.state.battery_capacity
    
    battery_cost = 50**2 * (1 -  (1 / (1 + jnp.exp(-100*(soc - 0.1)))))

    return r_2 + battery_cost

class TerminalCost:
    def __call__(self, balloon: JaxBalloon, wind_forecast: JaxWindField):
        pass


class NoTerminalCost(TerminalCost):
    def __call__(self, balloon: JaxBalloon, wind_forecast: JaxWindField):
        return 0.0
    
    def tree_flatten(self):
        return (), {}
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return NoTerminalCost()

jax.tree_util.register_pytree_node_class(NoTerminalCost)
@partial(jax.jit, static_argnums=(5, 6))
def jax_plan_cost(plan, balloon, wind_field, atmosphere, terminal_cost_fn, time_delta, stride):
    return jax_plan_cost_no_jit(plan, balloon, wind_field, atmosphere, terminal_cost_fn, time_delta, stride)

def jax_plan_cost_no_jit(plan, balloon: JaxBalloon, wind_field: JaxWindField, atmosphere: JaxAtmosphere, terminal_cost_fn: TerminalCost, time_delta: 'int, seconds', stride: 'int, seconds'):
    cost = 0.0
    discount_factor = 0.99

    plan = sigmoid(plan)
    def update_step(i, balloon_and_cost: tuple[JaxBalloon, float]):
        balloon, cost = balloon_and_cost
        #jax.debug.print("update_step: i = {i}", i=i)

        wind_vector = wind_field.get_forecast(balloon.state.x/1000, balloon.state.y/1000, balloon.state.pressure, balloon.state.time_elapsed)
        
        # action = plan[i]
        action = jax.lax.cond((balloon.state.battery_charge/balloon.state.battery_capacity) < 0.025,
                     lambda op: jnp.astype(0.0,jnp.float64),
                     lambda op: op[0],
                     operand=(plan[i],))
        
        #jax.debug.print("action[{i}] = {a}", i=i, a=action)

        next_balloon = balloon.simulate_step_continuous_no_jit(wind_vector, atmosphere, action, time_delta, stride)

        #jax.debug.print("state[{i}] x = {x}, y = {y}, pressure = {p}, mols = {m}", i=i, x=next_balloon.state.x, y=next_balloon.state.y, p=next_balloon.state.pressure, m=next_balloon.state.mols_air)

        cost += (discount_factor**i) * jax_balloon_cost(next_balloon)

        return next_balloon, cost
    #jax.debug.print("entering loop")
    final_balloon, cost = jax.lax.fori_loop(0, len(plan), update_step, init_val=(balloon, cost))
    #jax.debug.print("final state x = {x}, y = {y}, pressure = {p}", x=final_balloon.state.x, y=final_balloon.state.y, p=final_balloon.state.pressure)
    terminal_cost = (discount_factor**len(plan)) * (jax_balloon_cost(final_balloon) + terminal_cost_fn(final_balloon, wind_field))
    return cost + terminal_cost


np.random.seed(seed=42)
def get_initial_plans(balloon: JaxBalloon, num_plans, forecast: JaxWindField, atmosphere: JaxAtmosphere, plan_steps, time_delta, stride):
    # flight_record = [(atmosphere.at_pressure(balloon.state.pressure).height.km, 0)]
    flight_record = {atmosphere.at_pressure(balloon.state.pressure).height.km.item(): 0}

    time_to_top = 0
    max_km_to_explore = 19.1

    up_balloon = balloon
    while time_to_top < plan_steps and atmosphere.at_pressure(up_balloon.state.pressure).height.km < max_km_to_explore:
        wind_vector = forecast.get_forecast(up_balloon.state.x/1000, up_balloon.state.y/1000, up_balloon.state.pressure, up_balloon.state.time_elapsed)
        up_balloon = up_balloon.simulate_step_continuous(wind_vector, atmosphere, 0.99, time_delta, stride)
        time_to_top += 1

        flight_record[atmosphere.at_pressure(up_balloon.state.pressure).height.km.item()] = time_to_top

    time_to_bottom = 0
    min_km_to_explore = 15.4

    down_balloon = balloon
    while time_to_bottom < plan_steps and atmosphere.at_pressure(down_balloon.state.pressure).height.km > min_km_to_explore:
        wind_vector = forecast.get_forecast(down_balloon.state.x/1000, down_balloon.state.y/1000, down_balloon.state.pressure, down_balloon.state.time_elapsed)
        down_balloon = down_balloon.simulate_step_continuous(wind_vector, atmosphere, -0.99, time_delta, stride)
        time_to_bottom += 1

        flight_record[atmosphere.at_pressure(down_balloon.state.pressure).height.km.item()] = time_to_bottom
    
    # sorted (should be)
    # flight_record = flight_record_down[::-1] + flight_record_up

    # Sort the dictionary by keys (altitudes) and split them into two separate lists
    sorted_flight_record = sorted(flight_record.items())

    flight_record_altitudes = [altitude for altitude, _ in sorted_flight_record]
    flight_record_steps = [steps for _, steps in sorted_flight_record]
    
    interpolator = scipy.interpolate.RegularGridInterpolator((flight_record_altitudes, ), flight_record_steps, bounds_error=False, fill_value=None)

    plans = []

    for i in range(num_plans):
        random_height = np.random.uniform(15.4, 19.1)
        going_up = random_height >= atmosphere.at_pressure(balloon.state.pressure).height.km
        steps = int(round(interpolator(np.array([random_height]))[0]))
        # print(steps)

        plan = np.zeros((plan_steps, ))
        plan[:steps] = +0.99 if going_up else -0.99 
        # print(random_height, steps)
        try:
            if steps < plan_steps:
                plan[steps:] += np.random.uniform(-0.3, 0.3, plan_steps - steps)
        except:
            print(atmosphere.at_pressure(balloon.state.pressure).height.km.item(), random_height, steps, plan_steps)

        plans.append(plan)
    
    return inverse_sigmoid(np.array(plans))


class MPPIAgent(agent.Agent):
        
    def __init__(self, num_actions: int, observation_shape): # Sequence[int]
        super(MPPIAgent, self).__init__(num_actions, observation_shape)
        self.forecast = None # WindField
        self.ble_atmosphere = None 
        self.atmosphere = None # Atmosphere

        self.mppi_K = 100  # Number of samples
        self.mppi_lambda = 0.25
        self.mppi_sigma = 0.1  # Standard deviation of noise

        self.plan_time = 2*24*60*60
        self.time_delta = 3*60
        self.stride = 10

        # self.plan_steps = 960 + 23 
        self.plan_steps = 240 # (self.plan_time // self.time_delta) // 3
        # self.N = self.plan_steps

        self.plan = None # jnp.full((self.plan_steps, ), fill_value=1.0/3.0)
        self.i = 0

        self.key = jax.random.key(seed=0)

        self.avg_opt_time=0
        self.curr_mean = 0

        self.balloon = None
        self.time = None
        self.steps_within_radius = 0
        self.terminal_cost_fn = NoTerminalCost()

    """
    calculates the current position of a moving object by using a previously determined position, or fix, and incorporating estimates of speed, heading, and elapsed time.
    an observation is a state. it has balloon_obersvation and wind information. we use ob_t and MPC predicts a_t which then gives us ob_t+1
    """
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

    np.random.seed(42)
    def mppi_optimize(self, nominal_plan, balloon, forecast, atmosphere, terminal_cost_fn, mean=0):
        K = self.mppi_K #num of samples

        plans = np.tile(nominal_plan, (K, 1)) + np.random.normal(mean, self.mppi_sigma, size=(K, self.plan_steps))
        
        plans = jnp.clip(plans, -4, 4)  # covers most of the inverse sigmoid space, should help with stability

        costs = []
        for k in range(K):
            plan_k = plans[k]
            cost_k = jax_plan_cost(plan_k, balloon, forecast, atmosphere, terminal_cost_fn, self.time_delta, self.stride)
            costs.append(cost_k)
        # updates mean and variance by averaging plans by combination of cost and noise term (noise is control term)
        # normally use the random noise is derivative of control
        #softmin weighting so each is between 0 or 1
        costs = np.array(costs)
        beta = np.min(costs)
        weights = np.exp((-1 / self.mppi_lambda) * (costs - beta))

        weights /= np.sum(weights)

        weighted_plan = np.average(plans, axis=0, weights=weights)
        #Apply SGF filter
        weighted_plan = scipy.signal.savgol_filter(weighted_plan, int(len(weighted_plan)/2), int(len(weighted_plan)/4))
        print(f"weighted plan is {weighted_plan[:15]}")
        #return weighted_plan, mean
        return weighted_plan

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

        if initialization_type == 'best_altitude':
            if self.plan == None:
                print('First')
            initial_plans = get_initial_plans(self.balloon, 100, self.forecast, self.atmosphere, self.plan_steps, self.time_delta, self.stride)
            
            batched_cost = [jax_plan_cost(init_plan, self.balloon, self.forecast, self.atmosphere, self.terminal_cost_fn, self.time_delta, self.stride) for init_plan in initial_plans]
            
            min_index_so_far = np.argmin(batched_cost)
            min_value_so_far = batched_cost[min_index_so_far]

            initial_plan = initial_plans[min_index_so_far]
            isUsingPrevPlan = False
            if self.plan is not None and jax_plan_cost(self.plan, self.balloon, self.forecast, self.atmosphere, self.terminal_cost_fn, self.time_delta, self.stride) < min_value_so_far:
                print('Using the previous optimized plan as initial plan')
                initial_plan = self.plan
                isUsingPrevPlan = True

            coast = inverse_sigmoid(np.random.uniform(-0.2, 0.2, size=(self.plan_steps, )))
            if jax_plan_cost(coast, self.balloon, self.forecast, self.atmosphere, self.terminal_cost_fn, self.time_delta, self.stride) < min_value_so_far:
                print('Using the nothing plan as initial plan')
                initial_plan = coast

        elif initialization_type == 'random':
            initial_plan = np.random.uniform(-1.0, 1.0, size=(self.plan_steps, ))
        else:
            initial_plan = np.zeros((self.plan_steps, ))

        # MPPI optimization
        b4 = time.time()
        # optimized_plan, next_mean = self.mppi_optimize(initial_plan, self.balloon, self.forecast, self.atmosphere, self.terminal_cost_fn, mean=self.curr_mean)
        # self.curr_mean = next_mean
        optimized_plan = self.mppi_optimize(initial_plan, self.balloon, self.forecast, self.atmosphere, self.terminal_cost_fn)
        self.plan = sigmoid(optimized_plan)
        after = time.time()
        print(after - b4, 's to get optimized plan')
        self.avg_opt_time += after-b4

        b4 = time.time()
        self._deadreckon()
        # print(time.time() - b4, 's to deadreckon ballooon')
        action = self.plan[self.i]
        print(f"plan is {self.plan[:15]}")
        print("done initializing episode")
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
            
            N = min(len(self.plan), 7)
            if self.i>0 and self.i%N==0:
                # self.plan_steps -= N
                self.key, rng = jax.random.split(self.key, 2)
                # self.plan = self.plan[N:]
                self.plan = inverse_sigmoid(jnp.hstack((self.plan[N:], jax.random.uniform(rng, (N, ), minval=-0.3, maxval=0.3))))

                print(self.plan.shape)
                self.i = 0
                start_cost = jax_plan_cost(self.plan, self.balloon, self.forecast, self.atmosphere, self.terminal_cost_fn, self.time_delta, self.stride)

                b4 = time.time()
                optimized_plan = self.mppi_optimize(self.plan, self.balloon, self.forecast, self.atmosphere, self.terminal_cost_fn)
                self.plan = sigmoid(optimized_plan)
                after = time.time()
                print(after - b4, 's to get optimized plan')
                self.avg_opt_time += after-b4

                action = self.plan[self.i]
                after_cost = jax_plan_cost(self.plan, self.balloon, self.forecast, self.atmosphere, self.terminal_cost_fn, self.time_delta, self.stride)
                print(f" ∆cost = {after_cost} - {start_cost} = {after_cost - start_cost}")
                print(f"plan is {self.plan[:15]}")
                print("done one step")
                return action.item()
                #return self.begin_episode(observation)
            else:
                # print('not replanning')
                self._deadreckon()
                action = self.plan[self.i]
                # print('action', action)
                return action.item()
            
    def write_diagnostics_start(self, observation, diagnostics):
        if 'mppi_agent' not in diagnostics:
            diagnostics['mppi_agent'] = {'x': [], 'y': [], 'z':[], 'wind':[], 'plan':[]}

        observation: JaxBalloonState = observation
        balloon = JaxBalloon(observation)
        
        height = self.atmosphere.at_pressure(balloon.state.pressure).height.km.item()

        diagnostics['mppi_agent']['x'].append(balloon.state.x.item()/1000)
        diagnostics['mppi_agent']['y'].append(balloon.state.y.item()/1000)
        diagnostics['mppi_agent']['z'].append(height)
        # diagnostics['mpc4_agent']['plan'].append(0.0)

        wind_vector = self.forecast.get_forecast(
            balloon.state.x.item()/1000, 
            balloon.state.y.item()/1000, 
            balloon.state.pressure.item(), 
            balloon.state.time_elapsed)
        diagnostics['mppi_agent']['wind'].append([wind_vector[0].item(), wind_vector[1].item()])

    
    def write_diagnostics(self, diagnostics):
        if 'mppi_agent' not in diagnostics:
            diagnostics['mppi_agent'] = {'x': [], 'y': [], 'z':[], 'wind':[], 'plan':[]}
        
        height = self.atmosphere.at_pressure(self.balloon.state.pressure).height.km.item()

        diagnostics['mppi_agent']['x'].append(self.balloon.state.x.item()/1000)
        diagnostics['mppi_agent']['y'].append(self.balloon.state.y.item()/1000)
        diagnostics['mppi_agent']['z'].append(height)
        diagnostics['mppi_agent']['plan'].append(self.plan[self.i].item())

        wind_vector = self.forecast.get_forecast(
            self.balloon.state.x/1000, 
            self.balloon.state.y/1000, 
            self.balloon.state.pressure, 
            self.balloon.state.time_elapsed)
        diagnostics['mppi_agent']['wind'].append([wind_vector[0].item(), wind_vector[1].item()])


    def write_diagnostics_end(self, diagnostics):
        if 'mppi_agent' not in diagnostics:
            diagnostics['mppi_agent'] = {'x': [], 'y': [], 'z':[], 'wind':[], 'plan':[]}
        
        X = diagnostics['mppi_agent']['x']
        if len(X) != 0:
            diagnostics['mppi_agent']['twr'] = self.steps_within_radius/len(X)
        else:
            diagnostics['mppi_agent']['twr'] = 0 
 
    def end_episode(self, reward: float, terminal: bool = True) -> None:
        self.i = 0
        self.steps_within_radius = 0
        self.balloon = None
        self.plan = None
        # self.plan_steps = 960 + 23

    def update_forecast(self, forecast: agent.WindField): 
        self.ble_forecast = forecast
        self.forecast = forecast.to_jax_wind_field()

    def update_atmosphere(self, atmosphere: agent.standard_atmosphere.Atmosphere): 
        self.ble_atmosphere = atmosphere
        self.atmosphere = atmosphere.to_jax_atmosphere() 
