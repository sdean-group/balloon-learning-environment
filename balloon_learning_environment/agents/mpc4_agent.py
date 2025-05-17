import scipy.interpolate
from balloon_learning_environment.agents import agent, opd
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

class QTerminalCost(TerminalCost):
    def __init__(self, num_wind_layers, distilled_params):
        self.num_wind_layers = num_wind_layers
        self.distilled_params = distilled_params

    def __call__(self, balloon: JaxBalloon, wind_forecast: JaxWindField):
        model = jax_perciatelli.DistilledNetwork()
        feature_vector = jax_perciatelli.jax_construct_feature_vector(balloon, wind_forecast, self.get_input_size(), self.num_wind_layers)
        q_vals = model.apply(self.distilled_params, feature_vector)
        terminal_cost = -(jnp.mean(q_vals)**2) # NOTE: can also test with max(Q_values)
        return terminal_cost
    
    def get_input_size(self):
        return jax_perciatelli.get_distilled_model_input_size(self.num_wind_layers)
    
    def tree_flatten(self): 
        return (self.distilled_params, ), {'num_wind_layers': self.num_wind_layers}

    @classmethod
    def tree_unflatten(cls, aux_data, children): 
        q_func = QTerminalCost(aux_data['num_wind_layers'], children[0])
        return q_func

jax.tree_util.register_pytree_node_class(QTerminalCost)

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

        wind_vector = wind_field.get_forecast(balloon.state.x/1000, balloon.state.y/1000, balloon.state.pressure, balloon.state.time_elapsed)
        
        # action = plan[i]
        action = jax.lax.cond(balloon.state.battery_charge/balloon.state.battery_capacity < 0.025,
                     lambda op: jnp.astype(0.0,jnp.float64),
                     lambda op: op[0],
                     operand=(plan[i],))

        next_balloon = balloon.simulate_step_continuous_no_jit(wind_vector, atmosphere, action, time_delta, stride)

        cost += (discount_factor**i) * jax_balloon_cost(next_balloon)

        return next_balloon, cost

    final_balloon, cost = jax.lax.fori_loop(0, len(plan), update_step, init_val=(balloon, cost))
    terminal_cost = (discount_factor**len(plan)) * (jax_balloon_cost(final_balloon) + terminal_cost_fn(final_balloon, wind_field))
    return cost + terminal_cost

def grad_descent_optimizer(initial_plan, dcost_dplan, balloon, forecast, atmosphere, terminal_cost_fn, time_delta, stride):
    start_cost = jax_plan_cost(initial_plan, balloon, forecast, atmosphere, terminal_cost_fn, time_delta, stride)
    plan = initial_plan
    for gradient_steps in range(100):
        dplan = dcost_dplan(plan, balloon, forecast, atmosphere, terminal_cost_fn, time_delta, stride)
        if  np.isnan(dplan).any() or abs(jnp.linalg.norm(dplan)) < 1e-7:
            # print('Exiting early, |∂plan| =',abs(jnp.linalg.norm(dplan)))
            break
        # print("A", gradient_steps, abs(jnp.linalg.norm(dplan)))
        plan -= dplan / jnp.linalg.norm(dplan)

    after_cost = jax_plan_cost(plan, balloon, forecast, atmosphere, terminal_cost_fn, time_delta, stride)
    print("GD", gradient_steps, f"∆cost = {after_cost} - {start_cost} = {after_cost - start_cost}")
    return plan

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


@partial(jax.jit, static_argnums=(5, 6))
@partial(jax.grad, argnums=0)
def get_dplan(plan, balloon: JaxBalloon, wind_field: JaxWindField, atmosphere: JaxAtmosphere, terminal_cost_fn: TerminalCost, time_delta, stride):
    # jax.debug.print("{balloon}, {wind_field}, {atmosphere}, {terminal_cost_fn}, {time_delta}, {stride}", balloon=balloon, wind_field=wind_field, atmosphere=atmosphere, terminal_cost_fn=terminal_cost_fn, time_delta=time_delta, stride=stride)
    return jax_plan_cost_no_jit(plan, balloon, wind_field, atmosphere, terminal_cost_fn, time_delta, stride)

class MPC4Agent(agent.Agent):
        
    def __init__(self, num_actions: int, observation_shape): # Sequence[int]
        super(MPC4Agent, self).__init__(num_actions, observation_shape)
        self.forecast = None # WindField
        self.ble_atmosphere = None 
        self.atmosphere = None # Atmosphere

        # self._get_dplan = jax.jit(jax.grad(jax_plan_cost, argnums=0), static_argnames=("time_delta", "stride"))

        self.plan_time = 2*24*60*60
        self.time_delta = 3*60
        self.stride = 10

        # self.plan_steps = 960 + 23 
        self.plan_steps = 8 # (self.plan_time // self.time_delta) // 3
        # self.N = self.plan_steps

        self.plan = None # jnp.full((self.plan_steps, ), fill_value=1.0/3.0)
        self.i = 0

        self.key = jax.random.key(seed=0)

        self.balloon = None
        self.time = None
        self.steps_within_radius = 0

        using_Q_function = True

        if using_Q_function:
            self.num_wind_levels = 181
            params = jax_perciatelli.get_distilled_perciatelli(self.num_wind_levels)[0]
            self.terminal_cost_fn = QTerminalCost(self.num_wind_levels, params)
        else:
            self.terminal_cost_fn = NoTerminalCost()

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
                # tmp = jax.make_jaxpr(jax_plan_cost, static_argnums=(5, 6))(initial_plans[i], self.balloon, self.forecast, self.atmosphere, self.terminal_cost_fn, self.time_delta, self.stride)
                # print(tmp)

                batched_cost.append(jax_plan_cost(initial_plans[i], self.balloon, self.forecast, self.atmosphere, self.terminal_cost_fn, self.time_delta, self.stride))
            
            min_index_so_far = np.argmin(batched_cost)
            min_value_so_far = batched_cost[min_index_so_far]

            initial_plan = initial_plans[min_index_so_far]
            if self.plan is not None and jax_plan_cost(self.plan, self.balloon, self.forecast, self.atmosphere, self.terminal_cost_fn, self.time_delta, self.stride) < min_value_so_far:
                print('Using the previous optimized plan as initial plan')
                initial_plan = self.plan

            coast = inverse_sigmoid(np.random.uniform(-0.2, 0.2, size=(self.plan_steps, )))
            if jax_plan_cost(coast, self.balloon, self.forecast, self.atmosphere, self.terminal_cost_fn, self.time_delta, self.stride) < min_value_so_far:
                print('Using the nothing plan as initial plan')
                initial_plan = coast

        elif initialization_type == 'random':
            initial_plan = np.random.uniform(-1.0, 1.0, size=(self.plan_steps, ))
        else:
            initial_plan = np.zeros((self.plan_steps, ))

        optimizing_on = True
        if optimizing_on:
            b4 = time.time()
            self.plan = grad_descent_optimizer(
                initial_plan, 
                get_dplan, 
                self.balloon, 
                self.forecast, 
                self.atmosphere,
                self.terminal_cost_fn,
                self.time_delta, 
                self.stride)
            print(time.time() - b4, 's to get optimized plan')
            self.plan = sigmoid(self.plan)
            print(time.time() - b4, 's to get optimized plan')
        else:
            self.plan = sigmoid(initial_plan)

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
            
            N = min(len(self.plan), 4)
            if self.i>0 and self.i%N==0:
                # self.plan_steps -= N
                self.key, rng = jax.random.split(self.key, 2)
                # self.plan = self.plan[N:]
                self.plan = inverse_sigmoid(jnp.hstack((self.plan[N:], jax.random.uniform(rng, (N, ), minval=-0.3, maxval=0.3))))
                print(self.plan.shape)
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

        diagnostics['mpc4_agent']['x'].append(balloon.state.x.item()/1000)
        diagnostics['mpc4_agent']['y'].append(balloon.state.y.item()/1000)
        diagnostics['mpc4_agent']['z'].append(height)
        # diagnostics['mpc4_agent']['plan'].append(0.0)

        wind_vector = self.forecast.get_forecast(
            balloon.state.x.item()/1000, 
            balloon.state.y.item()/1000, 
            balloon.state.pressure.item(), 
            balloon.state.time_elapsed)
        diagnostics['mpc4_agent']['wind'].append([wind_vector[0].item(), wind_vector[1].item()])

    
    def write_diagnostics(self, diagnostics):
        if 'mpc4_agent' not in diagnostics:
            diagnostics['mpc4_agent'] = {'x': [], 'y': [], 'z':[], 'wind':[], 'plan':[]}
        
        height = self.atmosphere.at_pressure(self.balloon.state.pressure).height.km.item()

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
        self.plan = None
        # self.plan_steps = 960 + 23

    def update_forecast(self, forecast: agent.WindField): 
        self.ble_forecast = forecast
        self.forecast = forecast.to_jax_wind_field()

    def update_atmosphere(self, atmosphere: agent.standard_atmosphere.Atmosphere): 
        self.ble_atmosphere = atmosphere
        self.atmosphere = atmosphere.to_jax_atmosphere() 


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
