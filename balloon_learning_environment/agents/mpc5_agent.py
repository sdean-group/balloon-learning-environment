import scipy.interpolate
from balloon_learning_environment.agents import agent, opd
from balloon_learning_environment.env.balloon.control import AltitudeControlCommand
from balloon_learning_environment.env.balloon.jax_balloon import JaxBalloon, JaxBalloonState, JaxBalloonDynamicsParams
from balloon_learning_environment.env.wind_field import JaxWindField
from balloon_learning_environment.env import features
from balloon_learning_environment.env.balloon.standard_atmosphere import JaxAtmosphere
from balloon_learning_environment.models import jax_perciatelli
import numpy as np
import jax
import jax.numpy as jnp
import scipy
from functools import partial
import time
import json

from balloon_learning_environment.env import wind_gp
from balloon_learning_environment.env.grid_based_wind_field import JaxColumnBasedWindField, JaxGridBasedWindField, JaxInterpolatingWindField
from balloon_learning_environment.utils import units
from balloon_learning_environment.utils import constants


def jax_balloon_cost(balloon: JaxBalloon):
    # COST FUNCTION SIMILAR TO THAT PRESENT IN 	arXiv:2403.10784, but provided similar 
    # performance. We cannot take advantage of the dropoff term because we have a continuous 
    # cost.
    # d = jnp.sqrt((balloon.state.x/1000)**2 + (balloon.state.y/1000)**2)
    # return jnp.exp((d-100)/20)
    
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

@partial(jax.jit, static_argnums=(5, 6, 7))
def jax_plan_cost(plan, balloon, wind_field, atmosphere, terminal_cost_fn, time_delta, stride, dynamics_params: JaxBalloonDynamicsParams):
    return jax_plan_cost_no_jit(plan, balloon, wind_field, atmosphere, terminal_cost_fn, time_delta, stride, dynamics_params)

def jax_plan_cost_no_jit(plan, balloon: JaxBalloon, wind_field: JaxWindField, atmosphere: JaxAtmosphere, terminal_cost_fn: TerminalCost, time_delta: 'int, seconds', stride: 'int, seconds', dynamics_params: JaxBalloonDynamicsParams):
    cost = 0.0
    discount_factor = 0.99
    
    def update_step(i, balloon_and_cost: tuple[JaxBalloon, float]):
        balloon, cost = balloon_and_cost

        wind_vector = wind_field.get_forecast(balloon.state.x/1000, balloon.state.y/1000, balloon.state.pressure, balloon.state.time_elapsed)
        
        # action = plan[i]
        action = jax.lax.cond(balloon.state.battery_charge/balloon.state.battery_capacity < 0.025,
                     lambda op: jnp.astype(0.0,jnp.float64),
                     lambda op: op[0],
                     operand=(plan[i],))

        next_balloon = balloon.simulate_step_continuous_no_jit(wind_vector, atmosphere, action, time_delta, stride, dynamics_params)

        cost += (discount_factor**i) * jax_balloon_cost(next_balloon)

        return next_balloon, cost

    final_balloon, cost = jax.lax.fori_loop(0, len(plan), update_step, init_val=(balloon, cost))
    terminal_cost = (discount_factor**len(plan)) * (jax_balloon_cost(final_balloon) + terminal_cost_fn(final_balloon, wind_field))
    return cost + terminal_cost

from jax.tree_util import register_pytree_node_class

from typing import Callable, NamedTuple, Union, Tuple

def sample_action_noise(rng,
    horizon_length: int,
    batch_size: int,
    action_dim: int,
    sample_indices: tuple,
    action_std: float):
    """ 
    Sample noise using a 'knot' interpolation scheme.
    Outputs shape: (batch_size, horizon_length, action_dim)
    """

    key, rng = jax.random.split(rng)
    sample_indices = jnp.asarray(sample_indices, dtype=jnp.int32)
    num_knots = sample_indices.shape[0]
    
    # 1. Generate knot noise with batch_size as the first dimension
    # Shape: (batch_size, num_knots, action_dim)
    knot_noise = action_std * jax.random.normal(key, (batch_size, num_knots, action_dim))
    
    # 2. Handle sorting logic for searchsorted
    sort_perm = jnp.argsort(sample_indices)
    sample_indices_sorted = sample_indices[sort_perm]
    
    # 3. Create the lookup table for every timestep in the horizon
    times = jnp.arange(horizon_length, dtype=jnp.int32)
    idx_per_t = jnp.searchsorted(sample_indices_sorted, times, side="right") - 1
    idx_per_t = jnp.clip(idx_per_t, 0, num_knots - 1)
    
    # 4. Use Advanced Indexing to map knots to time
    # We want to pick indices from the 'num_knots' dimension (axis 1)
    # The result of knot_noise[:, idx_per_t] will be (batch_size, horizon_length, action_dim)
    noise = knot_noise[:, idx_per_t, :]
    
    return rng, noise

class MPPIState(NamedTuple):
    nominal_actions: jnp.ndarray  # (horizon, action_dim)
    rng: jax.Array                # PRNG key

def find_adaptive_temperature(costs, target_ess_pct, num_envs, low=0.001, high=1000.0, steps=15):
    target_ess = target_ess_pct * num_envs
    costs_min = jnp.min(costs)
    shifted_costs = costs - costs_min

    def get_ess(temp):
        # Calculate ESS for a given temperature
        logits = -1.0 / (temp + 1e-6) * shifted_costs
        weights = jax.nn.softmax(logits)
        return 1.0 / jnp.sum(jnp.square(weights))

    def bisection_step(bounds, _):
        low, high = bounds
        mid = (low + high) / 2.0
        ess = get_ess(mid)
        # If ESS is too low, we need a higher temperature (softer weights)
        new_bounds = jax.lax.cond(
            ess < target_ess,
            lambda: (mid, high), # Increase temp
            lambda: (low, mid)  # Decrease temp
        )
        return new_bounds, None

    # Run fixed-step bisection (15 steps is usually enough for high precision)
    final_bounds, _ = jax.lax.scan(bisection_step, (low, high), jnp.arange(steps))
    
    return (final_bounds[0] + final_bounds[1]) / 2.0

@register_pytree_node_class
class MPPI:
    def __init__(
            self, horizon: int, num_envs: int, action_dim: int, 
            action_std: Union[float, jax.Array], target_pct: Union[float, jax.Array], sample_indices: Tuple[int], sample_fn: Callable
    ):
        # Hyperparams
        self.action_std = action_std
        self.target_pct = target_pct

        # Configuration (Static)
        self.horizon = horizon
        self.num_envs = num_envs
        self.action_dim = action_dim
        self.sample_indices = sample_indices
        self.sample_fn = sample_fn

    def init(self, seed: int = 0) -> MPPIState:
        """Initialize the state."""
        return MPPIState(
            nominal_actions=jnp.zeros((self.horizon, self.action_dim)),
            rng=jax.random.PRNGKey(seed)
        )

    def update(self, state: MPPIState, args) -> tuple[jnp.ndarray, MPPIState]:
        """The core Optax-style update function."""
        next_rng, delta_actions = sample_action_noise(state.rng, self.horizon, self.num_envs, self.action_dim, self.sample_indices, self.action_std)
        explore_actions = state.nominal_actions + delta_actions # shape: (num_envs, horizon, action_dim)

        # 2. Score (Rollout)
        costs = self.sample_fn(explore_actions, args) # shape: (num_envs, )
        costs /= self.horizon

        # 2.5 
        target_pct = self.target_pct
        temperature = find_adaptive_temperature(costs, target_pct, self.num_envs)

        # 3. Reweight (The MPPI Update)
        weights = jax.nn.softmax(-1.0 / temperature * (costs - jnp.min(costs)))
        
        # Weighted average across the num_envs dimension
        update_delta = jnp.sum(weights[:, None, None] * delta_actions, axis=0)
        new_nominal_actions = state.nominal_actions + update_delta

        ess = 1.0 / jnp.sum(jnp.square(weights))
        jax.debug.print("ESS: {ess} / {ne} (Temp: {t})",ess=ess,ne=self.num_envs,t=temperature)

        new_state = MPPIState(nominal_actions=new_nominal_actions, rng=next_rng)        
        return new_state

    def shift(self, state: MPPIState, n: int = 1) -> MPPIState:
        """
        Shifts the nominal actions forward by N steps.
        Discards the first N actions and pads the end with the last action.
        """
        # 1. Roll the array N steps to the left
        shifted_actions = jnp.roll(state.nominal_actions, shift=-n, axis=0)
        
        # 2. Identify the last action to use for padding
        last_action = state.nominal_actions[-1]
        
        # 3. Create a mask using explicit functional comparison
        indices = jnp.arange(self.horizon)
        # Using jnp.greater_equal instead of >=
        mask = jnp.greater_equal(indices, self.horizon - n)
        
        # 4. Apply the padding
        # jnp.where(condition, x, y)
        new_nominal_actions = jnp.where(
            mask[:, None], 
            last_action[None, :], 
            shifted_actions
        )
        
        return state._replace(nominal_actions=new_nominal_actions)

    def tree_flatten(self):
        children = (self.action_std, self.target_pct) 
        aux_data = (self.horizon, self.num_envs, self.action_dim, self.sample_indices, self.sample_fn)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        action_std, target_pct = children
        horizon, num_envs, action_dim, sample_indices, sample_fn = aux_data
        return MPPI(horizon, num_envs, action_dim, action_std, target_pct, sample_indices, sample_fn)

_MODEL_FIDELITIES: dict[str, JaxBalloonDynamicsParams] = {
    # Temperature, Volume, Battery, ACS
    'high': JaxBalloonDynamicsParams(True, True, True, True),
    'lower': JaxBalloonDynamicsParams(True, True, False, True), 
    'low': JaxBalloonDynamicsParams(False, True, False, True),
    'lowest': JaxBalloonDynamicsParams(False, False, False, True),
    'none': JaxBalloonDynamicsParams(False, False, False, False),

    'test1': JaxBalloonDynamicsParams(False, True, True, True),
    'test2': JaxBalloonDynamicsParams(True, False, True, True),
    'test3': JaxBalloonDynamicsParams(True, True, False, True),
    'test4': JaxBalloonDynamicsParams(True, True, True, False),
    'test5': JaxBalloonDynamicsParams(False, False, False, False),
    'test6': JaxBalloonDynamicsParams(True, True, True, True),
    'test7' : JaxBalloonDynamicsParams(False, True, False, True),
    'test8' : JaxBalloonDynamicsParams(False, True, True, False),
    'test9' : JaxBalloonDynamicsParams(True, False, False, True),
    'test10': JaxBalloonDynamicsParams(True, False, True, False),
    'test11': JaxBalloonDynamicsParams(True, True, False, False),
    'test12': JaxBalloonDynamicsParams(False, False, True, True),
    'test13': JaxBalloonDynamicsParams(False, False, False, True),
    'test14': JaxBalloonDynamicsParams(False, False, True, False),
    'test15': JaxBalloonDynamicsParams(False, True, False, False),
    'test16': JaxBalloonDynamicsParams(True, False, False, False),
}

class MPC5Agent(agent.Agent):
        
    def __init__(self, num_actions: int, observation_shape, args): # Sequence[int]
        super(MPC5Agent, self).__init__(num_actions, observation_shape)
        self.forecast = None # WindField
        self.ble_atmosphere = None 
        self.atmosphere = None # Atmosphere

        # self._get_dplan = jax.jit(jax.grad(jax_plan_cost, argnums=0), static_argnames=("time_delta", "stride"))

        self.plan_time = 2*24*60*60
        self.time_delta = 3*60
        self.stride = 10

        # self.plan_steps = 960 + 23 
        self.plan_steps: int= args[0] # (self.plan_time // self.time_delta) // 3
        self.replan_steps: int = args[1]
        self.model_fidelity: str = args[2] # 'high' or 'low' fidelity model
        self.num_envs: int = args[3] # number of initializations to try
        self.action_std: int = args[4]
        self.temperature: int = args[5]
        _num_indices = 12
        self.sample_indices: tuple = tuple([i*_num_indices for i in range(self.plan_steps//_num_indices)])
        self.wind_model = args[6] # 'gp_grid', 'grid', 'gp_column', 'column'
        if self.wind_model not in ('gp_grid', 'grid', 'gp_column', 'column'):
            raise ValueError(f'{self.wind_model} is not a valid wind model')

        if self.wind_model == 'gp_grid':
            self.gk_distance = 100.0
            self.gk_time = 30.0
            print(f'gp_grid guassian kernel distance={self.gk_distance} time={self.gk_time}')

        self.dynamics_params: JaxBalloonDynamicsParams = _MODEL_FIDELITIES[self.model_fidelity]

        print(f'MPC5 Agent Args: plan_steps={self.plan_steps} replan_steps={self.replan_steps} model_fidelity={self.model_fidelity} num_initializations={self.num_envs} wind_model={self.wind_model}')

        # self.N = self.plan_steps

        self.state: MPPIState = None # jnp.full((self.plan_steps, ), fill_value=1.0/3.0)
        self.i = 0
        self.key = jax.random.key(seed=0)

        self.balloon = None
        self.time = None
        self.steps_within_radius = 0

        using_Q_function = False

        if using_Q_function:
            self.num_wind_levels = 181
            params = jax_perciatelli.get_distilled_perciatelli(self.num_wind_levels)[0]
            self.terminal_cost_fn = QTerminalCost(self.num_wind_levels, params)
        else:
            self.terminal_cost_fn = NoTerminalCost()
        
        self.discretize_action = False
        self.discretization_cutoff = 0.25
        print("Discretizing action", self.discretize_action, "with cutoff", self.discretization_cutoff)

        self._time_taken = 0.0

        self.mppi = None #MPPI(self.plan_steps, self.num_envs, 1, self.action_std, self.temperature, self.sample_indices, None)

    def _get_current_action(self):
        action = self.state.nominal_actions[0, 0] if self.state is not None else 0.0
        
        if not self.discretize_action:
            # print('using continuous action:', action)
            return action

        # print('discretizing action with cutoff:', self.discretization_cutoff)
        if action > self.discretization_cutoff:
            return AltitudeControlCommand.UP
        elif action < -self.discretization_cutoff:
            return AltitudeControlCommand.DOWN
        else:
            return AltitudeControlCommand.STAY
        
        # TODO: this is basic discretization,

    def begin_episode(self, observation: np.ndarray) -> int:

        # TODO: actually convert observation into an ndarray (it is a JaxBalloonState, see features.py)
        # balloon = JaxBalloon(jax_balloon_state_from_observation(observation))
        if self.wind_model == 'gp_grid' or self.wind_model == 'gp_column' or self.wind_model == 'column':
            perciatelli_features = observation[1]
            windgp: wind_gp.WindGP = observation[2]
            observation: JaxBalloonState = observation[0]
            self.balloon = JaxBalloon(observation)

            num_pressure_levels = 181

            pressure_levels = np.linspace(
                constants.PERCIATELLI_PRESSURE_RANGE_MIN, 
                constants.PERCIATELLI_PRESSURE_RANGE_MAX,
                num_pressure_levels)
        
            pressure_delta = pressure_levels[1] - pressure_levels[0]

            def clamp(idx):
                return min(num_pressure_levels - 1, max(0, idx))

            balloon_level = int(round((self.balloon.state.pressure - constants.PERCIATELLI_PRESSURE_RANGE_MIN) / pressure_delta))
            balloon_level = clamp(balloon_level) # Make sure it's a good index
            num_levels_lower = num_pressure_levels - balloon_level - 1

            assert num_levels_lower >= 0
            
            named_features = features.NamedPerciatelliFeatures(perciatelli_features)
            safe_pressure_levels = []
            for i in range(named_features.num_pressure_levels):
                if named_features.level_is_valid(i):
                    safe_pressure_levels.append(pressure_levels[clamp(i-num_levels_lower)])

            batch = np.zeros((len(safe_pressure_levels), 4))
            batch[:, 0] = self.balloon.state.x
            batch[:, 1] = self.balloon.state.y
            batch[:, 2] = np.array(safe_pressure_levels)
            batch[:, 3] = self.balloon.state.time_elapsed

            if self.wind_model == 'column':
                # delete observations to just directly use underlying wind field (this is a hack
                # to avoid hacking into the "abstractions" in BLE to make this cleaner)
                windgp.error_values.clear()
                windgp.measurement_locations.clear()

            means = windgp.query_batch(batch)[0]
            
            column_wind_field = JaxColumnBasedWindField(jnp.array(safe_pressure_levels), jnp.array(means))

            if self.wind_model == 'gp_grid':
                self.forecast = JaxInterpolatingWindField(
                    column_wind_field, 
                    self.jax_grid_forecast, 
                    self.gk_distance, 
                    self.gk_time, 
                    self.balloon.state.x/1000, 
                    self.balloon.state.y/1000)
            else:
                self.forecast = column_wind_field
        elif self.wind_model == 'grid':
            observation: JaxBalloonState = observation
            self.balloon = JaxBalloon(observation)

        # if self.balloon is not None:
        #     observation.x = self.balloon.state.x
        #     observation.y = self.balloon.state.y

        self.balloon = JaxBalloon(observation)


        def sample_fn(plans, args):
            # 1. Define the function we want to vectorize
            # We use a closure or a partial to fix the arguments that don't change
            def single_plan_cost(plan):
                return jax_plan_cost_no_jit(
                    jnp.squeeze(plan), 
                    args[0], 
                    args[1], 
                    self.atmosphere, 
                    self.terminal_cost_fn,
                    self.time_delta,
                    self.stride,
                    self.dynamics_params
                )

            # 2. Apply vmap
            # in_axes=(0,) means we map over the first dimension of the 'plans' argument
            vectorized_cost_fn = jax.vmap(single_plan_cost, in_axes=(0,))
            
            # 3. Execute on all plans at once
            costs = vectorized_cost_fn(plans)
            
            return costs # This will be shape (plans.shape[0],)
                
        if self.mppi is None:
            self.mppi = MPPI(self.plan_steps, self.num_envs, 1, self.action_std, self.temperature, self.sample_indices, sample_fn)
            self.mppi_update = jax.jit(self.mppi.update)

        # current_plan_cost = jax_plan_cost(self.plan, balloon, self.forecast, self.atmosphere, self.time_delta, self.stride)
        #if current_plan_cost < best_random_cost:
        #    initial_plan = self.plan

        # TODO: is it necessary to pass in forecast when just trying to get to a height?
        
        _start = time.time()
        b4 = time.time()
        
        if self.state is None:
            self.state = self.mppi.init(seed=0)
        self.state = self.mppi_update(self.state, (self.balloon, self.forecast))
        _test_cost = jax_plan_cost(
                    jnp.squeeze(self.state.nominal_actions), 
                    self.balloon, 
                    self.forecast, 
                    self.atmosphere, 
                    self.terminal_cost_fn,
                    self.time_delta,
                    self.stride,
                    self.dynamics_params)

        print(time.time() - b4, 's to get optimized plan with test cost', _test_cost)

        self.i = 0
        self._time_taken += time.time() - _start

        # self._deadreckon()
        # print(time.time() - b4, 's to deadreckon ballooon')

        return self._get_current_action()

    def step(self, reward: float, observation: np.ndarray) -> int:
        REPLANNING = True
        observation: JaxBalloonState = observation
        self.i+=1
        # self._deadreckon()
        # print(observation.battery_charge/observation.battery_capacity)
        if not REPLANNING:
            return self._get_current_action()
        else:
            if self.i>0 and self.i%self.replan_steps==0:
                self.state = self.mppi.shift(self.state, n=self.replan_steps)
                return self.begin_episode(observation)
            else:
                return self._get_current_action()

    def end_episode(self, reward: float, terminal: bool = True) -> None:
        self.i = 0
        self.steps_within_radius = 0
        self.balloon = None
        self.state = None
        self.mppi = None
        # self.plan_steps = 960 + 23

        self._time_taken = 0.0

    def update_forecast(self, forecast: agent.WindField): 
        self.ble_forecast = forecast
        if self.wind_model == 'grid':
            self.forecast = forecast.to_jax_wind_field()
        elif self.wind_model == 'gp_grid':
            self.jax_grid_forecast = forecast.to_jax_wind_field()

    def update_atmosphere(self, atmosphere: agent.standard_atmosphere.Atmosphere): 
        self.ble_atmosphere = atmosphere
        self.atmosphere = atmosphere.to_jax_atmosphere() 
