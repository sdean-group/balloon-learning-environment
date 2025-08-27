import copy
import random
import gym
import numpy as np
from typing import Union, Tuple
import jax
import jax.numpy as jnp
import optax
import equinox as eqx
import datetime as dt

# BLE stuff
from balloon_learning_environment.env.balloon_env import BalloonEnv
from balloon_learning_environment.utils import units
from balloon_learning_environment.env.balloon_arena import BalloonArena
from balloon_learning_environment.env import generative_wind_field
from balloon_learning_environment.env.features import FeatureConstructor
from balloon_learning_environment.env import wind_field, simulator_data
from balloon_learning_environment.env.balloon import standard_atmosphere
from balloon_learning_environment.env.balloon.balloon import BalloonState, Balloon

# Jax BLE stuff
from balloon_learning_environment.env.balloon.jax_balloon import JaxBalloon, JaxBalloonState
from balloon_learning_environment.env.wind_field import JaxWindField
from balloon_learning_environment.env.balloon.standard_atmosphere import JaxAtmosphere
from balloon_learning_environment.agents.mpc4_agent import get_initial_plan, grad_descent_optimizer, TerminalCost, get_dplan, sigmoid, inverse_sigmoid

### Tunable Parameters ###

"""
🎯 Concrete starting point for your case
Param	Value
Z	23 (update every replan) or 46 (every other replan)
G	10 (frequent updates) to 25 (less frequent updates)
H	24 (aim to reduce counterfactual tail issues)
N	12 (focus value function training on near-future that’s better predicted)

🚀 Pro tip: monitor ensemble disagreement + reward variance
If you see high disagreement growing at H’s tail → shorten H or downweight terminal cost

If value targets have high variance → shorten N
"""

# MPC settings

time_delta = 3*60
stride = 10

mpc_plan_horizon = 64
mpc_replan_frequency = 64 # disable replanning [while testing]

mpc_initialization_num_plans = 100

# POLO Settings

polo_ensemble_size = 8
polo_value_update_frequency = 16
# Training parameters

training_seeds = list(range(10_000, 11_000))
testing_seeds = list(range(11_000, 11_100)) 
testing_seeds = testing_seeds[:3] # temporary only up to 3

training_num_episodes = 1000
training_max_episode_length = 64 # make things go quicker [while testing]

training_num_gradient_steps = 25 # because we are replanning less frequently
training_batch_size = 128

### Helper Functions ###

class StateAsFeature(FeatureConstructor):
    """ this is a BLE class used to get the balloon environment to return balloon state state (as opposed to vector states) """

    def __init__(self,
                forecast: wind_field.WindField,
                atmosphere: standard_atmosphere.Atmosphere):
        self.observation = None

    def observe(self, observation: simulator_data.SimulatorObservation):
        self.observation = observation

    def get_features(self) -> BalloonState: # This function usually returns a numpy array, but we return BalloonState directly
        return self.observation.balloon_observation

    def observation_space(self) -> gym.Space:
        return gym.Space() # This function isn't called so this baseclass can be used

def get_balloon_arena(seed: int) -> BalloonArena:
    feature_constructor_factory = lambda forecast, atmosphere: StateAsFeature(forecast, atmosphere) 
    wind_field_factory = generative_wind_field.generative_wind_field_factory
    arena = BalloonArena(feature_constructor_factory, wind_field_factory(), seed=seed)
    return arena

def get_optimized_plan(balloon_state: JaxBalloonState, forecast: JaxWindField, atmosphere: JaxAtmosphere, terminal_cost_fn: TerminalCost, previous_plan=None, return_cost=False):
    """ Helper function to perform a trajectory optimization of a balloon in a wind field """
    balloon = JaxBalloon(balloon_state)
    
    initial_plan = get_initial_plan(
        balloon,
        mpc_initialization_num_plans, 
        forecast, 
        atmosphere, 
        terminal_cost_fn,
        mpc_plan_horizon, 
        time_delta, 
        stride, 
        sigmoid(previous_plan) if previous_plan is not None else previous_plan)

    results: Union[np.ndarray, Tuple[np.ndarray, float]] = grad_descent_optimizer(
        initial_plan,
        get_dplan,
        balloon,
        forecast,
        atmosphere,
        terminal_cost_fn,
        time_delta,
        stride,
        return_cost)

    if return_cost:
        return (inverse_sigmoid(results[0]), results[1])
    else:
        return inverse_sigmoid(results)

def evaluate_plan_in_arena(plan: np.ndarray, balloon_state: BalloonState, arena: BalloonArena):
    """ 
    Helper function to evaluate a plan in a balloon arena WITH wind noise 
    
    balloon_state is the state you want to evaluate the plan from
    arena is used to get the real wind 

    Note: does not modify the arena object
    Note: should return a list of states and then the caller can define the cost function? 
        or pass in a cost function that takes in a list of states
    Returns: final_balloon, total_steps_within_radius, total_steps, distances 
    """    
    balloon = Balloon(copy.deepcopy(balloon_state)) # NOTE: needs a copy of the balloon state because otherwise this method modifies the reference

    steps_within_radius = 0
    steps_completed = len(plan)

    distances = []

    for action in plan:
        wind_vector = arena._wind_field.get_ground_truth(
            balloon.state.x,
            balloon.state.y,
            balloon.state.pressure,
            balloon.state.time_elapsed)

        try:
            balloon.simulate_step(
                wind_vector, 
                arena._atmosphere, 
                action, 
                dt.timedelta(seconds=time_delta), 
                dt.timedelta(seconds=stride))
        except:
            print("error occurred with action", action)

        if balloon.state.x.km**2 + balloon.state.y.km**2 < 50**2:
            steps_within_radius += 1

        distances.append(balloon.state.x.km**2 + balloon.state.y.km**2)

    return balloon.state, steps_within_radius, steps_completed, distances

def evaluate_mpc_performance(arena: BalloonArena, jax_forecast: JaxWindField, jax_atmosphere: JaxAtmosphere, terminal_cost_fn: TerminalCost):
    """ Helper function to evaluate MPC's performance in a balloon arena (e.g. how it would actually perform with wind noise) """
    
    # Initialize loop variables
    previous_plan = None
    balloon_state = None

    # Performance
    total_steps_within_radius = 0
    total_steps_completed = 0

    total_distance = 0
    
    while total_steps_completed < training_max_episode_length:
        balloon_state = arena.get_balloon_state()
        plan = get_optimized_plan(
            JaxBalloonState.from_ble_state(balloon_state), 
            jax_forecast, 
            jax_atmosphere, 
            terminal_cost_fn,
            previous_plan)

        balloon_state, steps_within_radius, steps_completed, distances = evaluate_plan_in_arena(plan[:mpc_replan_frequency], balloon_state, arena)
        
        which_steps = range(total_steps_completed, total_steps_completed + steps_completed)
        total_distance += sum(distance * 0.99**step for distance, step in zip(distances, which_steps))
        
        total_steps_within_radius += steps_within_radius
        total_steps_completed += steps_completed
    
    return total_distance, total_steps_within_radius / total_steps_completed

### Network definitions & Terminal Cost ###

## NOTE: NEEED TO MAKE SURE ALL THESE CLASSES ARE JAX-COMPILED CORRECTLY
# (e.g. that only the things we want to be differentiated are differentiated)

class ValueNetwork(eqx.Module):
    """ A value network to be trained in the POLO framework. """
    residual: eqx.nn.MLP
    prior: eqx.nn.MLP
    optimizer: optax.GradientTransformation
    opt_state: optax.OptState

    def __init__(self, key, input_dim=5, hidden_dim=128, lr=1e-3):
        key_res, key_prior = jax.random.split(key)
        self.residual = eqx.nn.MLP(input_dim, 1, hidden_dim, depth=2, key=key_res)
        self.prior = eqx.nn.MLP(input_dim, 1, hidden_dim, depth=2, key=key_prior)

        self.optimizer = optax.adam(lr)
        self.opt_state = self.optimizer.init(eqx.filter(self.residual, eqx.is_array))

    def __call__(self, x):
        # Predict total value = prior + residual
        prior_val = jnp.squeeze(self.prior(x), axis=-1)
        resid_val = jnp.squeeze(self.residual(x), axis=-1)
        return prior_val + resid_val

    def update(self, x, y):
        def loss_fn(residual_params):
            model = eqx.tree_at(lambda m: m.residual, self, residual_params)
            return jnp.mean((model(x) - y) ** 2)

        grads = jax.grad(loss_fn)(eqx.filter(self.residual, eqx.is_array))
        updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
        self.residual = eqx.apply_updates(self.residual, updates)



class ValueNetworkFeature(eqx.Module):
    """ interface for computing feature vectors given a balloon and wind forecast for a value network """

    def __init__(self):
        pass

    @property
    def num_input_dimensions(self):
        raise NotImplementedError('Implement num_input_dimensions')

    def compute(self, balloon: JaxBalloon, wind_forecast: JaxWindField):
        raise NotImplementedError('Implement computing')

class BasicValueNetworkFeature(ValueNetworkFeature):
    """ this is for testing that all the jax features work """
    def __init__(self): pass

    @property
    def num_input_dimensions(self):
        return 5

    def compute(self, balloon_state: JaxBalloonState, wind_forecast: JaxWindField) -> jnp.ndarray:
        wind_vector = wind_forecast.get_forecast(balloon_state.x/1000, balloon_state.y/1000, balloon_state.pressure, balloon_state.time_elapsed)
        return jnp.array([ 
            balloon_state.x/1000, 
            balloon_state.y/1000, 
            balloon_state.pressure, 
            wind_vector[0], 
            wind_vector[1] ])


class SpatialAveragingFeature(ValueNetworkFeature):
    @property
    def num_input_dimensions(self):
        raise NotImplementedError('Implement num_input_dimensions')

    def compute(self, balloon: JaxBalloon, wind_forecast: JaxWindField):
        raise NotImplementedError('Implement computing')

class EnsembleTerminalCost(eqx.Module, TerminalCost):
    """ Uses an ensemble of value networks to calculate terminal cost """
    vn_feature: ValueNetworkFeature
    ensemble: list[ValueNetwork]
    kappa: float = 1.0

    def __call__(self, balloon: JaxBalloon, wind_forecast: JaxWindField):
        # return 0
        features = self.vn_feature.compute(balloon, wind_forecast)
        preds = jnp.stack([net(features) for net in ensemble])
        return jax.scipy.special.logsumexp(self.kappa * preds) / self.kappa


class ValueTerminalCost(eqx.Module, TerminalCost):
    """ Use a single value network as a terminal cost """
    def __init__(self, vn_feature: ValueNetworkFeature, network: ValueNetwork):
        self.vn_feature = vn_feature
        self.network = network

    def __call__(self, balloon: JaxBalloon, wind_forecast: JaxWindField):
        return self.network(self.vn_feature(balloon, wind_forecast))

### Training loop ###
# if __name__ == "__main__":
vn_feature = BasicValueNetworkFeature()
ensemble = [ ValueNetwork(key=jax.random.key(seed=seed), input_dim=vn_feature.num_input_dimensions) for seed in range(polo_ensemble_size) ]
D: list[JaxBalloonState] = []

for episode in range(training_num_episodes):
    print(f"Episode {episode}")
    
    # Evaluate performance of MPC with ensemble terminal cost on testing seeds
    if episode % 10 == 0:
        print(f"Evaluating performance on testing seeds at episode {episode}")
        ensemble_value_fn = EnsembleTerminalCost(vn_feature, ensemble)

        average_twr = 0
        average_cost = 0
        for seed in testing_seeds:
            arena = get_balloon_arena(seed)
            jax_forecast, jax_atmosphere = arena._wind_field.to_jax_wind_field(), arena._atmosphere.to_jax_atmosphere()

            cost, twr = evaluate_mpc_performance(arena, jax_forecast, jax_atmosphere, ensemble_value_fn)
            average_twr += twr
            average_cost += cost
        
        average_twr /= len(testing_seeds)
        average_cost /= len(testing_seeds)

        print(f"{average_twr=}, {average_cost=}")

    # Sample seed randomly from the training seeds
    seed = random.choice(training_seeds)
    print(f"Training episode {episode + 1}/{training_num_episodes} with seed {seed}")

    # Create balloon and wind field from seed
    arena = get_balloon_arena(seed)
    jax_forecast, jax_atmosphere = arena._wind_field.to_jax_wind_field(), arena._atmosphere.to_jax_atmosphere()
    # NOTE: starting position of balloon will be the same for a seed, so the balloon will always start at the same position given a seed
    # NOTE: call arena.reset to get initial observation

    plan = None # Initialize plan variable, potentially re-use across iterations based on replan frequency
    plan_idx = 0

    for t in range(training_max_episode_length):
        jax_balloon_state = JaxBalloonState.from_ble_state(arena.get_balloon_state())
        ensemble_value_fn = EnsembleTerminalCost(vn_feature, ensemble) # NOTE: this is recreated every time so it should just be a light wrapper around ensemble to weighted softmax for terminal cost

        # Generate a new MPC plan if it's the first step or if we need to replan
        if t % mpc_replan_frequency == 0:
            plan = get_optimized_plan(jax_balloon_state, jax_forecast, jax_atmosphere, ensemble_value_fn, plan)
            plan_idx = 0

        s_next = JaxBalloonState.from_ble_state(arena.step(plan[plan_idx])) # NOTE: we sample real transitions here (arena.step calculated the real wind)
        plan_idx += 1
        D.append(s_next)

        if t % polo_value_update_frequency == 0:
            for g in range(training_num_gradient_steps):
                state_batch: list[JaxBalloonState] = random.sample(D, min(len(D), training_batch_size))
                feature_batch = [vn_feature.compute(state, jax_forecast) for state in state_batch]

                for k in range(len(ensemble)):
                    targets = []
                    value_terminal_fn = ValueTerminalCost(vn_feature, ensemble[k])

                    for s_i in state_batch:
                        twr, cost = evaluate_mpc_performance(arena, jax_forecast, jax_atmosphere, value_terminal_fn)
                        targets.append(cost) # NOTE: twr may make sense but this cost kind of has the exact unit we are looking for

                    ensemble[k].update(feature_batch, targets)


                