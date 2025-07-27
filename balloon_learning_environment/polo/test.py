import jax
import jax.numpy as jnp
import equinox as eqx

from .value_network import ValueNetwork, ValueNetworkFeature, ValueNetworkTerminalCost, ValueNetworkTrainer
from .value_network_feature import BasicValueNetworkFeature

from balloon_learning_environment.env.balloon_env import BalloonEnv
from balloon_learning_environment.utils import units
from balloon_learning_environment.env.balloon_arena import BalloonArena
from balloon_learning_environment.env import generative_wind_field
from balloon_learning_environment.env.features import FeatureConstructor
from balloon_learning_environment.env import wind_field, simulator_data
from balloon_learning_environment.env.balloon import standard_atmosphere
from balloon_learning_environment.env.balloon.balloon import BalloonState, Balloon
# import stuff from mpc4_agent
from balloon_learning_environment.agents.mpc4_agent import sigmoid, inverse_sigmoid

from balloon_learning_environment.env.balloon.jax_balloon import JaxBalloon, JaxBalloonState
from balloon_learning_environment.env.wind_field import JaxWindField
from balloon_learning_environment.env.balloon.standard_atmosphere import JaxAtmosphere
import datetime as dt
from .utils import get_balloon_arena

class TestSuite:
    def __init__(self):
        self._to_run = []

    def test_case(self, fn):
        self._to_run.append(fn)
        return fn
    
    def run_tests(self):
        for fn in self._to_run:
            print(f"Running test: {fn.__name__}")
            fn()

test_suite = TestSuite()

@test_suite.test_case
def test_train_network():
    x = jnp.array([[1.0, 2.0, 3.0, 4.0, 5.0]]).squeeze()
    y = jnp.array([[10.0]])

    trainer = ValueNetworkTrainer(
        initial_model=ValueNetwork.create(key=jax.random.key(0)), 
        learning_rate=1e-3, 
        jit_model_loss_and_grad=False)
    
    loss_value = trainer.model.loss_and_grad(x, y)[0]
    print('Initial loss value:', loss_value)

    assert loss_value.shape == (), "Loss value should be scalar"

    trainer.update(x, y)
    new_loss_value = trainer.model.loss_and_grad(x, y)[0]
    print('Updated loss value:', new_loss_value)
    
    assert new_loss_value < loss_value, "Loss did not decrease after gradient update"

@test_suite.test_case
def test_jit_train_network():
    x = jnp.array([[1.0, 2.0, 3.0, 4.0, 5.0]]).squeeze()
    y = jnp.array([[10.0]])

    trainer = ValueNetworkTrainer(initial_model=ValueNetwork.create(key=jax.random.key(0)), learning_rate=1e-3)

    loss_value = trainer.model.loss_and_grad(x, y)[0]
    print('Initial loss value:', loss_value)

    assert loss_value.shape == (), "Loss value should be scalar"

    for i in range(10):
        # NOTE: Shouldn't recompile during this loop
        trainer.update(x, y)
        new_loss_value = trainer.model.loss_and_grad(x, y)[0]
        print('Updated loss value:', new_loss_value)

    assert new_loss_value < loss_value, "Loss did not decrease after gradient update"

@test_suite.test_case
def test_in_cost_function():
    model = ValueNetwork.create(key=jax.random.key(0))
    x = jnp.array([[1.0, 2.0, 3.0, 4.0, 5.0]]).squeeze()
    
    def cost_fn(x, model):
        return model(x)

    grad_fn = eqx.filter_jit(eqx.filter_grad(cost_fn))
    cost0 = cost_fn(x, model)
    grad = grad_fn(x, model)
    new_x = x - 0.1 * grad
    cost1 = cost_fn(new_x, model)

    print('Cost before:', cost0, 'Cost after:', cost1)

    # check if cost decreased
    assert cost1 < cost0, "Cost did not decrease after gradient step"
    # check that x changed
    assert not jnp.array_equal(new_x, x), "Input x did not change after gradient step"

@test_suite.test_case
def test_terminal_cost():
    arena = get_balloon_arena(seed=0)
    
    simple_feature = BasicValueNetworkFeature()

    model = ValueNetwork.create(key=jax.random.key(0), input_dim=simple_feature.num_input_dimensions)
    terminal_cost = ValueNetworkTerminalCost(simple_feature, model)

    jax_balloon = JaxBalloon(JaxBalloonState.from_ble_state(arena.get_balloon_state()))
    jax_forecast = arena._wind_field.to_jax_wind_field()
    jax_atmosphere = arena._atmosphere.to_jax_atmosphere()

    def cost_fn(plan, balloon: JaxBalloon, jax_forecast: JaxWindField, jax_atmosphere: JaxAtmosphere, terminal_cost_fn: ValueNetworkTerminalCost):
        def step_fn(carry, a):
            balloon, total_cost = carry
            wind_vector = jax_forecast.get_forecast(
                balloon.state.x / 1000,
                balloon.state.y / 1000,
                balloon.state.pressure,
                balloon.state.time_elapsed)
            balloon = balloon.simulate_step_continuous_no_jit(
                wind_vector,
                jax_atmosphere,
                a,
                180,
                10)
            step_cost = (balloon.state.x/1000)**2 + (balloon.state.y/1000)**2
            total_cost = total_cost + step_cost
            return (balloon, total_cost), None

        (final_balloon, total_cost), _ = jax.lax.scan(step_fn, (balloon, 0.0), inverse_sigmoid(plan))
        total_cost = total_cost + terminal_cost_fn(final_balloon, jax_forecast)
        return total_cost

    grad = eqx.filter_jit(eqx.filter_grad(cost_fn))

    plan = sigmoid(jnp.array([ 0.1, -0.1, 0.05, -0.05, 0.1]))
    for i in range(10):
        # NOTE: this loop was sensitive to learning rate (in terms of not getting any issues, should pay attention for that in the real code)
        plan_delta = grad(plan, jax_balloon, jax_forecast, jax_atmosphere, terminal_cost)
        print('Cost:', cost_fn(sigmoid(plan), jax_balloon, jax_forecast, jax_atmosphere, terminal_cost), 'Plan:', inverse_sigmoid(plan), 'Plan delta:', plan_delta)
        plan -= 0.001 * plan_delta
    print(cost_fn(plan, jax_balloon, jax_forecast, jax_atmosphere, terminal_cost))

    # Assert that the final plan does not have NaN values
    assert not jnp.any(jnp.isnan(plan)), "Plan contains NaN values after optimization"

    # Assert that the final cost is finite
    final_cost = cost_fn(plan, jax_balloon, jax_forecast, jax_atmosphere, terminal_cost)
    assert jnp.isfinite(final_cost), "Final cost is not finite after optimization"

    # Assert that the final plan is not all zeros
    assert not jnp.all(plan == 0), "Final plan is all zeros after optimization"

    # Assert that the final cost is less than the initial cost
    initial_cost = cost_fn(sigmoid(jnp.array([0.1, -0.1, 0.05, -0.05, 0.1])), jax_balloon, jax_forecast, jax_atmosphere, terminal_cost)
    assert final_cost < initial_cost, "Final cost is not less than initial cost after optimization"

if __name__ == "__main__":
    test_suite.run_tests()
    print("All tests completed successfully.")
    