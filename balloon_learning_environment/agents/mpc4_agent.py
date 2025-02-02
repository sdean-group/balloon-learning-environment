import jax.scipy.optimize
import scipy.optimize
from balloon_learning_environment.agents import agent
from balloon_learning_environment.env.balloon.jax_balloon import JaxBalloon, JaxBalloonState
from balloon_learning_environment.env.wind_field import JaxWindField
from balloon_learning_environment.env.balloon.standard_atmosphere import JaxAtmosphere
import numpy as np
import jax
import jax.numpy as jnp
import scipy
from functools import partial

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

    up_balloon = balloon
    while time_to_top < plan_steps and atmosphere.at_pressure(up_balloon.state.pressure).height.km < max_km_to_explore:
        wind_vector = forecast.get_forecast(up_balloon.state.x/1000, up_balloon.state.y/1000, up_balloon.state.pressure, up_balloon.state.time_elapsed)
        up_balloon = up_balloon.simulate_step_continuous(wind_vector, atmosphere, 1.0, time_delta, stride)
        time_to_top += 1

    time_to_bottom = 0
    min_km_to_explore = 12.0 # descending is harder

    down_balloon = balloon
    while time_to_bottom < plan_steps and atmosphere.at_pressure(down_balloon.state.pressure).height.km > min_km_to_explore:
        wind_vector = forecast.get_forecast(down_balloon.state.x/1000, down_balloon.state.y/1000, down_balloon.state.pressure, down_balloon.state.time_elapsed)
        down_balloon = down_balloon.simulate_step_continuous(wind_vector, atmosphere, -1.0, time_delta, stride)
        time_to_bottom += 1

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
    
    return inverse_sigmoid(np.array(plans))


class MPC4Agent(agent.Agent):
        
    def __init__(self, num_actions: int, observation_shape): # Sequence[int]
        super(MPC4Agent, self).__init__(num_actions, observation_shape)
        self.forecast = None # WindField
        self.atmosphere = None # Atmosphere

        # self.get_dplan = jax.jit(jax.grad(jax_plan_cost, argnums=0), static_argnums=(-1,-2))

        self.get_dplan = jax.grad(jax_plan_cost, argnums=0)

        self.plan_time = 2*24*60*60
        self.time_delta = 3*60
        self.stride = 60

        self.plan_steps = (self.plan_time // self.time_delta) # // 3

        self.plan = None # jnp.full((self.plan_steps, ), fill_value=1.0/3.0)
        self.i = 0

        self.key = jax.random.key(seed=0)

    def begin_episode(self, observation: np.ndarray) -> int:
        # TODO: actually convert observation into an ndarray (it is a JaxBalloonState, see features.py)
        # balloon = JaxBalloon(jax_balloon_state_from_observation(observation))

        balloon = JaxBalloon(observation)

        # current_plan_cost = jax_plan_cost(self.plan, balloon, self.forecast, self.atmosphere, self.time_delta, self.stride)
        #if current_plan_cost < best_random_cost:
        #    initial_plan = self.plan

        # TODO: is it necessary to pass in forecast when just trying to get to a height?

        initial_plans =get_initial_plans(balloon, 50, self.forecast, self.atmosphere, self.plan_steps, self.time_delta, self.stride)
        batched_cost = []
        for i in range(len(initial_plans)):
            batched_cost.append(jax_plan_cost(jnp.array(initial_plans[i]), balloon, self.forecast, self.atmosphere, self.time_delta, self.stride))

        # print(np.min(batched_cost))
        initial_plan = initial_plans[np.argmin(batched_cost)]

        self.plan = grad_descent_optimizer(
            initial_plan, 
            self.get_dplan, 
            balloon, 
            self.forecast, 
            self.atmosphere,
            self.time_delta, 
            self.stride)
        self.plan = sigmoid(self.plan)

        self.i = 0
        action = self.plan[self.i]
        self.i+=1
        # print('action', action)
        return action

    def step(self, reward: float, observation: np.ndarray) -> int:
        REPLANNING = True
        observation: JaxBalloonState = observation
        balloon = JaxBalloon(observation)
        # print(observation.battery_charge/observation.battery_capacity)
        if not REPLANNING:
            self.i += 1
            action = self.plan[self.i]
            return action
        else:
            N = 23
            if self.i>0 and self.i%N==0:
                # self.plan = jnp.vstack((self.plan[N:], jax.random.uniform(self.key, (N, ))))
                return self.begin_episode(observation)
            else:
                action = self.plan[self.i]
                self.i += 1
                # print('action', action)
                return action

 
    def end_episode(self, reward: float, terminal: bool = True) -> None:
        self.i

    def update_forecast(self, forecast: agent.WindField): 
        self.forecast = forecast.to_jax_wind_field()

    def update_atmosphere(self, atmosphere: agent.standard_atmosphere.Atmosphere): 
        self.atmosphere = atmosphere.to_jax_atmopshere() 