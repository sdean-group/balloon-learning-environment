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

def jax_balloon_cost(balloon: JaxBalloon):
    # return (balloon.state.acs_power-0.5)**2
    # return  1e-4 * (26436.27 - balloon.state.pressure)**2
    return (balloon.state.x/1000)**2 + (balloon.state.y/1000)**2

def convert_plan_to_action(target_height, balloon: JaxBalloon, atmosphere: JaxAtmosphere):
    height = atmosphere.at_pressure(balloon.state.pressure).height.km
    action = jax.lax.cond(
        jnp.abs(height - target_height) < 0.05,
        lambda _: 1,
        lambda op: jax.lax.cond(
            op[0] < op[1],
            lambda _: 2,
            lambda _: 0,
            operand=None),
        operand=(height, target_height))
    
    return action

@jax.jit
def jax_plan_cost(plan, balloon: JaxBalloon, wind_field: JaxWindField, atmosphere: JaxAtmosphere, time_delta: 'int, seconds', stride: 'int, seconds'):
    cost = 0.0
    discount_factor = 0.99#1.00
    
    def update_step(i, balloon_and_cost: tuple[JaxBalloon, float]):
        balloon, cost = balloon_and_cost

        wind_vector = wind_field.get_forecast(balloon.state.x/1000, balloon.state.y/1000, balloon.state.pressure, balloon.state.time_elapsed)
        
        action = convert_plan_to_action(plan[i], balloon, atmosphere)
        next_balloon = balloon.simulate_step(wind_vector, atmosphere, action, time_delta, stride)
        
        cost += (discount_factor**i) * jax_balloon_cost(next_balloon)

        return next_balloon, cost

    final_balloon, final_cost = jax.lax.fori_loop(0, len(plan), update_step, init_val=(balloon, cost))
    return final_cost

def scipy_optimizer(initial_plan, cost, dcost_dplan, balloon, forecast, atmosphere, time_delta, stride):
    # hessian = jax.hessian(grad_with_1d_array)
    opt_res = scipy.optimize.minimize(
        fun=cost,
        x0=initial_plan, 
        args=(balloon, forecast, atmosphere, time_delta, stride), 
        jac=dcost_dplan,
        # hess=hessian,
        method="CG")
    print("Iterations:", opt_res.nit)
    return opt_res.x

def make_plan(num_plans, num_steps, balloon:JaxBalloon, wind:JaxWindField, atmosphere:JaxAtmosphere, time_delta, stride):
    np.random.seed(seed=42)
    best_plan = -1
    best_cost = +np.inf
    for i in range(num_plans):
        # plan = 13 + 9*np.random.rand(1)
        # plan = np.full((num_steps, 1), plan)

        plan = 22*np.random.random(1) + np.sin(2*np.pi*np.random.rand(1)*np.arange(num_steps)/10)
        # plan = np.reshape(plan, (num_steps, 1))

        cost = jax_plan_cost(plan, balloon, wind, atmosphere, time_delta, stride)
        # print(cost)
        if cost < best_cost:
            best_plan = plan
            best_cost = cost

    return jnp.array(best_plan), best_cost

def grad_descent_optimizer(initial_plan, dcost_dplan, balloon, forecast, atmosphere, time_delta, stride):
    start_cost = jax_plan_cost(initial_plan, balloon, forecast, atmosphere, time_delta, stride)
    plan = initial_plan
    for gradient_steps in range(100):
        dplan = dcost_dplan(plan, balloon, forecast, atmosphere, time_delta, stride)
        if abs(jnp.linalg.norm(dplan)) < 1e-7:
            print('Exiting early, |∂plan| =',abs(jnp.linalg.norm(dplan)))
            break
        plan -= dplan / jnp.linalg.norm(dplan)

    print("GD", gradient_steps, f"{jax_plan_cost(plan, balloon, forecast, atmosphere, time_delta, stride)} - {start_cost}")
    return plan

class MPC3Agent(agent.Agent):
        
    def __init__(self, num_actions: int, observation_shape): # Sequence[int]
        super(MPC3Agent, self).__init__(num_actions, observation_shape)
        self.forecast = None # WindField
        self.atmosphere = None # Atmosphere

        self.get_dplan = jax.jit(jax.grad(jax_plan_cost, argnums=0), static_argnums=(-1,-2))

        self.plan_time = 2*24*60*60
        self.time_delta = 3*60
        self.stride = 60

        self.plan_steps = (self.plan_time // self.time_delta)

        self.plan = jnp.full((self.plan_steps, ), fill_value=1.0/3.0)
        self.i = 0

        self.key = jax.random.key(seed=0)

    def begin_episode(self, observation: np.ndarray) -> int:
        # TODO: actually convert observation into an ndarray (it is a JaxBalloonState, see features.py)
        # balloon = JaxBalloon(jax_balloon_state_from_observation(observation))

        balloon = JaxBalloon(observation)

        best_random_plan, best_random_cost = make_plan(50, self.plan_steps, balloon, self.forecast, self.atmosphere, self.time_delta, self.stride)
        initial_plan = best_random_plan
        
        # current_plan_cost = jax_plan_cost(self.plan, balloon, self.forecast, self.atmosphere, self.time_delta, self.stride)
        #if current_plan_cost < best_random_cost:
        #    initial_plan = self.plan


        # self.plan = initial_plan

        self.plan = grad_descent_optimizer(
            initial_plan, 
            self.get_dplan, 
            balloon, 
            self.forecast, 
            self.atmosphere,
            self.time_delta, 
            self.stride)
        
        # self.plan = scipy_optimizer(
        #     initial_plan, 
        #     jax_plan_cost,
        #     self.get_dplan,
        #     balloon, 
        #     self.forecast, 
        #     self.atmosphere,
        #     self.time_delta, 
        #     self.stride)

        self.i = 0
        return convert_plan_to_action(self.plan[self.i], balloon, self.atmosphere)

    def step(self, reward: float, observation: np.ndarray) -> int:
        REPLANNING = True
        observation: JaxBalloonState = observation
        balloon = JaxBalloon(observation)
        # print(observation.battery_charge/observation.battery_capacity)
        if not REPLANNING:
            self.i += 1
            action = convert_plan_to_action(self.plan[self.i], balloon, self.atmosphere)
            return action
        else:
            N = 23
            if self.i>0 and self.i%N==0:
                # self.plan = jnp.vstack((self.plan[N:], jax.random.uniform(self.key, (N, ))))
                return self.begin_episode(observation)
            else:
                self.i += 1
                action = convert_plan_to_action(self.plan[self.i], balloon, self.atmosphere)
                return action

 
    def end_episode(self, reward: float, terminal: bool = True) -> None:
        self.i

    def update_forecast(self, forecast: agent.WindField): 
        self.forecast = forecast.to_jax_wind_field()

    def update_atmosphere(self, atmosphere: agent.standard_atmosphere.Atmosphere): 
        self.atmosphere = atmosphere.to_jax_atmosphere() 