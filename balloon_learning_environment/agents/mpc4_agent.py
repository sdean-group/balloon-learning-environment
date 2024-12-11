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
    # return 0.0001*(26436.27 - balloon.state.pressure)**2
    # return (balloon.state.acs_power-0.5)**2
    return (balloon.state.x/1000)**2 + (balloon.state.y/1000)**2

def convert_plan_to_action(acs_control, balloon: JaxBalloon, atmosphere: JaxAtmosphere):
    action = jax.lax.cond(
        acs_control > 0,
        lambda _: 2,
        lambda _: 0,
        operand=None)
    
    return action

# @jax.jit
# def jax_plan_cost(plan, balloon: JaxBalloon, wind_field: JaxWindField, atmosphere: JaxAtmosphere, time_delta: 'int, seconds', stride: 'int, seconds'):
#     cost = 0.0
#     discount_factor = 0.99#1.00
    
#     def update_step(i, balloon_and_cost: tuple[JaxBalloon, float]):
#         balloon, cost = balloon_and_cost

#         wind_vector = wind_field.get_forecast(balloon.state.x/1000, balloon.state.y/1000, balloon.state.pressure, balloon.state.time_elapsed)
        
#         next_balloon = balloon.simulate_step_continuous(wind_vector, atmosphere, plan[i], time_delta, stride)
        
#         cost += (discount_factor**i) * jax_balloon_cost(next_balloon)

#         return next_balloon, cost

#     final_balloon, final_cost = jax.lax.fori_loop(0, len(plan), update_step, init_val=(balloon, cost))
#     return final_cost

@jax.jit
def jax_plan_cost(plan, balloon: JaxBalloon, wind_field: JaxWindField, atmosphere: JaxAtmosphere, time_delta: int, stride: int):
    cost = 0.0
    discount_factor = 0.99

    def scan_step(balloon_and_cost: tuple[JaxBalloon, float], i: int):
        balloon, cost = balloon_and_cost

        # Get the wind vector forecast based on the current balloon state
        wind_vector = wind_field.get_forecast(
            balloon.state.x / 1000, 
            balloon.state.y / 1000, 
            balloon.state.pressure, 
            balloon.state.time_elapsed
        )

        # jax.debug.print("")
        
        # Simulate the next balloon state based on the plan at index i
        next_balloon = balloon.simulate_step_continuous(
            wind_vector, atmosphere, plan[i], time_delta, stride
        )
        
        # Update the cost with the discounted balloon cost
        cost += (discount_factor ** i) * jax_balloon_cost(next_balloon)

        return (next_balloon, cost), None  # No outputs required for scan

    # Use jax.lax.scan to iterate over the plan indices
    (final_balloon, final_cost), _ = jax.lax.scan(
        scan_step, 
        init=(balloon, cost),  # Initial values for balloon and cost
        xs=jax.numpy.arange(len(plan))  # Indices of the plan
    )
    
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

np.random.seed(seed=42)
def make_plan(num_plans, num_steps, balloon:JaxBalloon, wind:JaxWindField, atmosphere:JaxAtmosphere, time_delta, stride):
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
        if  np.isnan(dplan).any() or abs(jnp.linalg.norm(dplan)) < 1e-7:
            print('Exiting early, |∂plan| =',abs(jnp.linalg.norm(dplan)))
            break
        print(gradient_steps, abs(jnp.linalg.norm(dplan)))
        plan -= dplan / jnp.linalg.norm(dplan)

    after_cost = jax_plan_cost(plan, balloon, forecast, atmosphere, time_delta, stride)
    print("GD", gradient_steps, f"∆cost = {after_cost} - {start_cost} = {after_cost - start_cost}")
    return plan

# def numerical_grad_descent_optimizer():
#     pass

class MPC4Agent(agent.Agent):
        
    def __init__(self, num_actions: int, observation_shape): # Sequence[int]
        super(MPC4Agent, self).__init__(num_actions, observation_shape)
        self.forecast = None # WindField
        self.atmosphere = None # Atmosphere

        self.get_dplan = jax.jit(jax.grad(jax_plan_cost, argnums=0), static_argnums=(-1,-2))

        self.plan_time = 2*24*60*60
        self.time_delta = 3*60
        self.stride = 60

        self.plan_steps = (self.plan_time // self.time_delta) #// 3

        self.plan = jnp.full((self.plan_steps, ), fill_value=1.0/3.0)
        self.i = 0

        self.key = jax.random.key(seed=0)

    def begin_episode(self, observation: np.ndarray) -> int:
        # TODO: actually convert observation into an ndarray (it is a JaxBalloonState, see features.py)
        # balloon = JaxBalloon(jax_balloon_state_from_observation(observation))

        balloon = JaxBalloon(observation)

        # best_random_plan, best_random_cost = make_plan(50, self.plan_steps, balloon, self.forecast, self.atmosphere, self.time_delta, self.stride)
        # initial_plan = best_random_plan
        
        # current_plan_cost = jax_plan_cost(self.plan, balloon, self.forecast, self.atmosphere, self.time_delta, self.stride)
        #if current_plan_cost < best_random_cost:
        #    initial_plan = self.plan

        initial_plans = np.random.uniform(-1.0, 1.0, (50, self.plan_steps))
        # _, self.key = jax.random.split(self.key)

        batched_cost = []
        for i in range(len(initial_plans)):
            batched_cost.append(jax_plan_cost(jnp.array(initial_plans[i]), balloon, self.forecast, self.atmosphere, self.time_delta, self.stride))

        initial_plan = initial_plans[np.argmin(batched_cost)]
        
        # self.plan = initial_plan
        
        # def get_gradient():
        #     gradient = np.zeros((self.plan_steps, ))
        #     for i in range(self.plan_steps):
        #         d = np.zeros((self.plan_steps, ))
        #         ε = 0.01
        #         d[i] = ε
        #         before = jax_plan_cost(initial_plan, balloon, self.forecast, self.atmosphere, self.time_delta, self.stride)
        #         after = jax_plan_cost(initial_plan + d, balloon, self.forecast, self.atmosphere, self.time_delta, self.stride)
        #         gradient[i] = (after - before) / ε
        #     return gradient
        
        # plan = initial_plan
        # for gd in range(10):
        #     dplan = get_gradient()
        #     if abs(jnp.linalg.norm(dplan)) < 1e-7:
        #         print('Exiting early, |∂plan| =',abs(jnp.linalg.norm(dplan)))
        #         break
        #     plan -= 0.5* dplan / jnp.linalg.norm(dplan)
        # print("∆ cost:", jax_plan_cost(plan, balloon, self.forecast, self.atmosphere, self.time_delta, self.stride) - np.min(batched_cost))
        # print('Took', gd, 'steps')
        # self.plan = plan

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
        self.atmosphere = atmosphere.to_jax_atmopshere() 