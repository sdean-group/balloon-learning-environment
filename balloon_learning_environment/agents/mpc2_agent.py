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

def jax_balloon_state_from_observation(observation):
    x = observation[1].m
    y = observation[2].m
    # print(x, y)
    pressure = observation[3]
    t = observation[0].seconds
    # see TODO: below

def jax_balloon_cost(balloon: JaxBalloon):
    return (balloon.state.x)**2 + (balloon.state.y)**2# - balloon.state.acs_power

@jax.jit
def jax_plan_cost(plan, balloon: JaxBalloon, wind_field: JaxWindField, atmosphere: JaxAtmosphere, time_delta: 'int, seconds', stride: 'int, seconds'):
    cost = 0.0
    discount_factor = 0.99#1.00
    
    def update_step(i, balloon_and_cost: tuple[JaxBalloon, float]):
        balloon, cost = balloon_and_cost

        wind_vector = wind_field.get_forecast(balloon.state.x/1000, balloon.state.y/1000, balloon.state.pressure, balloon.state.time_elapsed)
        
        down_balloon = balloon.simulate_step(wind_vector, atmosphere, 0, time_delta, stride)
        stay_balloon = balloon.simulate_step(wind_vector, atmosphere, 1, time_delta, stride)
        up_balloon = balloon.simulate_step(wind_vector, atmosphere, 2, time_delta, stride)
        
        action_distribution = jax.nn.softmax(plan[i])
        cost += (discount_factor**i) * (action_distribution[0] * jax_balloon_cost(down_balloon) + \
            action_distribution[1] * jax_balloon_cost(stay_balloon) + \
            action_distribution[2] * jax_balloon_cost(up_balloon))
        
        # cost +=
        
        next_balloon_which = jnp.argmax(action_distribution)
        next_balloon = jax.lax.cond(next_balloon_which == 0,
                                    lambda op: op[1],
                                    lambda op: jax.lax.cond(
                                        op[0] == 1,
                                        lambda ops: ops[0],
                                        lambda ops: ops[1],
                                        operand=(op[2], op[3])),
                                    operand=(next_balloon_which, down_balloon, stay_balloon, up_balloon))
        # next_balloon = [ down_balloon, stay_balloon, up_balloon ][jnp.argmax(action_distribution)]
        return next_balloon, cost

    final_balloon, final_cost = jax.lax.fori_loop(0, len(plan), update_step, init_val=(balloon, cost))
    return final_cost

def grad_descent_optimizer(initial_plan, dcost_dplan, balloon, forecast, atmosphere, time_delta, stride):
    start_cost = jax_plan_cost(initial_plan, balloon, forecast, atmosphere, time_delta, stride)
    plan = initial_plan
    for gradient_steps in range(100):
        dplan = dcost_dplan(plan, balloon, forecast, atmosphere, time_delta, stride)
        if abs(jnp.linalg.norm(dplan)) < 1e-7:
            break
        plan -= dplan / jnp.linalg.norm(dplan)
    # print(f"After {gradient_steps} gd steps", plan, jax_plan_cost(plan, balloon, forecast, atmosphere, time_delta, stride))
    # input()
    print("GD", gradient_steps, (jax_plan_cost(initial_plan, balloon, forecast, atmosphere, time_delta, stride) - start_cost))
    return plan

def jax_scipy_optimizer(initial_plan, balloon, forecast, atmosphere, time_delta, stride):
    def cost_with_1d_array(plan, balloon, forecast, atmosphere, time_delta, stride):
        return jax_plan_cost(plan.reshape(-1, 3), balloon, forecast, atmosphere, time_delta, stride)

    opt_res = jax.scipy.optimize.minimize(
        fun=cost_with_1d_array, 
        x0=initial_plan.flatten(), 
        args=(balloon, forecast, atmosphere, time_delta, stride), 
        method="BFGS")
    return opt_res.x.reshape(-1, 3)

def scipy_optimizer(initial_plan, dcost_dplan, balloon, forecast, atmosphere, time_delta, stride):
    def cost_with_1d_array(plan, balloon, forecast, atmosphere, time_delta, stride):
        return jax_plan_cost(plan.reshape(-1, 3), balloon, forecast, atmosphere, time_delta, stride)
    
    def grad_with_1d_array(plan, balloon, forecast, atmosphere, time_delta, stride):
        return dcost_dplan(plan.reshape(-1, 3),balloon, forecast, atmosphere, time_delta, stride).flatten()

    # hessian = jax.hessian(grad_with_1d_array)
    opt_res = scipy.optimize.minimize(
        fun=cost_with_1d_array, 
        x0=initial_plan.flatten(), 
        args=(balloon, forecast, atmosphere, time_delta, stride), 
        jac=grad_with_1d_array,
        # hess=hessian,
        method="CG")
    return opt_res.x.reshape(-1, 3)

def get_initials_plans(key, balloon: JaxBalloon,  atmosphere: JaxAtmosphere, num_plans, plan_steps):
    # return jnp.array([ jnp.full((plan_steps, 3), fill_value=1.0/3.0) ])

    # initial_plans = jax.random.uniform(self.key, (50, self.plan_steps, 3))
    # _, self.key = jax.random.split(self.key)

    # goal_altitudes = jnp.linspace(5, 20, 10, num_plans)
    # for goal_altitude in goal_altitudes:
    #     for i in range(plan_steps):
    #         current_altitude = atmosphere.at_pressure(balloon.state.pressure).height.km
    #         action = 1 # Stay
    #         if current_altitude < goal_altitude:
    #             action = 0 # Up
    #         else:
    #             action = 2 
    #         balloon_i = balloon.simulate_step()

    plans = []
    for i in range(num_plans):
        plan_i = jnp.full((plan_steps, 3), fill_value=1.0/3.0)
        # plan_i = jax.random.uniform(key, (plan_steps, 3))
        # key, _ = jax.random.split(key)

        up_down = jax.random.choice(key, jnp.array([ 0, 2 ])) # down = 0 ; up = 2
        length = jax.random.randint(key, (1, ), 0, plan_steps)[0]

        for j in range(plan_steps): 
            if j < length:
                plan_i[j][up_down] += 1.0/3.0
            else:
                plan_i[j][1] += 1.0/3.0


        plans.append(plan_i)

    return plans

class MPC2Agent(agent.Agent):
    
    def __init__(self, num_actions: int, observation_shape): # Sequence[int]
        super(MPC2Agent, self).__init__(num_actions, observation_shape)
        self.forecast = None # WindField
        self.atmosphere = None # Atmosphere

        # self.dplan = jax.jit(jax_plan_cost, static_argnums=(-1,-2), )
        self.get_dplan = jax.jit(jax.grad(jax_plan_cost, argnums=0), static_argnums=(-1,-2))

        self.plan_time = 2*24*60*60
        self.time_delta = 3*60
        self.stride = 60

        self.plan_steps = (self.plan_time // self.time_delta)

        self.plan = jnp.full((self.plan_steps, 3), fill_value=1.0/3.0)
        self.i = 0

        self.key = jax.random.key(seed=0)

    def begin_episode(self, observation: np.ndarray) -> int:
        # TODO: actually convert observation into an ndarray (it is a JaxBalloonState, see features.py)
        # balloon = JaxBalloon(jax_balloon_state_from_observation(observation))
        balloon = JaxBalloon(observation)
        # initial_plan = np.full((self.plan_steps, 3), fill_value=0.5) # everything is equally likely

        initial_plans = jax.random.uniform(self.key, (50, self.plan_steps, 3))
        _, self.key = jax.random.split(self.key)

        # initial_plans = get_initials_plans(self.key, balloon, self.atmosphere, -1, self.plan_steps)

        # print("doing initialization")
        batched_cost = []
        for i in range(len(initial_plans)):
            batched_cost.append(jax_plan_cost(jnp.array(initial_plans[i]), balloon, self.forecast, self.atmosphere, self.time_delta, self.stride))
        # print("finished initialization")

        # batched_cost_fn = jax.jit(jax.vmap(jax_plan_cost, in_axes=(0, None, None, None, None, None)))
        # batched_cost = batched_cost_fn(initial_plans, balloon, self.forecast, self.atmosphere, self.time_delta, self.stride)

        current_plan_cost = jax_plan_cost(self.plan, balloon, self.forecast, self.atmosphere, self.time_delta, self.stride)
        which_min_cost = np.argmin(batched_cost)
        if current_plan_cost < batched_cost[which_min_cost]:
            initial_plan = self.plan
        else:
            initial_plan = initial_plans[np.argmin(batched_cost)]
        

        # no optimization
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
        #     self.get_dplan,
        #     balloon, 
        #     self.forecast, 
        #     self.atmosphere,
        #     self.time_delta, 
        #     self.stride)

        self.i = 0
        # self.i = 1
        # print("argmaxing", self.plan[self.i], "is", np.argmax(self.plan[self.i]))
        action = np.argmax(self.plan[self.i])
        # print(f'Action at iter {self.i}: {action}')
        return action

    def step(self, reward: float, observation: np.ndarray) -> int:
        REPLANNING = True
        observation: JaxBalloonState = observation
        # print(observation.battery_charge/observation.battery_capacity)
        if not REPLANNING:
            self.i += 1
            # print("argmaxing", self.plan[self.i], "is", np.argmax(self.plan[self.i]))
            action = np.argmax(self.plan[self.i%len(self.plan)])
            # print(f'Action at iter {self.i}: {action}')
            return action
        else:
            N = 23
            if self.i>0 and self.i%N==0:
                self.plan = jnp.vstack((self.plan[N:], jax.random.uniform(self.key, (N, 3))))
                return self.begin_episode(observation)
            else:
                self.i += 1
                action = np.argmax(self.plan[self.i])
                return action

 
    def end_episode(self, reward: float, terminal: bool = True) -> None:
        self.i

    def update_forecast(self, forecast: agent.WindField): 
        self.forecast = forecast.to_jax_wind_field()

    def update_atmosphere(self, atmosphere: agent.standard_atmosphere.Atmosphere): 
        self.atmosphere = atmosphere.to_jax_atmopshere() 