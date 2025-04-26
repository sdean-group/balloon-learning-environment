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
from balloon_learning_environment.models import models
from balloon_learning_environment.utils import units
import numpy as np
from typing import Optional, Sequence, Union
import jax.numpy as jnp
import datetime as dt
from atmosnav import *
import atmosnav as atm
from scipy.optimize import minimize
from functools import partial

class DeterministicDiscreteAltitudeModel:

    def __init__(self, state, integration_time_step):
        self.dt = integration_time_step
        self.vlim = 1.7
        
        self.state = state # jnp.zeros((5, ))

    #@profile
    def simulate_step(self, control_input, wind_vector):
        state = self.state
        h = jax.lax.cond(control_input == 0,
                    lambda op: op[2].get_next_h(op[0], op[0][2] - 0.5), # down
                    lambda op: jax.lax.cond(op[1] == 1,
                                            lambda ops: ops[0][2],
                                            lambda ops: ops[2].get_next_h(ops[0], ops[0][2] + 0.5), # up
                                            operand=op),
                    operand=(state, control_input, self))

        return DeterministicDiscreteAltitudeModel(
            state=state + self.dt * jnp.array([ wind_vector[0], wind_vector[1], (h - state[2])/self.dt, 0.0, 1 ]),
            integration_time_step=self.dt)
    
    #@profile
    def get_next_h(self, state, waypoint):
        return jax.lax.cond(jnp.abs(waypoint-state[2]) > self.vlim / 3600.0 * self.dt,
                        lambda op: op[0][2] + self.vlim / 3600.0 * self.dt * jnp.sign(op[1]-op[0][2]),
                        lambda op: op[1],
                        operand = (state, waypoint))

    def tree_flatten(self):
        children = (self.state, )  # arrays / dynamic values
        aux_data = {'dt':self.dt, 'vlim':self.vlim}  # static values
        return (children, aux_data)
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return DeterministicDiscreteAltitudeModel(children[0], aux_data['dt'])

from jax.tree_util import register_pytree_node_class
register_pytree_node_class(DeterministicDiscreteAltitudeModel)

def jax_balloon_cost(balloon: DeterministicDiscreteAltitudeModel):
    return (balloon.state[0])**2 + (balloon.state[1])**2# - balloon.state.acs_power
    # return (balloon.state.x)**2 + (balloon.state.y)**2

@jax.jit
def jax_plan_cost(plan, balloon: DeterministicDiscreteAltitudeModel, wind_field: JaxWindField, atmosphere: JaxAtmosphere, time_delta: 'int, seconds', stride: 'int, seconds'):
    cost = 0.0
    discount_factor = 0.99#1.00
    
    def update_step(i, balloon_and_cost: tuple[DeterministicDiscreteAltitudeModel, float]):
        balloon, cost = balloon_and_cost

        wind_vector = wind_field.get_forecast(balloon.state[0], balloon.state[1], atmosphere.at_height(balloon.state[2]*1000).pressure, balloon.state[4])
        
        down_balloon = balloon.simulate_step(0, wind_vector)
        stay_balloon = balloon.simulate_step(1, wind_vector)
        up_balloon = balloon.simulate_step(2, wind_vector)
        
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
        # jax.debug.print("{x}", x=cost)
        return next_balloon, cost

    final_balloon, final_cost = jax.lax.fori_loop(0, len(plan), update_step, init_val=(balloon, cost))
    return final_cost

def grad_descent_optimizer(initial_plan, dcost_dplan, balloon, forecast, atmosphere, time_delta, stride):
    start_cost = jax_plan_cost(initial_plan, balloon, forecast, atmosphere, time_delta, stride)
    plan = initial_plan
    for gradient_steps in range(100):
        dplan = dcost_dplan(plan, balloon, forecast, atmosphere, time_delta, stride)
        if jnp.isnan(dplan).any() or abs(jnp.linalg.norm(dplan)) < 1e-7:
            print('Exiting early, |∂plan| =',abs(jnp.linalg.norm(dplan)))
            break
        plan -= dplan / jnp.linalg.norm(dplan)
    # print(f"After {gradient_steps} gd steps", plan, jax_plan_cost(plan, balloon, forecast, atmosphere, time_delta, stride))
    # input()
    print("GD", gradient_steps, f"{jax_plan_cost(initial_plan, balloon, forecast, atmosphere, time_delta, stride)} - {start_cost}")
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
    print("optimization iterations:", opt_res.nit)
    return opt_res.x.reshape(-1, 3)

class MPCDiscreteAgent(agent.Agent):
    
    def __init__(self, num_actions: int, observation_shape): # Sequence[int]
        super(MPCDiscreteAgent, self).__init__(num_actions, observation_shape)
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
        x = observation[1].km
        y = observation[2].km
        # print(x, y)
        pressure = observation[3]
        t = observation[0].seconds

        # # t, x, y, pressure = observation
        balloon = DeterministicDiscreteAltitudeModel(jnp.array([x, y, self.atmosphere.at_pressure(pressure).height.km, 0, t]), self.stride)
        

        print("BEFORE:", balloon.state[2], "AFTER:", balloon.simulate_step(2, jnp.array([0.0,0.0])).state[2])
        

        initial_plans = jax.random.uniform(self.key, (50, self.plan_steps, 3))
        _, self.key = jax.random.split(self.key)

        batched_cost = []
        for i in range(len(initial_plans)):
            batched_cost.append(jax_plan_cost(jnp.array(initial_plans[i]), balloon, self.forecast, self.atmosphere, self.time_delta, self.stride))

        which_min_cost = jnp.argmin(jnp.array(batched_cost))

        current_plan_cost = jax_plan_cost(self.plan, balloon, self.forecast, self.atmosphere, self.time_delta, self.stride)
        
        if current_plan_cost < batched_cost[which_min_cost]:
            print('using existing plan')
            initial_plan = self.plan
        else:
            initial_plan = initial_plans[np.argmin(batched_cost)]
        
        # print("Initial cost: ", jax_plan_cost(initial_plan, balloon, self.forecast, self.atmosphere, self.time_delta, self.stride))

        # initial_plan = np.full((self.plan_steps, 3), fill_value=0.5) # everything is equally likely
        # initial_plan[:, 2]=1.0

        # no optimization
        self.plan = initial_plan

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
        REPLANNING = False
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
        self.atmosphere = atmosphere.to_jax_atmosphere() 