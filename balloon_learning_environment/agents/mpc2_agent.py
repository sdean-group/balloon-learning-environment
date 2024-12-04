from balloon_learning_environment.agents import agent
from balloon_learning_environment.env.balloon.jax_balloon import JaxBalloon
from balloon_learning_environment.env.wind_field import JaxWindField
from balloon_learning_environment.env.balloon.standard_atmosphere import JaxAtmosphere
import numpy as np
import jax
import jax.numpy as jnp

def jax_balloon_state_from_observation(observation):
    x = observation[1].m
    y = observation[2].m
    # print(x, y)
    pressure = observation[3]
    t = observation[0].seconds
    # see TODO: below

# class Optimizer:
#     def __init__(self):
#         pass


def jax_balloon_cost(balloon):
    return balloon.state.x**2 + balloon.state.y**2

def jax_plan_cost(plan, balloon: JaxBalloon, wind_field: JaxWindField, atmosphere: JaxAtmosphere, time_delta: 'int, seconds', stride: 'int, seconds'):
    cost = 0.0
    discount_factor = 0.95
    
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

        self.plan = None
        self.i = 0

    def begin_episode(self, observation: np.ndarray) -> int:
        # TODO: actually convert observation into an ndarray (it is a JaxBalloonState, see features.py)
        # balloon = JaxBalloon(jax_balloon_state_from_observation(observation))
        balloon = JaxBalloon(observation)
        initial_plan = np.full((self.plan_steps, 3), fill_value=0.5) # everything is equally likely
        
        self.plan = initial_plan
        for _ in range(100):
            dplan = self.get_dplan(self.plan, balloon, self.forecast, self.atmosphere, self.time_delta, self.stride)
            # print("plan")
            # print(self.plan)
            # print("∆ plan")
            # print(dplan)
            self.plan -= dplan / jnp.linalg.norm(dplan)
            # input()
        
        # self.i = 0
        self.i = 1
        action = np.argmax(self.plan[self.i])
        print(f'Action at iter {self.i}: {action}')
        return action

    def step(self, reward: float, observation: np.ndarray) -> int:
        REPLANNING = False
        if not REPLANNING:
            self.i += 1
            action = np.argmax(self.plan[self.i])
            print(f'Action at iter {self.i}: {action}')
            return action
        else:
            return self.begin_episode(observation)
 
    def end_episode(self, reward: float, terminal: bool = True) -> None:
        self.i

    def update_forecast(self, forecast: agent.WindField): 
        self.forecast = forecast.to_jax_wind_field()

    def update_atmosphere(self, atmosphere: agent.standard_atmosphere.Atmosphere): 
        self.atmosphere = atmosphere.to_jax_atmopshere() 