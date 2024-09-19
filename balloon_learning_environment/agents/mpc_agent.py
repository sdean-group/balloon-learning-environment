from balloon_learning_environment.agents import agent
from balloon_learning_environment.models import models
import numpy as np
from typing import Optional, Sequence, Union

# Idea: use observations to improve forecast (like perciatelli feature uses WindGP)

class MPCAgent(agent.Agent):
    """An agent that takes uniform random actions."""

    def __init__(self, num_actions: int, observation_shape: Sequence[int]):
        super(MPCAgent, self).__init__(num_actions, observation_shape)
        self.forecast = None

        self.plan = [ np.randint(0, 3) for _ in range(50) ]
        self.i = 0

    def begin_episode(self, observation: np.ndarray) -> int:
        i += 1
        return self.plan[0]

    def step(self, reward: float, observation: np.ndarray) -> int:
        action = self.plan[self.i%len(self.plan)]
        self.i+=1
        return action

    def end_episode(self, reward: float, terminal: bool = True) -> None:
        self.i = 0 

    def update_forecast(self, forecast: agent.WindField):
        self.forecast = forecast