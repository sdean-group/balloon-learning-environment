# coding=utf-8
# Copyright 2022 The Balloon Learning Environment Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Evaluation library for Balloon Learning Environment agents."""

import dataclasses
import datetime as dt
import json
from typing import Any, List, Sequence

from absl import logging
from balloon_learning_environment.agents import agent as base_agent
from balloon_learning_environment.env import balloon_env
from balloon_learning_environment.env import simulator_data
from balloon_learning_environment.env.balloon import balloon
from balloon_learning_environment.eval import suites
from balloon_learning_environment.utils import units
from jax import numpy as jnp
import numpy as np
from tqdm import tqdm

import json


class EvalResultEncoder(json.JSONEncoder):
  """A JSON encoder for encoding EvaluationResult objects.

  e.g. `json.dumps(eval_result_object, cls=EvalResultEncoder)`.
  """

  def default(self, o: Any):
    if isinstance(o, SimpleBalloonState):
      return {
          'x': o.x.kilometers,
          'y': o.y.kilometers,
          'pressure': o.pressure,
          'superpressure': o.superpressure,
          'elapsed_seconds': o.time_elapsed.total_seconds(),
          'power': o.battery_soc,
      }
    elif dataclasses.is_dataclass(o):
      # Note: don't use dataclasses.asdict since it recurses.
      return o.__dict__
    elif isinstance(o, (np.ndarray, jnp.ndarray)) and o.size == 1:
      return o.item()
    else:
      return json.JSONEncoder.default(self, o)


@dataclasses.dataclass
class SimpleBalloonState:
  """A class for keeping track of a balloon state during evaluation."""
  x: units.Distance
  y: units.Distance
  pressure: float
  superpressure: float
  time_elapsed: dt.timedelta
  battery_soc: float

  @classmethod
  def from_balloon_state(
      cls,
      balloon_state: balloon.BalloonState) -> 'SimpleBalloonState':
    """Creates a SimpleBalloonState from a BalloonState."""
    return cls(balloon_state.x,
               balloon_state.y,
               balloon_state.pressure,
               balloon_state.superpressure,
               balloon_state.time_elapsed,
               balloon_state.battery_soc)


# TODO(joshgreaves): Add some notion of wind difficulty.
@dataclasses.dataclass
class EvaluationResult:
  """A class that holds the results of a single evaluation flight.

  Attributes:
    seed: The seed the controller was evaluated on.
    cumulative_reward: The total reward received by the agent during its flight.
    time_within_radius: The proportion of time the agent spent within the
      station keeping radius. This will be in [0, 1].
    out_of_power: True if the environment terminated because the balloon ran
      out of power.
    envelope_burst: True if the environment terminated because the envelope
      burst.
    zeropressure: True if the environment ended because the balloon
      zeropressured.
    final_timestep: The index of the final timestep. May be used to detect
      whether the balloon reached a terminal state.
    flight_path: The flight path the balloon took.
  """
  seed: int
  cumulative_reward: float
  time_within_radius: float
  out_of_power: bool
  envelope_burst: bool
  zeropressure: bool
  final_timestep: int
  flight_path: Sequence[SimpleBalloonState]

  def __str__(self) -> str:
    return (f'EvaluationResult(seed={self.seed}, '
            f'cumulative_reward={self.cumulative_reward}, '
            f'time_within_radius={self.time_within_radius}, '
            f'out_of_power={self.out_of_power}, '
            f'final_timestep={self.final_timestep})')


def _balloon_is_within_radius(balloon_state: balloon.BalloonState,
                              radius: units.Distance) -> bool:
  return units.relative_distance(balloon_state.x, balloon_state.y) <= radius


def eval_agent(agent: base_agent.Agent,
               env: balloon_env.BalloonEnv,
               eval_suite: suites.EvaluationSuite,
               *,
               render_period: int = 10,
               calculate_flight_path: bool = True) -> List[EvaluationResult]:
  """Evaluates an agent on a given test suite.

  If the agent being evaluated is deterministic, the result of this function
  will also be deterministic.

  Args:
    agent: The agent to evaluate.
    env: The environment to use for evaluation.
    eval_suite: The evaluation suite to evaluate the agent on.
    render_period: The period with which to render the environment.
      Only has an effect if the environment as a renderer.
    calculate_flight_path: Whether to calculate flight path.

  Returns:
    A list of evaluation results, corresponding to the seeds passed in by
      the eval_suite.
  """
  assert eval_suite.max_episode_length > 0, 'max_episode_length must be > 0.'

  results = list()

  logging.info('Starting evaluation of %s on %s', agent.get_name(), eval_suite)
  agent.set_mode(base_agent.AgentMode.EVAL)

  diagnostics = {}
  
  def simulator_write_diagnostics(diagnostic, simulator_state: simulator_data.SimulatorState, start=False):
    state = simulator_state.balloon_state
    if 'simulator' not in diagnostic:
      diagnostic['simulator'] = {'x':[],'y': [], 'z': [], 'wind':[], 'plan': [], 'power_soc': [], 'alt_safety': [], 'env_safety': [], 'power_safety': []}
    
    diagnostic['simulator']['x'].append(state.x.km)
    diagnostic['simulator']['y'].append(state.y.km)
    diagnostic['simulator']['z'].append(simulator_state.atmosphere.at_pressure(state.pressure).height.kilometers)
    
    if not start:
      diagnostic['simulator']['plan'].append(state.last_command)

    diagnostic['simulator']['power_soc'].append(state.battery_soc)
    diagnostic['simulator']['alt_safety'].append(state.altitude_safety_layer.safety_triggered)
    diagnostic['simulator']['env_safety'].append(state.envelope_safety_layer.safety_triggered)
    diagnostic['simulator']['power_safety'].append(state.power_safety_layer._triggered)

    wind_vector = simulator_state.wind_field.get_ground_truth(state.x, state.y, state.pressure, state.time_elapsed)
    diagnostic['simulator']['wind'].append([wind_vector.u.meters_per_second, wind_vector.v.meters_per_second])

  # def simulator_write_diagnostics_end(diagnostic, simulator_state: simulator_data.SimulatorState):
  #   diagnostics.append({'seed': seed, 'twr': twr, 'reward': total_reward, 'steps': step_count, 'diagnostic': diagnostic})

  for seed_idx, seed in enumerate(eval_suite.seeds):
    total_reward = 0.0
    steps_within_radius = 0
    flight_path = list()

    diagnostic = {}

    step_count = 0

    env.seed(seed)
    observation = env.reset()
    agent.update_forecast(env.get_wind_forecast())
    agent.update_atmosphere(env.get_atmosphere())
    agent.write_diagnostics_start(observation, diagnostic)
    action = agent.begin_episode(observation)
    agent.write_diagnostics(diagnostic)
    simulator_write_diagnostics(diagnostic, env.get_simulator_state(), start=True)

    # Implement json debugging

    out_of_power = False
    envelope_burst = False
    zeropressure = False
    with tqdm(total=eval_suite.max_episode_length) as pbar:
      while step_count < eval_suite.max_episode_length:
        observation, reward, is_done, info = env.step(action)

        action = agent.step(reward, observation)
        agent.write_diagnostics(diagnostic)

        total_reward += reward
        balloon_state = env.get_simulator_state().balloon_state
        if calculate_flight_path:
          flight_path.append(
              SimpleBalloonState.from_balloon_state((balloon_state)))
        steps_within_radius += _balloon_is_within_radius(balloon_state,
                                                        env.radius)

        simulator_write_diagnostics(diagnostic, env.get_simulator_state())

        if step_count % render_period == 0:
          env.render()  # No-op if renderer is None.

        step_count += 1

        if is_done:
          out_of_power = info.get('out_of_power', False)
          envelope_burst = info.get('envelope_burst', False)
          zeropressure = info.get('zeropressure', False)
          break
        
        pbar.update(1)

    twr = steps_within_radius / step_count
    agent.write_diagnostics_end(diagnostic)
    simulator_write_diagnostics(diagnostic, env.get_simulator_state())

    # 
    diagnostics[seed]={'seed': seed, 'twr': twr, 'reward': total_reward, 'steps': step_count, 'rollout': diagnostic}
    
    agent.end_episode(reward, is_done)

    eval_result = EvaluationResult(
        seed=seed,
        cumulative_reward=total_reward,
        time_within_radius=twr,
        out_of_power=out_of_power,
        envelope_burst=envelope_burst,
        zeropressure=zeropressure,
        final_timestep=step_count,
        flight_path=flight_path)

    # This logs the fraction of seeds evaluated, the seed, and the eval result.
    # e.g. "10 / 100: (seed 10) EvalResult(cumulative_reward=...)"
    logging.info('%d / %d: (seed %d) %s',
                 seed_idx + 1,
                 len(eval_suite.seeds),
                 seed,
                 eval_result)
    logging.info('Power safety layer violations: %d', env.arena.get_balloon_state().power_safety_layer._triggered)
    logging.info('Altitude safety layer violations: %d', env.arena.get_balloon_state().altitude_safety_layer.safety_triggered)
    logging.info('Envelope safety layer violations: %d', env.arena.get_balloon_state().envelope_safety_layer.safety_triggered)
    
    
    results.append(eval_result)

  datafile = f'diagnostics/{type(agent).__name__}-{int(dt.datetime.now().timestamp()*1000)}.json'
  with open(datafile, 'w', encoding='utf-8') as f:
    json.dump(diagnostics, f, ensure_ascii=False, indent=4)


  return results
