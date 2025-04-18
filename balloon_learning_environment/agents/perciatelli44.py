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

"""A frozen Perciatelli44 agent."""

from typing import Sequence

from balloon_learning_environment.agents import agent
from balloon_learning_environment.models import models
from balloon_learning_environment.env.features import NamedPerciatelliFeatures
import numpy as np
import tensorflow as tf
from balloon_learning_environment.utils import constants
from balloon_learning_environment.utils import units
import datetime as dt
from balloon_learning_environment.utils import transforms
import pickle


def load_perciatelli_session() -> tf.compat.v1.Session:
  serialized_perciatelli = models.load_perciatelli44()

  sess = tf.compat.v1.Session()
  graph_def = tf.compat.v1.GraphDef()
  graph_def.ParseFromString(serialized_perciatelli)

  tf.compat.v1.import_graph_def(graph_def)
  return sess


class Perciatelli44(agent.Agent):
  """Perciatelli44 Agent.

  This is the agent which was reported as state of the art in
  "Autonomous navigation of stratospheric balloons using reinforcement
  learning" (Bellemare, Candido, Castro, Gong, Machado, Moitra, Ponda,
  and Wang, 2020).

  This agent has its weights frozen, and is intended for comparison in
  evaluation, not for retraining.
  """

  def __init__(self, num_actions: int, observation_shape: Sequence[int]):
    super(Perciatelli44, self).__init__(num_actions, observation_shape)

    if num_actions != 3:
      raise ValueError('Perciatelli44 only supports 3 actions.')
    if list(observation_shape) != [1099]:
      raise ValueError('Perciatelli44 only supports 1099 dimensional input.')

    # TODO(joshgreaves): It would be nice to use the saved_model API
    # for loading the Perciatelli graph.
    # TODO(joshgreaves): We wanted to avoid a dependency on TF, but adding
    # this to the agent registry makes TF a necessity.
    self._sess = load_perciatelli_session()
    self._action = self._sess.graph.get_tensor_by_name('sleepwalk_action:0')
    self._q_vals = self._sess.graph.get_tensor_by_name('q_values:0')
    self._observation = self._sess.graph.get_tensor_by_name('observation:0')

  def begin_episode(self, observation: np.ndarray) -> int:
    observation = observation.reshape((1, 1099))
    q_vals = self._sess.run(self._q_vals,
                            feed_dict={self._observation: observation})
    return np.argmax(q_vals).item()

  def step(self, reward: float, observation: np.ndarray) -> int:
    observation = observation.reshape((1, 1099))
    q_vals = self._sess.run(self._q_vals,
                            feed_dict={self._observation: observation})
    return np.argmax(q_vals).item()

  def end_episode(self, reward: float, terminal: bool = True) -> None:
    pass

def get_distilled_model_features(perciatelli_features: np.ndarray, wind_forecast: agent.WindField, elapsed_time: dt.timedelta) -> np.ndarray:
  # TODO: need to have a good way to determine elapsed time, maybe change this to be a model that takes in the actual balloon state
  # and contains two feature constructors internally: perciatelli and distilled feature constructor. It will be easier to write down 
  # our distilled model parameters

  num_wind_levels = 181
  distilled_features = np.zeros(4 + 3 * num_wind_levels)
  features = NamedPerciatelliFeatures(perciatelli_features)

  distance_km = transforms.undo_squash_to_unit_interval(features.distance_to_station, 250)
  angle_to_station = np.arctan2(features.sin_heading_to_station, features.cos_heading_to_station)
  # TODO: unsure of units
  distilled_features[0] = features.balloon_pressure 
  distilled_features[1] = distance_km
  distilled_features[2] = angle_to_station
  distilled_features[3] = features.battery_charge

  # print(distance_km, angle_to_station)

  pressure_levels = np.linspace(constants.PERCIATELLI_PRESSURE_RANGE_MIN,
                                constants.PERCIATELLI_PRESSURE_RANGE_MAX,
                                num_wind_levels)

  x,y = units.Distance(km=-distance_km * features.sin_heading_to_station), units.Distance(km=-distance_km * features.cos_heading_to_station)

  wind_column = wind_forecast.get_forecast_column(x, y, pressure_levels, elapsed_time)

  # from minimum pressure to maximum pressure
  for i, wind_vector in enumerate(wind_column):
    distilled_features[4 + i * 3 + 0] = np.sqrt(wind_vector.u.meters_per_second**2 + wind_vector.v.meters_per_second**2)
    distilled_features[4 + i * 3 + 1] = np.arctan2(wind_vector.v.meters_per_second, wind_vector.u.meters_per_second)
    distilled_features[4 + i * 3 + 2] = pressure_levels[i]

  return distilled_features

class Perciatelli44DataCollector(agent.Agent):
  """Perciatelli44 Agent.

  This is the agent which was reported as state of the art in
  "Autonomous navigation of stratospheric balloons using reinforcement
  learning" (Bellemare, Candido, Castro, Gong, Machado, Moitra, Ponda,
  and Wang, 2020).

  This agent has its weights frozen, and is intended for comparison in
  evaluation, not for retraining.
  """

  def __init__(self, num_actions: int, observation_shape: Sequence[int]):
    super(Perciatelli44DataCollector, self).__init__(num_actions, observation_shape)

    if num_actions != 3:
      raise ValueError('Perciatelli44 only supports 3 actions.')
    if list(observation_shape) != [1099]:
      raise ValueError('Perciatelli44 only supports 1099 dimensional input.')

    # TODO(joshgreaves): It would be nice to use the saved_model API
    # for loading the Perciatelli graph.
    # TODO(joshgreaves): We wanted to avoid a dependency on TF, but adding
    # this to the agent registry makes TF a necessity.
    self._sess = load_perciatelli_session()
    self._action = self._sess.graph.get_tensor_by_name('sleepwalk_action:0')
    self._q_vals = self._sess.graph.get_tensor_by_name('q_values:0')
    self._observation = self._sess.graph.get_tensor_by_name('observation:0')

    # Stuff for data collecting
    self.forecast = None
    self.elapsed_time = dt.timedelta(seconds=0)

    self.data_collection = []

  def begin_episode(self, observation: np.ndarray) -> int:
    self.elapsed_time = dt.timedelta(seconds=0)
    distilled_features = get_distilled_model_features(observation, self.forecast, self.elapsed_time)
    
    observation = observation.reshape((1, 1099))
    q_vals = self._sess.run(self._q_vals,
                            feed_dict={self._observation: observation})
    
    self.data_collection.append((distilled_features, q_vals))

    return np.argmax(q_vals).item()

  def step(self, reward: float, observation: np.ndarray) -> int:
    self.elapsed_time += dt.timedelta(minutes=3)
    distilled_features = get_distilled_model_features(observation, self.forecast, self.elapsed_time)
    observation = observation.reshape((1, 1099))
    q_vals = self._sess.run(self._q_vals,
                            feed_dict={self._observation: observation})


    self.data_collection.append((distilled_features, q_vals))

    return np.argmax(q_vals).item()

  def end_episode(self, reward: float, terminal: bool = True) -> None:
    self.elapsed_time = dt.timedelta(seconds=0)

  def update_forecast(self, forecast):
    self.forecast = forecast

  def write_diagnostics_end(self, diagnostics):
    # Pickle data_collection
    with open('perciatelli-training-data', 'wb') as f:
      pickle.dump(self.data_collection, f)

    # # Read pickled file
    # with open('perciatelli-training-data', 'rb') as f:
    #   data_collection = pickle.load(f)