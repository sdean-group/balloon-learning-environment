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

"""Stratospheric wind fields.

These classes allow point-based lookups into a 4D stratospheric wind field.

This file defines WindField, the base wind field class that is subclassed
by different wind fields. It also handles adding wind noise.
"""

import abc
import datetime as dt
from typing import List, NamedTuple, Sequence

from balloon_learning_environment.env import simplex_wind_noise
from balloon_learning_environment.utils import units
import gin
import jax
import numpy as np
from jax import numpy as jnp
from atmosnav import *

# WindVector contains the following elements:
#   u: Wind magnitude along the x axis in meters per second.
#   v: Wind magnitude along the y axis in meters per second.
# Note: x axis is parallel to latitude, and y axis is parallel to longitude.
class WindVector(NamedTuple):
  """Describes the wind at a given location."""
  u: units.Velocity
  v: units.Velocity

  def add(self, other: 'WindVector') -> 'WindVector':
    if not isinstance(other, WindVector):
      raise NotImplementedError(
          f'Cannot add WindVector with {type(other)}')
    return WindVector(self.u + other.u, self.v + other.v)

  def __str__(self) -> str:
    return f'({self.u}, {self.v})'

class JaxWindField(JaxTree):
  """ simplified wind class for jax environments (e.g. gradient of a function that reads wind data) """

  @abc.abstractmethod
  def get_forecast(self, x: float, y: float, pressure: float, elapsed_time: float):
    """ one wind forecast please """

@gin.configurable
class WindField(abc.ABC):
  """Abstract class for point-based lookups in a wind field."""

  def __init__(self):
    self._noise_model = SimplexWindNoise()

  def to_jax_wind_field(self):
    raise NotImplementedError('no conversion to jax wind field')

  @abc.abstractmethod
  def reset_forecast(self, key: jnp.ndarray, date_time: dt.datetime) -> None:
    """Resets the wind forecast.

    Args:
      key: A jax PRNGKey used for sampling a new wind field forecast.
      date_time: An instance of a datetime object, representing the start of
                 the wind field.
    """

  @abc.abstractmethod
  def get_forecast(self, x: units.Distance, y: units.Distance, pressure: float,
                   elapsed_time: dt.timedelta) -> WindVector:
    """Returns forecast at a point in the field.

    Args:
      x: Distance from the station keeping target along the latitude
        parallel.
      y: Distance from the station keeping target along the longitude
        parallel.
      pressure: Pressure at this point in the wind field in Pascals. (This is
        a proxy for altitude.)
      elapsed_time: Elapsed time from the "beginning" of the wind field.

    Returns:
      A WindVector for the position in the WindField.
    """

  def get_forecast_column(self,
                          x: units.Distance,
                          y: units.Distance,
                          pressures: Sequence[float],
                          elapsed_time: dt.timedelta) -> List[WindVector]:
    """A convenience function for getting multiple forecasts in a column.

    This allows a simple optimization of the generative wind field.

    Args:
      x: Distance from the station keeping target along the latitude
        parallel.
      y: Distance from the station keeping target along the longitude
        parallel.
      pressures: Multiple pressures to get a forecast for, in Pascals. (This is
        a proxy for altitude.)
      elapsed_time: Elapsed time from the "beginning" of the wind field.

    Returns:
      WindVectors for each pressure level in the WindField.
    """
    return [self.get_forecast(x, y, pressure, elapsed_time)
            for pressure in pressures]

  def reset(self, key: jnp.ndarray, date_time: dt.datetime) -> None:
    """Resets the wind field with a specific PRNG key.

    Args:
      key: A jax PRNGKey used for sampling a new wind field.
      date_time: An instance of a datetime object, representing the start of
                 the wind field.
    """
    noise_key, forecast_key = jax.random.split(key, num=2)
    self._noise_model.reset_wind_noise(noise_key, date_time)
    self.reset_forecast(forecast_key, date_time)

  def get_ground_truth(self,
                       x: units.Distance, y: units.Distance, pressure: float,
                       elapsed_time: dt.timedelta) -> WindVector:
    """Returns ground truth at a point in the field.

    Args:
      x: Distance from the station keeping target along the latitude
        parallel.
      y: Distance from the station keeping target along the longitude
        parallel.
      pressure: Pressure at this point in the wind field in Pascals. (This is
        a proxy for altitude.)
      elapsed_time: Elapsed time from the "beginning" of the wind field.

    Returns:
      A WindVector for the position in the WindField.
    """
    forecast = self.get_forecast(x, y, pressure, elapsed_time)

    noise = self._noise_model.get_wind_noise(x, y, pressure, elapsed_time)
    return forecast.add(noise)


@gin.configurable(allowlist=[])
class SimpleStaticWindField(WindField):
  """A static wind field.

  This wind field flows in the four cardinal directions based on the pressure
  of the point in the field.
  """
  def to_jax_wind_field(self):
    return JaxSimpleStaticWindField()

  def reset_forecast(
      self, unused_key: jnp.ndarray, unused_date_time: dt.datetime) -> None:
    pass

  def get_forecast(
      self, unused_x: units.Distance, unused_y: units.Distance,
      pressure: float, unused_elapsed_time: dt.timedelta) -> WindVector:
    """Returns wind at a point in the field.

    Args:
      unused_x: Distance from the station keeping target along the latitude
        parallel.
      unused_y: Distance from the station keeping target along the longitude
        parallel.
      pressure: Pressure at this point in the wind field in Pascals. (This is a
        proxy for altitude.)
      unused_elapsed_time: Elapsed time from the "beginning" of the wind field.

    Returns:
      A WindVector for the position in the WindField.
    """
    if pressure < 8000.0:
      return WindVector(units.Velocity(mps=10.0), units.Velocity(mps=0.0))
    elif pressure < 10000.0:
      return WindVector(units.Velocity(mps=0.0), units.Velocity(mps=10.0))
    elif pressure < 12000.0:
      return WindVector(units.Velocity(mps=-10.0), units.Velocity(mps=0.0))
    else:
      return WindVector(units.Velocity(mps=0.0), units.Velocity(mps=-10.0))
    
class JaxSimpleStaticWindField(JaxWindField):
  """A static wind field for jax environments."""

  def get_forecast(self, x: float, y: float, pressure: float, elapsed_time: float):
    """Returns wind at a point in the field.

    Args:
      x: Distance from the station keeping target along the latitude
        parallel.
      y: Distance from the station keeping target along the longitude
        parallel.
      pressure: Pressure at this point in the wind field in Pascals. (This is a
        proxy for altitude.)
      elapsed_time: Elapsed time from the "beginning" of the wind field.

    Returns:
      A WindVector for the position in the WindField.
    """
    return jax.lax.cond(
        pressure < 8000.0,
        lambda _: jnp.array([10.0, 0.0]),
        lambda _: jax.lax.cond(
            pressure < 10000.0,
            lambda _: jnp.array([0.0, 10.0]),
            lambda _: jax.lax.cond(
                pressure < 12000.0,
                lambda _: jnp.array([-10.0, 0.0]),
                lambda _: jnp.array([0.0, -10.0]),
                operand=None
            ),
            operand=None
        ),
        operand=None
    )
  
  def tree_flatten(self):
    return tuple(), {}

  @classmethod
  def tree_unflatten(cls, aux_data, children): 
    return JaxSimpleStaticWindField()
  
class Pt2CenterWindField(WindField):
  """A wind field that flows from a point to the center of the field."""

  def to_jax_wind_field(self):
    return JaxPt2CenterWindField()

  def reset_forecast(self, key: jnp.ndarray, date_time: dt.datetime) -> None:
    pass

  def get_forecast(self, x: units.Distance, y: units.Distance, pressure: float,
                    elapsed_time: dt.timedelta) -> WindVector:
    
    if (x.km**2 + y.km**2) < 0.01: 
      return WindVector(units.Velocity(mps=0), units.Velocity(mps=0))
  
    mag = (x.km**2 + y.km**2)**0.5
    return WindVector(units.Velocity(mps= 10 * -x.km / mag), units.Velocity(mps= 10 * -y.km / mag))
    

class JaxPt2CenterWindField(JaxWindField):

  def get_forecast(self, x: float, y: float, pressure: float, elapsed_time: float):
    mag = (x**2 + y**2)**0.5
    return jax.lax.cond(
      x**2 + y**2 < 0.01,
      lambda op: jnp.array([ 0.0, 0.0 ]),
      lambda op: jnp.array([-op[0] / op[2], -op[1] / op[2]]),
      operand=(x,y,mag))
  
  def tree_flatten(self):
    return tuple(), {}

  @classmethod
  def tree_unflatten(cls, aux_data, children): 
    return JaxPt2CenterWindField()
  

class SpinnyWindField(WindField):
  """A wind field that flows from a point to the center of the field."""

  def to_jax_wind_field(self):
    return JaxSpinnyWindField()

  def reset_forecast(self, key: jnp.ndarray, date_time: dt.datetime) -> None:
    pass

  def get_forecast(self, x: units.Distance, y: units.Distance, pressure: float,
                    elapsed_time: dt.timedelta) -> WindVector:
    
    a, b = 3689.3997945759265, 101517.76878288877
    n = 2 *np.pi *(pressure - a) / (b - a)
    return WindVector(units.Velocity(mps=10 * np.cos(n)), units.Velocity(mps=10* np.sin(n)))
    

class JaxSpinnyWindField(JaxWindField):

  def get_forecast(self, x: float, y: float, pressure: float, elapsed_time: float):
    a, b = 3689.3997945759265, 101517.76878288877
    n = 10 *jnp.pi *(pressure - a) / (b - a)
    return jnp.array([ 10 * jnp.cos(n), 10 * jnp.sin(n) ])
  
  def tree_flatten(self):
    return tuple(), {}

  @classmethod
  def tree_unflatten(cls, aux_data, children): 
    return JaxSpinnyWindField()

# TODO(bellemare): Should this be moved to units?
class SimplexWindNoise(object):
  """Wind noise model based on the simplex algorithm."""

  def __init__(self):
    self.noise_u = simplex_wind_noise.NoisyWindComponent(which='u')
    self.noise_v = simplex_wind_noise.NoisyWindComponent(which='v')

  def reset_wind_noise(self, key: jnp.ndarray, date_time: dt.datetime) -> None:
    """Resets the wind noise model.

    Args:
      key: A jax PRNGKey used for sampling a new wind field.
      date_time: An instance of a datetime object, representing the start of
                 the wind field.
    """
    del date_time

    noise_u_key, noise_v_key = jax.random.split(key, num=2)
    self.noise_u.reset(noise_u_key)
    self.noise_v.reset(noise_v_key)

  def get_wind_noise(
      self, x: units.Distance, y: units.Distance, pressure: float,
      elapsed_time: dt.timedelta) -> WindVector:
    """Returns noise at a point in the field."""
    wind_noise_u = self.noise_u.get_noise(x, y, pressure, elapsed_time)
    wind_noise_v = self.noise_v.get_noise(x, y, pressure, elapsed_time)

    return WindVector(
        units.Velocity(meters_per_second=wind_noise_u),
        units.Velocity(meters_per_second=wind_noise_v))

