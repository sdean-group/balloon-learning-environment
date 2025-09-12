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

"""A wind field that interpolates from a grid."""

import datetime as dt
from typing import List, Sequence, Union

from balloon_learning_environment.env import grid_wind_field_sampler
from balloon_learning_environment.env import wind_field
from balloon_learning_environment.env.features import NamedPerciatelliFeatures
from balloon_learning_environment.utils import units
from balloon_learning_environment.utils import constants
import jax
from jax import numpy as jnp
import numpy as np
import scipy.interpolate
from atmosnav import JaxTree
# from memory_profiler import profile

class JaxColumnBasedWindField(wind_field.JaxWindField, JaxTree):
  def __init__(self, pressure_levels, wind_column):
    self.pressure_levels = pressure_levels
    self.wind_columns = wind_column

  def get_forecast(self, x:float, y:float, pressure:float, elapsed_time:float) -> jnp.ndarray:
    dist = jnp.hypot(x, y)
    away = jnp.where(dist < 1e-5, jnp.zeros(2), jnp.array([x, y]) / dist)

    interp = jax.scipy.interpolate.RegularGridInterpolator(
      (self.pressure_levels, ), 
      self.wind_columns, 
      fill_value=None)
    wind_vec = interp(jnp.array([pressure]))[0]

    jnp.where(jnp.logical_and(pressure>=self.pressure_levels[0], pressure<=self.pressure_levels[-1]), wind_vec, away)
    return wind_vec
  
  def tree_flatten(self):
    return (self.pressure_levels, self.wind_columns), {}

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    return JaxColumnBasedWindField(*children)

    # self._winds = features[16:]
    # wind = self._winds[level * 3:level * 3 + 3]
    #  def _nearest_pressure_level(
    #   self, pressure: float) -> int:
    # """Returns the pressure level nearest to a given pressure value.

    # Args:
    #   pressure: Desired pressure.

    # Returns:
    #   level: The corresponding nearest level.
    # """
    # if pressure < self.min_pressure or pressure > self.max_pressure:
    #   # A warning has been logged. Quantize the pressure level.
    #   pressure = min(max(pressure, self.min_pressure), self.max_pressure)

    # # Basically quantize 'pressure'.
    # # Note: this assumes uniform pressure levels.
    # assert len(self.pressure_levels) >= 2
    # delta = self.pressure_levels[1] - self.pressure_levels[0]

    # rescaled = (pressure - self.min_pressure) / delta
    # level = int(round(rescaled))

    # assert level >= 0 and level < self.num_pressure_levels
    # return level



class JaxGridBasedWindField(wind_field.JaxWindField, JaxTree):

  def __init__(self, field_shape, field):
    super(JaxGridBasedWindField, self).__init__()
    self.field_shape = field_shape
    self.field = field 

    self._grid = (
        self.field_shape.latlng_grid_points(),    # Lats.
        self.field_shape.latlng_grid_points(),    # Lngs.
        self.field_shape.pressure_grid_points(),  # Pressures.
        self.field_shape.time_grid_points())      # Times.
    
  #@profile
  @jax.jit
  def get_forecast(self, x: float, y: float, pressure: float,
                   elapsed_time: float) -> jnp.ndarray:
    """
    x,y are km
    pressure is pascals (Pa)
    elapsed_time is seconds
    """
    point = self._prepare_get_forecast_inputs(x, y, pressure, elapsed_time)
    interp = jax.scipy.interpolate.RegularGridInterpolator(self._grid, self.field)
    uv = interp(point)
    return uv[0] # uv[0][0] is lat mps for wind, uv[0][1] is lon mps

  def _boomerang(self, t, max_val):
    cycle_direction = (t // max_val) % 2
    remainder = t % max_val

    # Use lax.cond to handle the conditional direction.
    return jax.lax.cond(cycle_direction == 0, lambda op: op[0], lambda op: op[1] - op[0], operand=(remainder, max_val))

  #@profile
  def _prepare_get_forecast_inputs(self, x: float, y: float, pressure: float, elapsed_time: float) -> jnp.ndarray:
    x = jnp.clip(x, -self.field_shape.latlng_displacement_km, self.field_shape.latlng_displacement_km)
    y = jnp.clip(y, -self.field_shape.latlng_displacement_km, self.field_shape.latlng_displacement_km)
    pressure = jnp.clip(pressure, self.field_shape.min_pressure_pa, self.field_shape.max_pressure_pa)

    elapsed_hours = elapsed_time / 3600.0
    time_field_position = jax.lax.cond(
      elapsed_hours < self.field_shape.time_horizon_hours,
      lambda op: op[0],
      lambda op: self._boomerang(op[0], op[1]),
      operand=(elapsed_hours, self.field_shape.time_horizon_hours))

    return jnp.array([ x, y, pressure, time_field_position ])
  
  def tree_flatten(self):
    return (self.field_shape, self.field), {}

  @classmethod
  def tree_unflatten(cls, aux_data, children): 
    return JaxGridBasedWindField(*children)

  
class JaxInterpolatingWindField(wind_field.JaxWindField, JaxTree):
  def __init__(self, column_wind_field: JaxColumnBasedWindField, grid_wind_field: JaxGridBasedWindField, gk_distance: float, gk_time: float, column_x: float, column_y: float):
    self.column_wind_field = column_wind_field
    self.grid_wind_field = grid_wind_field

    self.gk_distance = gk_distance # km scalar
    self.gk_time = gk_time # hours scalar

    self.column_x = column_x
    self.column_y = column_y
  
  def get_forecast(self, x: float, y: float, pressure: float,
                   elapsed_time: float) -> jnp.ndarray:
    
    column_wind = self.column_wind_field.get_forecast(x, y, pressure, elapsed_time)
    grid_wind = self.grid_wind_field.get_forecast(x, y, pressure, elapsed_time)

    dist = jnp.hypot(x - self.column_x, y - self.column_y) # km
    hours = elapsed_time / 3600.0 # Hours
    weight = jnp.exp(-((dist / self.gk_distance)**2 + (hours / self.gk_time)**2))

    return column_wind * weight + (1 - weight) * grid_wind

  def tree_flatten(self):
    return (self.column_wind_field, self.grid_wind_field, self.gk_distance, self.gk_time, self.column_x, self.column_y), {}

  @classmethod
  def tree_unflatten(cls, aux_data, children): 
    return JaxInterpolatingWindField(*children)


class GridBasedWindField(wind_field.WindField):
  """A wind field that interpolates from a grid."""

  def to_jax_wind_field(self):
    return JaxGridBasedWindField(self.field_shape, self.jax_field)

  def __init__(
      self,
      wind_field_sampler: grid_wind_field_sampler.GridWindFieldSampler):
    """GridBasedWindField Constructor.

    Args:
      wind_field_sampler: An object that can be used to sample wind fields.
    """
    super(GridBasedWindField, self).__init__()
    self._wind_field_sampler = wind_field_sampler
    self._jax_wind_field_sampler = wind_field_sampler.to_jax_grid_wind_field_sampler()
    self.field_shape = self._wind_field_sampler.field_shape
    self.field = None  # Will be initialized with reset_forecast.
    self.jax_field = None

    # NOTE(scandido): We convert the field from a jax.numpy arrays to a numpy
    # arrays here, otherwise it'll be converted on the fly every time we
    # interpolate (due to scipy.interpolate). This conversion is not a huge cost
    # but the conversion on self.field (see reset() method) is significant so
    # we also do this one for completeness.
    self._grid = (
        np.asarray(self.field_shape.latlng_grid_points()),    # Lats.
        np.asarray(self.field_shape.latlng_grid_points()),    # Lngs.
        np.asarray(self.field_shape.pressure_grid_points()),  # Pressures.
        np.asarray(self.field_shape.time_grid_points()))      # Times.

  def reset_forecast(self, key: jnp.ndarray, date_time: dt.datetime) -> None:
    """Resets the wind field.

    Note: Must be overridden by child class!
    The child class should set self.field here. The shape of self.field
    should match the field_shape passed to the constructor.

    Args:
      key: A PRNG key used to sample a new location and time for the wind field.
      date_time: An instance of a datetime object, representing the start
          of the wind field.
    """
    self.jax_field = self._jax_wind_field_sampler.sample_field(key, date_time)
    self.field = self._wind_field_sampler.sample_field(key, date_time)

  def get_forecast(self, x: units.Distance, y: units.Distance, pressure: float,
                   elapsed_time: dt.timedelta) -> wind_field.WindVector:
    """Gets a wind in the wind field at the specified location and time.

    Args:
      x: An x offset (parallel to latitude).
      y: A y offset (parallel to longitude).
      pressure: A pressure level in pascals.
      elapsed_time: The time offset from the beginning of the wind field.

    Returns:
      The wind vector at the specified position and time.

    Raises:
      RuntimeError: if called before reset().
    """
    if self.field is None:
      raise RuntimeError('Must call reset before get_forecast.')

    point = self._prepare_get_forecast_inputs(x, y, pressure, elapsed_time)
    point = point.reshape(-1)
    uv = scipy.interpolate.interpn(
        self._grid, self.field, point, fill_value=True)
    return wind_field.WindVector(units.Velocity(mps=uv[0][0]),
                                 units.Velocity(mps=uv[0][1]))

  def get_forecast_column(
      self,
      x: units.Distance,
      y: units.Distance,
      pressures: Sequence[float],
      elapsed_time: dt.timedelta) -> List[wind_field.WindVector]:
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

    Raises:
      RuntimeError: if called before reset().
    """
    if self.field is None:
      raise RuntimeError('Must call reset before get_forecast.')

    point = self._prepare_get_forecast_inputs(x, y, pressures, elapsed_time)
    uv = scipy.interpolate.interpn(
        self._grid, self.field, point, fill_value=True)

    result = list()
    for i in range(len(pressures)):
      result.append(wind_field.WindVector(units.Velocity(mps=uv[i][0]),
                                          units.Velocity(mps=uv[i][1])))
    return result

  @staticmethod
  def _boomerang(t: float, max_val: float) -> float:
    """Computes a value that boomerangs between 0 and max_val."""
    cycle_direction = int(t / max_val) % 2
    remainder = t % max_val

    if cycle_direction % 2 == 0:  # Forward.
      return remainder
    else:  # Backward.
      return max_val - remainder

  def _prepare_get_forecast_inputs(self,
                                   x: units.Distance,
                                   y: units.Distance,
                                   pressure: Union[Sequence[float], float],
                                   elapsed_time: dt.timedelta) -> np.ndarray:
    # TODO(bellemare): Give a units might be wrong warning if querying 10,000s
    # km away.

    # NOTE(scandido): We extend the field beyond the limits of the VAE using
    # the values at the boundary.
    x_km = x.kilometers
    y_km = y.kilometers
    x_km = np.clip(x_km, -self.field_shape.latlng_displacement_km,
                   self.field_shape.latlng_displacement_km).item()
    y_km = np.clip(y_km, -self.field_shape.latlng_displacement_km,
                   self.field_shape.latlng_displacement_km).item()
    pressure = np.clip(pressure, self.field_shape.min_pressure_pa,
                       self.field_shape.max_pressure_pa)

    # Generated wind fields have a fixed time dimension, often 48 hours.
    # Typically queries will be between 0-48 hours so most of the time it is
    # simple to query a point in the field. However, to extend the limit of
    # the field we transform times 48+ hours out to some time in the 0-48
    # well defined region in such a way that two close times will remain close
    # (and thus not have a suddenly changing wind field). We use a "boomerang",
    # i.e., time reflects backwards after 48 hours until 2*48 hours at which
    # point time goes forward and so on.
    elapsed_hours = units.timedelta_to_hours(elapsed_time)
    if elapsed_hours < self.field_shape.time_horizon_hours:
      time_field_position = elapsed_hours
    else:
      time_field_position = self._boomerang(
          elapsed_hours,
          self.field_shape.time_horizon_hours)

    num_points = 1 if isinstance(pressure, float) else len(pressure)
    point = np.empty((num_points, 4), dtype=np.float32)
    point[:, 0] = x_km
    point[:, 1] = y_km
    point[:, 2] = pressure
    point[:, 3] = time_field_position

    return point

