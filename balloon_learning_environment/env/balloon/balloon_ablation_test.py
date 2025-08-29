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

"""Simulator for a stratospheric superpressure balloon.

This simulates a simplified model of stratospheric balloon flight for an
altitude controlled superpressure balloon. The state for the balloon exists
above a simplified Cartesian plane rather than a real coordinate on the globe.

We define the coordinate space with [x, y] as the position of the balloon
relative to the station keeping target. x is kilometers along the latitude line
and y is kilometers along the latitude line. pressure is the barometric pressure
(similar to altitude, but with a nontrivial relationship). time_elapsed is
time but relative to start of the simulation / wind field generation.

Typical usage:

  wind_field = ...
  balloon = Balloon(0., 0., 6000)  # At the station keeping target, 6000 Pa.
  stride = timedelta(minutes=3)
  horizon = timedelta(days=2)
  for _ in range(horizon / stride):
    command = np.random.choice([
      AltitudeControlCommand.UP,
      AltitudeControlCommand.DOWN,
      AltitudeControlCommand.STAY])
    balloon.simulate_step(wind_field, command, stride)
    print(balloon.x, balloon.y, balloon.pressure)
"""

import dataclasses
import datetime as dt
import enum
from typing import Any, Dict, Tuple, Union

from absl import logging
from balloon_learning_environment.env import wind_field
from balloon_learning_environment.env.balloon import acs
from balloon_learning_environment.env.balloon import altitude_safety
from balloon_learning_environment.env.balloon import control
from balloon_learning_environment.env.balloon import envelope_safety
from balloon_learning_environment.env.balloon import power_safety
from balloon_learning_environment.env.balloon import solar
from balloon_learning_environment.env.balloon import standard_atmosphere
from balloon_learning_environment.env.balloon import thermal
from balloon_learning_environment.env.balloon.balloon import BalloonState, BalloonStatus, calculate_superpressure_and_volume
from balloon_learning_environment.utils import constants
from balloon_learning_environment.utils import spherical_geometry
from balloon_learning_environment.utils import units
import numpy as np

from typing import Union

import s2sphere as s2


class BalloonAblation:
  """A simulation of a stratospheric balloon.

  This class holds the system state vector and equations of motion
  (simulate_step) for simulating a stratospheric balloon.
  """

  @staticmethod
  def get_ablation(update_step):
    return lambda state: BalloonAblation(state, update_step)

  def __init__(self, balloon_state: BalloonState, update_step):
    self.state = balloon_state
    self.update_step = update_step

  def simulate_step(
      self,
      wind_vector: wind_field.WindVector,
      atmosphere: standard_atmosphere.Atmosphere,
      action: Union[control.AltitudeControlCommand, float],
      time_delta: dt.timedelta,
      stride: dt.timedelta = dt.timedelta(seconds=10),
  ) -> None:
    """Steps forward the simulation.

    This moves the balloon's state forward according to the dynamics of motion
    for a stratospheric balloon.

    Args:
      wind_vector: A vector corresponding to the wind to apply to the balloon.
      atmosphere: The atmospheric conditions the balloon is flying in.
      action: An AltitudeControlCommand for the system to take during this
        simulation step, i.e., up/down/stay.
      time_delta: How much time is elapsing during this step. Must be a multiple
        of stride.
      stride: The step size for the simulation of the balloon physics.
    """

    self.state.last_command = action

    assert self.state.status == BalloonStatus.OK, (
        'Stepping balloon after a terminal event occured. '
        f'({self.state.status.name})')

    # Want to enable the environment to work with continuous and discrete actions
    using_discrete = isinstance(action, control.AltitudeControlCommand)

    # The safety layers may prevent some actions from taking effect in
    # certain situations. While the correct interpretation
    # is, e.g., that even when a down action is commanded the altitude control
    # system may not be powered up, we simplify the code readability by
    # remapping the action that takes effect to, in this example, the stay
    # command.
    # The envelope/atitude safety layers trumps the power safety layer. This is
    # because, at worst, the envelope/altitude safety layers only recommend
    # ascending, which doesn't take any extra power.
    # Finally, the altitude safety layer trumps the envelope safety layer.
    # This is because ascending shouldn't be harmful to superpressure in most
    # situtations.
    effective_action = action
    if self.state.power_safety_layer_enabled:
        effective_action = self.state.power_safety_layer.get_action(
            effective_action, self.state.date_time,
            self.state.nighttime_power_load, self.state.battery_charge,
            self.state.battery_capacity)
    # TODO: make safety layers work with continuous actions
    if False: # using_discrete:
      # NOTE: continuous actions don't have any safety layers implemented
      effective_action = self.state.envelope_safety_layer.get_action(
          effective_action, self.state.superpressure)
      effective_action = self.state.altitude_safety_layer.get_action(
          effective_action, atmosphere, self.state.pressure)

    outer_stride = int(time_delta.total_seconds())
    inner_stride = int(stride.total_seconds())
    assert outer_stride % inner_stride == 0, (
        f'The outer simulation stride (time_delta={time_delta}) must be a '
        f'multiple of the inner simulation stride (stride={stride})')


    simulation_steps = outer_stride // inner_stride

    # this will quantize the continuous actions to discrete ones instead of using 
    # modified dynamics
    _use_quantized_actions = False
    if not using_discrete and _use_quantized_actions: # if continuous then calculate relevant values for quantized actions
      # calculate the number of steps of up/down to take to approximate the continuous action
      _quantized_action = control.AltitudeControlCommand.UP if action > 0 else control.AltitudeControlCommand.DOWN
      _quantized_steps = min((abs(action) * outer_stride)//inner_stride, simulation_steps)

      print(f'using discrete actions, converting {action} into {_quantized_steps} steps of {_quantized_action} out of {simulation_steps} total steps')
        

    # print('running', outer_stride // inner_stride, 'times')
    for _ in range(simulation_steps):

      # Choose which dynamics to use based on action type
      if using_discrete:

        # NOTE: this version simply uses the discrete stuff,
        state_changes = self.update_step(
            self.state, wind_vector, atmosphere, effective_action, stride)

      elif _use_quantized_actions:
        # print('doing discrete dynamics')
        # NOTE: this version uses the quantized actions

        action_to_run = _quantized_action if _quantized_steps > 0 else control.AltitudeControlCommand.STAY
        state_changes = self.update_step(
            self.state, wind_vector, atmosphere, action_to_run, stride)

        _quantized_steps -= 1

      else:
        # print('doing continuous dynamics')
        raise NotImplementedError('not supported with ablation yet')
      for k, v in state_changes.items():
        setattr(self.state, k, v)

      if self.state.status != BalloonStatus.OK:
        break

def _clip(x, minval, maxval):
  """A clip function that should be faster than numpy for scalars."""
  return min(max(x, minval), maxval)

def make_ablation(
    update_internal_temperature: bool,
    update_volume_and_pressure: bool,
    update_battery: bool,
    use_acs: bool
):
  def simulate_step_with_ablations(
      state: BalloonState,
      wind_vector: wind_field.WindVector,
      atmosphere: standard_atmosphere.Atmosphere,
      action: control.AltitudeControlCommand,
      stride: dt.timedelta,
  ) -> Dict[str, Any]:
    state_changes = {}
    state_changes['x'] = state.x + (wind_vector.u * stride)
    state_changes['y'] = state.y + (wind_vector.v * stride)
    
    rho_air = (state.pressure * constants.DRY_AIR_MOLAR_MASS) / (
        constants.UNIVERSAL_GAS_CONSTANT * state.ambient_temperature)

    drag = state.envelope_cod * state.envelope_volume**(2.0 / 3.0)

    total_flight_system_mass = (
        constants.HE_MOLAR_MASS * state.mols_lift_gas +
        constants.DRY_AIR_MOLAR_MASS * state.mols_air + state.envelope_mass +
        state.payload_mass)

    direction = (1.0 if rho_air * state.envelope_volume >=
                total_flight_system_mass else -1.0)
    dh_dt = direction * np.sqrt(  # [m/s]
        np.abs(2 * (rho_air * state.envelope_volume -
                    total_flight_system_mass) * constants.GRAVITY /
              (rho_air * drag)))
    
    dp = 1.0  # [Pa] A small pressure delta.
    height0 = atmosphere.at_pressure(state.pressure).height.meters
    height1 = atmosphere.at_pressure(state.pressure +
                                    direction * dp).height.meters
    dp_dh = direction * dp / (height1 - height0)
    dp_dt = dp_dh * dh_dt
    state_changes['pressure'] = state.pressure + dp_dt * stride.total_seconds()

    solar_elevation, _, solar_flux = solar.solar_calculator(
      state.latlng, state.date_time)  
    
    state_changes['ambient_temperature'] = atmosphere.at_pressure(
      state.pressure).temperature
    
    if update_internal_temperature:
      d_internal_temp = thermal.d_balloon_temperature_dt(
          state.envelope_volume, state.envelope_mass, state.internal_temperature,
          state.ambient_temperature, state.pressure, solar_elevation,
          solar_flux, state.upwelling_infrared)
      state_changes['internal_temperature'] = state.internal_temperature + d_internal_temp * stride.total_seconds()

    if update_volume_and_pressure:
      state_changes['envelope_volume'], state_changes['superpressure'] = (
          calculate_superpressure_and_volume(
              state.mols_lift_gas,
              state.mols_air,
              state.internal_temperature,
              state.pressure,
              state.envelope_volume_base,
              state.envelope_volume_dv_pressure))

      if state_changes['superpressure'] > state.envelope_max_superpressure:
        state_changes['status'] = BalloonStatus.BURST
      if state_changes['superpressure'] <= 0.0:
        state_changes['status'] = BalloonStatus.ZEROPRESSURE

    ## Step 5: Calculate, based on desired action, whether we'll use the
    # altitude control system (ACS) ⚙️. Adjust power usage accordingly.

    if not use_acs:
      if action == control.AltitudeControlCommand.UP:
        state_changes['acs_power'] = units.Power(watts=0.0)
        state_changes['acs_mass_flow'] = -0.012
      elif action == control.AltitudeControlCommand.DOWN:
        state_changes['acs_power'] = units.Power(watts=195.0)
        state_changes['acs_mass_flow'] = 0.007
      else:
        state_changes['acs_power'] = units.Power(watts=0.0)
        state_changes['acs_mass_flow'] = 0.0

    else:
      if action == control.AltitudeControlCommand.UP:
        state_changes['acs_power'] = units.Power(watts=0.0)
        valve_area = np.pi * state.acs_valve_hole_diameter.meters**2 / 4.0
        # Coefficient of drag on the air passing through the ACS from the
        # aperture. A measured quantity.
        default_valve_hole_cd = 0.62  # [.]
        gas_density = (
            state.superpressure +
            state.pressure) * constants.DRY_AIR_MOLAR_MASS / (
                constants.UNIVERSAL_GAS_CONSTANT * state.internal_temperature)
        state_changes['acs_mass_flow'] = (
            -1 * default_valve_hole_cd * valve_area * np.sqrt(
                2.0 * state.superpressure * gas_density))
      elif action == control.AltitudeControlCommand.DOWN:
        # Run the ACS compressor at a power level that maximizes mols of air
        # pushed into the ballonet per watt of energy at the current pressure
        # ratio (backpressure the compressor is pushing against).
        state_changes['acs_power'] = acs.get_most_efficient_power(
            state.pressure_ratio)
        # Compute mass flow rate by first computing efficiency of air flow.
        efficiency = acs.get_fan_efficiency(state.pressure_ratio,
                                            state_changes['acs_power'])
        state_changes['acs_mass_flow'] = acs.get_mass_flow(
            state_changes['acs_power'], efficiency)
      else:  # action == control.AltitudeControlCommand.STAY.
        state_changes['acs_power'] = units.Power(watts=0.0)
        state_changes['acs_mass_flow'] = 0.0

    state_changes['mols_air'] = state.mols_air + (
        state_changes['acs_mass_flow'] /
        constants.DRY_AIR_MOLAR_MASS) * stride.total_seconds()
    # mols_air must be positive.
    state_changes['mols_air'] = max(state_changes['mols_air'], 0.0)

    if update_battery:
      is_day = solar_elevation > solar.MIN_SOLAR_EL_DEG
      # # print("D (ble): ", is_day) 
      state_changes['solar_charging'] = (
          solar.solar_power(solar_elevation, state.pressure)
          if is_day else units.Power(watts=0.0))
      # TODO(scandido): Introduce a variable power load for cold upwelling IR?
      state_changes['power_load'] = (
          state.daytime_power_load if is_day else state.nighttime_power_load)
      state_changes['power_load'] += state_changes['acs_power']

      # We use a simplified model of a battery that is kept at a constant
      # temperature and acts like an ideal energy reservoir.
      state_changes['battery_charge'] = state.battery_charge + (
          state_changes['solar_charging'] - state_changes['power_load']) * stride
      # print("(ble) Q: ", state_changes['solar_charging'], state_changes['power_load'])
      state_changes['battery_charge'] = _clip(state_changes['battery_charge'],
                                              units.Energy(watt_hours=0.0),
                                              state.battery_capacity)

      if state_changes['battery_charge'].watt_hours <= 0.0:
        state_changes['status'] = BalloonStatus.OUT_OF_POWER

    # This must be updated in the inner loop, since the safety layer and
    # solar calculations rely on the current time.
    state_changes['date_time'] = state.date_time + stride
    state_changes['time_elapsed'] = state.time_elapsed + stride

    return state_changes

  return simulate_step_with_ablations
