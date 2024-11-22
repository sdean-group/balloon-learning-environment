import abc
import jax
import enum
from balloon_learning_environment.env.balloon.standard_atmosphere import Atmosphere, JaxAtmosphere
from balloon_learning_environment.utils import constants
from jax.tree_util import register_pytree_node_class
from balloon_learning_environment.utils import jax_utils
import jax.numpy as jnp



# DOWN = 0
# STAY = 1
# UP = 2

class JaxBalloonStatus(enum.Enum):
  OK = 0
  OUT_OF_POWER = 1
  BURST = 2
  ZEROPRESSURE = 3

class JaxBalloon:

    def __init__(self):
        # TODO: figure out which states actually update and which states mostly function as constants

        self.x_meters = 0.0
        self.y_meters = 0.0

        # ...

        pass

    def simulate_step(
            self, 
            wind_vector: '[u, v], meters / second', 
            atmosphere: JaxAtmosphere, 
            action: '0/1/2, down/stay/up', 
            time_delta: 'float, seconds', 
            stride: 'float, seconds') -> 'next_balloon':
    
        # time_delta % stride == 0 must be true!

        # todo: double for loop
        pass

    @property
    def latlng(self) -> jax_utils.JaxLatLng:
        return jax_utils.spherical_geometry.calculate_latlng_from_offset(
            self.center_latlng, self.x_m, self.y_m)


    def _simulate_step_internal(
            self, 
            wind_vector, 
            atmosphere, 
            action,
            stride) -> 'd_state':
        
        # Step 1: balloon moves with the wind
        state_x_m = self.x_m + wind_vector[0] * stride
        state_y_m = self.y_m + wind_vector[1] * stride

        # Step 2: calculate pressure from bouyancy and drag forces
        rho_air = (self.pressure * constants.DRY_AIR_MOLAR_MASS) / (
            constants.UNIVERSAL_GAS_CONSTANT * self.ambient_temperature)
        
        drag = self.envolve_cod
        total_flight_system_mass = (
            jax_utils.HE_MOLAR_MASS * self.mols_lift_gas +
            jax_utils.DRY_AIR_MOLAR_MASS * self.mols_air + self.envelope_mass +
        self.payload_mass)
        direction = (1.0 if rho_air * self.envelope_volume >=
            total_flight_system_mass else -1.0)
        
        dh_dt = direction * jnp.sqrt(  # [m/s]
            jnp.abs(2 * (rho_air * self.envelope_volume -
                        total_flight_system_mass) * constants.GRAVITY /
                    (rho_air * drag)))
        dp = 1.0  # [Pa] A small pressure delta.
        height0 = atmosphere.at_pressure(self.pressure).height.meters
        height1 = atmosphere.at_pressure(self.pressure +
                                        direction * dp).height.meters
        dp_dh = direction * dp / (height1 - height0)
        dp_dt = dp_dh * dh_dt

        state_pressure = self.pressure + dp_dt * stride
        
        # Step 3: calculate internal temp of balloon
        latlng = jax_utils.calculate_jax_latlng_from_offset(self.center_latlng, self.x_meters, self.y_meters)
        solar_elevation, _, solar_flux = jax_utils.solar_calculator(self.latlng, self.date_time)

        ambient_temperature = atmosphere.at_pressure(self.pressure).temperature
        d_internal_temp = jax_utils.d_balloon_temperature_dt(
            self.envelope_volume, self.envelope_mass, self.internal_temperature,
            ambient_temperature, self.pressure, solar_elevation,
            solar_flux, self.upwelling_infrared)

        ## Step 4: Calculate superpressure and volume of the balloon 🎈.
        state_envelope_volume, state_superpressure = jax_utils.calculate_superpressure_and_volume(
                self.mols_lift_gas,
                self.mols_air,
                self.internal_temperature,
                self.pressure,
                self.envelope_volume_base,
                self.envelope_volume_dv_pressure
            )
        
        state_status = jax.lax.cond(
            self.superpressure > self.envelope_max_superpressure,
            lambda _: JaxBalloonStatus.BURST,
            lambda _: self.status,
            operand=None,
        )

        state_status = jax.lax.cond(
            self.superpressure <= 0.0,
            lambda _: JaxBalloonStatus.ZEROPRESSURE,
            lambda _: self.status,
            operand=None,
        )

        ## Step 5: Calculate, based on desired action, whether we'll use the
        # altitude control system (ACS) ⚙️. Adjust power usage accordingly.

        def on_action_up(self):
            state_acs_power = 0.0 # watts
            valve_area = jnp.pi * self.acs_valve_hole_diameter_meters**2 / 4.0
            # Coefficient of drag on the air passing through the ACS from the
            # aperture. A measured quantity.
            default_valve_hole_cd = 0.62  # [.]
            gas_density = (
                self.superpressure +
                self.pressure) * jax_utils.DRY_AIR_MOLAR_MASS / (
                    jax_utils.UNIVERSAL_GAS_CONSTANT * self.internal_temperature)
            state_acs_mass_flow= (
                -1 * default_valve_hole_cd * valve_area * jnp.sqrt(
                    2.0 * self.superpressure * gas_density))
            
            return state_acs_power, state_acs_mass_flow

        def on_action_down(self):
            superpressure = jnp.max(jnp.array([self.superpressure, 0.0]))
            pressure_ratio = (self.pressure + superpressure) / self.pressure
            state_acs_power = jax_utils.get_most_efficient_power(
            pressure_ratio)
            # Compute mass flow rate by first computing efficiency of air flow.
            efficiency = jax_utils.get_fan_efficiency(pressure_ratio,
                                                state_acs_power)
            state_acs_mass_flow = jax_utils.get_mass_flow(
                state_acs_power, efficiency)
            
            return state_acs_power, state_acs_mass_flow

        def on_action_stay():
            state_acs_power = 0.0
            state_acs_mass_flow = 0.0
            
            return state_acs_power, state_acs_mass_flow
        
        state_acs_power, state_acs_mass_flow = jax.lax.cond(
            action == 0,
            lambda op: on_action_down(op[0]),
            lambda op: jax.lax.cond(
                action == 1,
                lambda _:on_action_stay(),
                lambda op1: on_action_up(op1[0]),
                operand=op,
            ),
            operand=self,
        )
        
        state_mols_air= self.mols_air + (
            state_acs_mass_flow /
            jax_utils.DRY_AIR_MOLAR_MASS) * stride
        state_mols_air = jnp.max(jnp.array([state_mols_air, 0.0]))
        
        ## Step 6: Calculate energy usage and collection, and move coulombs onto
        # and off of the battery as apppropriate. 🔋

        is_day = solar_elevation > jax_utils.MIN_SOLAR_EL_DEG
        state_solar_charging = jax.lax.cond(
            is_day,
            lambda op: jax_utils.solar_power(solar_elevation, op.pressure),
            lambda _: 0.0,
            operand=self,
        )

        # TODO(scandido): Introduce a variable power load for cold upwelling IR?
        state_power_load = jax.lax.cond(
            is_day,
            lambda op: op.daytime_power_load,
            lambda op: op.nighttime_power_load,
            operand=self,
        )

        state_power_load += state_acs_power

        # We use a simplified model of a battery that is kept at a constant
        # temperature and acts like an ideal energy reservoir.
        state_battery_charge = state_battery_charge + (
            state_solar_charging - state_power_load) * stride
        
        # energer watts hr
        state_battery_charge = jnp.clip(state_battery_charge,
                                                0.0,
                                                self.battery_capacity)

        state_status = jax.lax.cond(state_battery_charge <= 0.0, 
                                    lambda _: JaxBalloonStatus.OUT_OF_POWER,
                                    lambda op: op.status,
                                    operand=self,
                                    )

        # This must be updated in the inner loop, since the safety layer and
        # solar calculations rely on the current time.
        state_date_time = self.date_time + stride
        state_time_elapsed = self.time_elapsed + stride

        # return JaxBalloon(state_x_m, state_y_m, )

        
        
    def tree_flatten(self): 
        pass

    @classmethod
    def tree_unflatten(cls, aux_data, children): 
        pass

register_pytree_node_class(JaxBalloon)


class MPC2Agent(agent.Agent):
    pass