import abc
from balloon_learning_environment.env.balloon.standard_atmosphere import Atmosphere, JaxAtmosphere
from balloon_learning_environment.utils import constants
from jax.tree_util import register_pytree_node_class
from balloon_learning_environment.utils import jax_utils
import jax.numpy as jnp



# DOWN = 0
# STAY = 1
# UP = 2

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

    def _simulate_step_internal(
            self, 
            wind_vector, 
            atmosphere, 
            action,
            stride) -> 'd_state':
        
        # Step 1: balloon moves with the wind
        state_x_m = self.x_meters + wind_vector[0] * stride
        state_y_m = self.y_meters + wind_vector[1] * stride

        # Step 2: calculate pressure from bouyancy and drag forces
        rho_air = (self.pressure * constants.DRY_AIR_MOLAR_MASS) / (
            constants.UNIVERSAL_GAS_CONSTANT * self.ambient_temperature)
        
        drag = self.envolve_cod
        total_flight_system_mass = (
            constants.HE_MOLAR_MASS * self.mols_lift_gas +
            constants.DRY_AIR_MOLAR_MASS * self.mols_air + self.envelope_mass +
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

        self.envelope_volume, self.superpressure = jax_utils.calculate_super_pressure_and_volume(
                self.mols_lift_gas,
                self.mols_air,
                self.internal_temperature,
                self.pressure,
                self.envelope_volume_base,
                self.envelope_volume_dv_pressure
            )
        
        # if state_changes['superpressure'] > state.envelope_max_superpressure:
        #     state_changes['status'] = BalloonStatus.BURST
        # if state_changes['superpressure'] <= 0.0:
        #     state_changes['status'] = BalloonStatus.ZEROPRESSURE


        
        
    def tree_flatten(self): 
        pass

    @classmethod
    def tree_unflatten(cls, aux_data, children): 
        pass

register_pytree_node_class(JaxBalloon)


class MPC2Agent(agent.Agent):
    pass