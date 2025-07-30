import abc 
import jax
from balloon_learning_environment.env.balloon.standard_atmosphere import Atmosphere, JaxAtmosphere
from balloon_learning_environment.utils import constants
from jax.tree_util import register_pytree_node_class
from balloon_learning_environment.utils import jax_utils
import jax.numpy as jnp
from balloon_learning_environment.utils import units
from balloon_learning_environment.env.balloon.balloon import BalloonState
from functools import partial

class JaxBalloonStatus:
  OK = jnp.astype(0, jnp.int32)
  OUT_OF_POWER = jnp.astype(1, jnp.int32)
  BURST = jnp.astype(2, jnp.int32)
  ZEROPRESSURE = jnp.astype(3, jnp.int32)

class JaxBalloonState:
    
    def __str__(self):
        return (f"JaxBalloonState(center_latlng={self.center_latlng}, x={self.x}, y={self.y}, "
                f"pressure={self.pressure}, ambient_temperature={self.ambient_temperature}, "
                f"internal_temperature={self.internal_temperature}, envelope_volume={self.envelope_volume}, "
                f"superpressure={self.superpressure}, status={self.status}, acs_power={self.acs_power}, "
                f"acs_mass_flow={self.acs_mass_flow}, mols_air={self.mols_air}, solar_charging={self.solar_charging}, "
                f"power_load={self.power_load}, battery_charge={self.battery_charge}, date_time={self.date_time}, "
                f"time_elapsed={self.time_elapsed}, payload_mass={self.payload_mass}, envelope_mass={self.envelope_mass}, "
                f"envelope_max_superpressure={self.envelope_max_superpressure}, envelope_volume_base={self.envelope_volume_base}, "
                f"envelope_volume_dv_pressure={self.envelope_volume_dv_pressure}, envelope_cod={self.envelope_cod}, "
                f"daytime_power_load={self.daytime_power_load}, nighttime_power_load={self.nighttime_power_load}, "
                f"acs_valve_hole_diameter_meters={self.acs_valve_hole_diameter_meters}, battery_capacity={self.battery_capacity}, "
                f"upwelling_infrared={self.upwelling_infrared}, mols_lift_gas={self.mols_lift_gas})")
    
    def from_jax_state(state: 'JaxBalloonState'):
        copy = JaxBalloonState()
        copy.center_latlng = state.center_latlng
        copy.x = state.x
        copy.y = state.y
        copy.pressure = state.pressure
        copy.ambient_temperature = state.ambient_temperature
        copy.internal_temperature = state.internal_temperature
        copy.envelope_volume = state.envelope_volume
        copy.superpressure = state.superpressure
        copy.status = state.status
        copy.acs_power = state.acs_power
        copy.acs_mass_flow = state.acs_mass_flow
        copy.mols_air = state.mols_air
        copy.solar_charging = state.solar_charging
        copy.power_load = state.power_load
        copy.battery_charge = state.battery_charge
        copy.date_time = state.date_time
        copy.time_elapsed = state.time_elapsed
        copy.payload_mass = state.payload_mass
        copy.envelope_mass = state.envelope_mass
        copy.envelope_max_superpressure = state.envelope_max_superpressure
        copy.envelope_volume_base = state.envelope_volume_base
        copy.envelope_volume_dv_pressure = state.envelope_volume_dv_pressure
        copy.envelope_cod = state.envelope_cod
        copy.daytime_power_load = state.daytime_power_load
        copy.nighttime_power_load = state.nighttime_power_load
        copy.acs_valve_hole_diameter_meters = state.acs_valve_hole_diameter_meters
        copy.battery_capacity = state.battery_capacity
        copy.upwelling_infrared = state.upwelling_infrared
        copy.mols_lift_gas = state.mols_lift_gas
        return copy

    def from_ble_state(state: BalloonState):
        copy = JaxBalloonState()
        copy.center_latlng = jax_utils.JaxLatLng(
            jnp.astype(state.center_latlng.lat().radians, jnp.float64), 
            jnp.astype(state.center_latlng.lng().radians, jnp.float64)
        )
        copy.x = jnp.astype(state.x.meters, jnp.float64)
        copy.y = jnp.astype(state.y.meters, jnp.float64)
        copy.pressure = jnp.astype(state.pressure, jnp.float64)
        copy.ambient_temperature = jnp.astype(state.ambient_temperature, jnp.float64)
        copy.internal_temperature = jnp.astype(state.internal_temperature, jnp.float64)
        copy.envelope_volume = jnp.astype(state.envelope_volume, jnp.float64)
        copy.superpressure = jnp.astype(state.superpressure, jnp.float64)
        copy.status = jnp.astype(state.status.value, jnp.int32)
        copy.acs_power = jnp.astype(state.acs_power.watts, jnp.float64)
        copy.acs_mass_flow = jnp.astype(state.acs_mass_flow, jnp.float64)
        copy.mols_air = jnp.astype(state.mols_air, jnp.float64)
        copy.solar_charging = jnp.astype(state.solar_charging.watts, jnp.float64)
        copy.power_load = jnp.astype(state.power_load.watts, jnp.float64)
        copy.battery_charge = jnp.astype(state.battery_charge.watt_hours, jnp.float64)
        copy.date_time = jnp.astype(state.date_time.timestamp(), jnp.float64)
        copy.time_elapsed = jnp.astype(state.time_elapsed.total_seconds(), jnp.float64)
        copy.payload_mass = jnp.astype(state.payload_mass, jnp.float64)
        copy.envelope_mass = jnp.astype(state.envelope_mass, jnp.float64)
        copy.envelope_max_superpressure = jnp.astype(state.envelope_max_superpressure, jnp.float64)
        copy.envelope_volume_base = jnp.astype(state.envelope_volume_base, jnp.float64)
        copy.envelope_volume_dv_pressure = jnp.astype(state.envelope_volume_dv_pressure, jnp.float64)
        copy.envelope_cod = jnp.astype(state.envelope_cod, jnp.float64)
        copy.daytime_power_load = jnp.astype(state.daytime_power_load.watts, jnp.float64)
        copy.nighttime_power_load = jnp.astype(state.nighttime_power_load.watts, jnp.float64)
        copy.acs_valve_hole_diameter_meters = jnp.astype(state.acs_valve_hole_diameter.meters, jnp.float64)
        copy.battery_capacity = jnp.astype(state.battery_capacity.watt_hours, jnp.float64)
        copy.upwelling_infrared = jnp.astype(state.upwelling_infrared, jnp.float64)
        copy.mols_lift_gas = jnp.astype(state.mols_lift_gas, jnp.float64)
        return copy

    def get_jax_features(self):
        """ Return JaxBalloonState instance variables that change in balloon._simulate_step_continuous_internal(and also station center) as an numpy array. 18 total variables.

        [center_lat, center_lon, x, y, pressure, ambient_temperature, internal_temperature, envelope_volume, superpressure, status, acs_power, acs_mass_flow, mols_air, solar_charging, power_load, battery_charge, date_time, time_elapsed]
        """        
        return jnp.array([
            self.center_latlng.lat.reshape(()),
            self.center_latlng.lng.reshape(()),
            self.x.reshape(()),
            self.y.reshape(()),
            self.pressure.reshape(()),
            self.ambient_temperature.reshape(()),
            self.internal_temperature.reshape(()),
            self.envelope_volume.reshape(()),
            self.superpressure.reshape(()),
            self.status.reshape(()),
            self.acs_power.reshape(()),
            self.acs_mass_flow.reshape(()),
            self.mols_air.reshape(()),
            self.solar_charging.reshape(()),
            self.power_load.reshape(()),
            self.battery_charge.reshape(()),
            self.date_time.reshape(()),
            self.time_elapsed.reshape(())
        ])


    def __init__(self):
        self.center_latlng = jax_utils.JaxLatLng(
            jnp.asarray(0.0, jnp.float64), 
            jnp.asarray(0.0, jnp.float64)) # doesn't change but x,y relative to this

        # State variables:
        self.x = jnp.astype(0.0, jnp.float64)  # meters
        self.y = jnp.astype(0.0, jnp.float64)  # meters
        self.pressure = jnp.astype(6000.0, jnp.float64)  # Pascals
        
        self.ambient_temperature = jnp.astype(206.0, jnp.float64)  # K
        self.internal_temperature = jnp.astype(206.0, jnp.float64)  # K   

        self.envelope_volume = jnp.astype(1804.0, jnp.float64)  # kg/s    
        self.superpressure = jnp.astype(0.0, jnp.float64)  # Pascals
        self.status = jnp.astype(0, jnp.int32)  # OK

        self.acs_power = jnp.astype(0.0, jnp.float64)  # Watts
        self.acs_mass_flow = jnp.astype(0.0, jnp.float64)  # kg/s
        self.mols_air = jnp.astype(0.0, jnp.float64)  # [mols]

        self.solar_charging = jnp.astype(0.0, jnp.float64)  # Watts
        self.power_load = jnp.astype(0.0, jnp.float64)  # Watts

        self.battery_charge = jnp.astype(2905.6, jnp.float64)  # Watt Hours (95% initial capacity)
        self.date_time = jnp.astype(0.0, jnp.float64)  # seconds
        self.time_elapsed = jnp.astype(0.0, jnp.float64)  # seconds

        # Balloon constants:
        self.payload_mass = jnp.astype(92.5, jnp.float64)  # kg        
        self.envelope_mass = jnp.astype(68.5, jnp.float64)  # kg
        self.envelope_max_superpressure = jnp.astype(2380, jnp.float64)  # Pa
        self.envelope_volume_base = jnp.astype(1804, jnp.float64)  # [m^3]
        self.envelope_volume_dv_pressure = jnp.astype(0.0199, jnp.float64)  # [m^3/Pa]
        self.envelope_cod = jnp.astype(0.25, jnp.float64)
        self.daytime_power_load = jnp.astype(183.7, jnp.float64)  # Watts
        self.nighttime_power_load = jnp.astype(120.4, jnp.float64)  # Watts

        self.acs_valve_hole_diameter_meters = jnp.astype(0.04, jnp.float64)  # m
        self.battery_capacity = jnp.astype(3058.56, jnp.float64)  # Watt Hours
        
        self.upwelling_infrared = jnp.astype(250.0, jnp.float64)  # W/m^2
        self.mols_lift_gas = jnp.astype(6830.0, jnp.float64)  # [mols]
        
    def tree_flatten(self): 
        children = (self.center_latlng, 
                    self.x, 
                    self.y, 
                    self.pressure, 
                    self.ambient_temperature, 
                    self.internal_temperature, 
                    self.envelope_volume, 
                    self.superpressure, 
                    self.status, 
                    self.acs_power, 
                    self.acs_mass_flow, 
                    self.mols_air, 
                    self.solar_charging,
                    self.power_load, 
                    self.battery_charge, 
                    self.date_time, 
                    self.time_elapsed,) 
        
        aux_data = {"payload_mass": self.payload_mass, 
                    "envelope_mass": self.envelope_mass, 
                    "envelope_max_superpressure": self.envelope_max_superpressure, 
                    "envelope_volume_base": self.envelope_volume_base, 
                    "envelope_volume_dv_pressure": self.envelope_volume_dv_pressure,
                    "envelope_cod": self.envelope_cod,
                    "daytime_power_load": self.daytime_power_load,
                    "nighttime_power_load": self.nighttime_power_load,
                    "acs_valve_hole_diameter_meters": self.acs_valve_hole_diameter_meters,
                    "battery_capacity": self.battery_capacity,
                    "upwelling_infrared": self.upwelling_infrared,
                    "mols_lift_gas": self.mols_lift_gas}
        
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children): 
        jax_balloon = JaxBalloonState()
        jax_balloon.center_latlng = children[0]
        jax_balloon.x = children[1]
        jax_balloon.y = children[2]
        jax_balloon.pressure = children[3]
        jax_balloon.ambient_temperature = children[4]
        jax_balloon.internal_temperature = children[5]
        jax_balloon.envelope_volume = children[6]
        jax_balloon.superpressure = children[7]
        jax_balloon.status = children[8]
        jax_balloon.acs_power = children[9]
        jax_balloon.acs_mass_flow = children[10]
        jax_balloon.mols_air = children[11]
        jax_balloon.solar_charging = children[12]
        jax_balloon.power_load = children[13]
        jax_balloon.battery_charge = children[14]
        jax_balloon.date_time = children[15]
        jax_balloon.time_elapsed = children[16]

        jax_balloon.payload_mass = aux_data["payload_mass"]
        jax_balloon.envelope_mass = aux_data["envelope_mass"]
        jax_balloon.envelope_max_superpressure = aux_data["envelope_max_superpressure"]
        jax_balloon.envelope_volume_base = aux_data["envelope_volume_base"]
        jax_balloon.envelope_volume_dv_pressure = aux_data["envelope_volume_dv_pressure"]
        jax_balloon.envelope_cod = aux_data["envelope_cod"]
        jax_balloon.daytime_power_load = aux_data["daytime_power_load"]
        jax_balloon.nighttime_power_load = aux_data["nighttime_power_load"]
        jax_balloon.acs_valve_hole_diameter_meters = aux_data["acs_valve_hole_diameter_meters"]
        jax_balloon.battery_capacity = aux_data["battery_capacity"]
        jax_balloon.upwelling_infrared = aux_data["upwelling_infrared"]
        jax_balloon.mols_lift_gas = aux_data["mols_lift_gas"]

        return jax_balloon

register_pytree_node_class(JaxBalloonState)


class JaxBalloon:

    def __init__(self, state: JaxBalloonState):
        self.state = state

    @partial(jax.jit, static_argnames=("time_delta", "stride"))
    def simulate_step(
            self, 
            wind_vector: '[u, v], meters / second', 
            atmosphere: JaxAtmosphere, 
            action: '0/1/2, down/stay/up', 
            time_delta: 'int, seconds', 
            stride: 'int, seconds') -> 'next_balloon':
    
        # time_delta % stride == 0 must be true!
        # check if self.state.status == OK
        # check safety layers

        # TODO: could make the seconds take in integers
        outer_stride = time_delta
        inner_stride = stride
        def update_step(i, balloon):
            return balloon._simulate_step_internal(wind_vector, atmosphere, action, stride)

        final_balloon = jax.lax.fori_loop(0, outer_stride//inner_stride, update_step, init_val=self)
        return final_balloon
    
    @partial(jax.jit, static_argnames=("time_delta", "stride")) 
    def simulate_step_continuous(self, wind_vector, atmosphere, acs_control, time_delta, stride):
        return self.simulate_step_continuous_no_jit(wind_vector, atmosphere, acs_control, time_delta, stride)
    
    def simulate_step_continuous_no_jit(
            self, 
            wind_vector: '[u, v], meters / second', 
            atmosphere: JaxAtmosphere, 
            acs_control: 'float, [-1, 1]', 
            time_delta: 'int, seconds', 
            stride: 'int, seconds') -> 'next_balloon':
        # time_delta % stride == 0 must be true!
        # check if self.state.status == OK
        # check safety layers

        def update_step(balloon, i):
            return balloon._simulate_step_continuous_internal(wind_vector, atmosphere, acs_control, stride), None

        num_steps = time_delta // stride
        stride *= 1.0
        # jax.debug.print("num_steps={x}, time_delta={y}, stride={z}", x=num_steps, y=time_delta, z=stride)
        final_balloon, _ = jax.lax.scan(update_step, init=self, xs=jnp.arange(num_steps))

        return final_balloon


    def _simulate_step_continuous_internal(
            self, 
            wind_vector, 
            atmosphere, 
            acs_control,
            stride) -> 'd_state':
        
        state = self.state
        new_state = JaxBalloonState.from_jax_state(state)
        
        # Step 1: balloon moves with the wind
        new_state.x = state.x + wind_vector[0] * stride
        new_state.y = state.y + wind_vector[1] * stride

        # Step 2: calculate pressure from bouyancy and drag forces
        rho_air = (state.pressure * jax_utils.DRY_AIR_MOLAR_MASS) / (
            jax_utils.UNIVERSAL_GAS_CONSTANT * state.ambient_temperature)
        
        drag = state.envelope_cod * state.envelope_volume**(2.0 / 3.0)
        total_flight_system_mass = (
            jax_utils.HE_MOLAR_MASS * state.mols_lift_gas +
            jax_utils.DRY_AIR_MOLAR_MASS * state.mols_air + state.envelope_mass +
        state.payload_mass)
        
        direction = jax.lax.cond(
            rho_air * state.envelope_volume >= total_flight_system_mass,
            lambda _: 1.0,
            lambda _: -1.0,
            operand=None,
        )
        
        # direction = (1.0 if rho_air * self.envelope_volume >=
        #     total_flight_system_mass else -1.0)
        
        dh_dt = direction * jnp.sqrt(  # [m/s]
            jnp.abs(2 * (rho_air * state.envelope_volume -
                        total_flight_system_mass) * jax_utils.GRAVITY /
                    (rho_air * drag)))
        # print("dh_dt: ", dh_dt)

        dp = 1.0  # [Pa] A small pressure delta.
        height0 = atmosphere.at_pressure(state.pressure).height.meters
        height1 = atmosphere.at_pressure(state.pressure +
                                        direction * dp).height.meters
        dp_dh = direction * dp / (height1 - height0)
        # print("dp_dh: ", dp_dh)
        dp_dt = dp_dh * dh_dt
        # print("P", dh_dt, stride)
        new_state.pressure = state.pressure + dp_dt * stride
        
        # Step 3: calculate internal temp of balloon

        latlng = jax_utils.calculate_jax_latlng_from_offset(state.center_latlng, state.x/1000, state.y/1000)
        # print("C(j): ", latlng)
        solar_elevation, _, solar_flux = jax_utils.solar_calculator(latlng, state.date_time)
        # print("solar_elevation", solar_elevation)

        new_state.ambient_temperature = jnp.astype(atmosphere.at_pressure(state.pressure).temperature, jnp.float64)

        d_internal_temperature = jax_utils.d_balloon_temperature_dt(
            state.envelope_volume, state.envelope_mass, state.internal_temperature,
            state.ambient_temperature, state.pressure, solar_elevation,
            solar_flux, state.upwelling_infrared)
        new_state.internal_temperature = state.internal_temperature + d_internal_temperature * stride

        ## Step 4: Calculate superpressure and volume of the balloon 🎈.
        new_state.envelope_volume, new_state.superpressure = jax_utils.calculate_superpressure_and_volume(
                state.mols_lift_gas,
                state.mols_air,
                state.internal_temperature,
                state.pressure,
                state.envelope_volume_base,
                state.envelope_volume_dv_pressure
            )
        
        new_state.status = jax.lax.cond(
            new_state.superpressure > state.envelope_max_superpressure,
            lambda _: JaxBalloonStatus.BURST,
            lambda op: op,
            operand=state.status,
        )

        new_state.status = jax.lax.cond(
            new_state.superpressure <= 0.0,
            lambda _: JaxBalloonStatus.ZEROPRESSURE,
            lambda op: op,
            operand=state.status,
        )

        ## Step 5: Calculate, based on desired action, whether we'll use the
        # altitude control system (ACS) ⚙️. Adjust power usage accordingly.

        def on_action_up(state: JaxBalloonState, acs_control: float):
            # jax.debug.print("jax balloon action up")
            state_acs_power = 0.0 # watts
            valve_area = acs_control * jnp.pi * state.acs_valve_hole_diameter_meters**2 / 4.0
            # Coefficient of drag on the air passing through the ACS from the
            # aperture. A measured quantity.
            default_valve_hole_cd = 0.62  # [.]
            gas_density = (
                state.superpressure +
                state.pressure) * jax_utils.DRY_AIR_MOLAR_MASS / (
                    jax_utils.UNIVERSAL_GAS_CONSTANT * state.internal_temperature)
            state_acs_mass_flow= (
                -1 * default_valve_hole_cd * valve_area * jnp.sqrt(
                    2.0 * state.superpressure * gas_density))
            
            return state_acs_power, state_acs_mass_flow

        def on_action_down(state: JaxBalloonState, acs_control: float):
            # jax.debug.print("jax balloon action down")
            superpressure = jnp.max(jnp.array([state.superpressure, 0.0]))
            pressure_ratio = (state.pressure + superpressure) / state.pressure
            
            state_acs_power = (-acs_control) * jax_utils.get_most_efficient_power(pressure_ratio)
            # Compute mass flow rate by first computing efficiency of air flow.
            efficiency = jax_utils.get_fan_efficiency(pressure_ratio,
                                                state_acs_power)
            state_acs_mass_flow = jax_utils.get_mass_flow(
                state_acs_power, efficiency)
            
            return state_acs_power, state_acs_mass_flow

        def on_action_stay():
            # jax.debug.print("jax balloon action stay")
            state_acs_power = 0.0
            state_acs_mass_flow = 0.0
            
            return state_acs_power, state_acs_mass_flow
        

        acs_power, acs_mass_flow = jax.lax.cond(
            acs_control < 0.0,
            lambda op: on_action_down(*op),
            lambda op: jax.lax.cond(
                op[1] > 0.0,
                lambda op1: on_action_up(*op1),
                lambda _: on_action_stay(),
                operand=op,
            ),
            operand=(state, acs_control),
        )
        acs_power = jnp.astype(acs_power, jnp.float64)
        acs_mass_flow = jnp.astype(acs_mass_flow, jnp.float64)

        new_state.acs_power, new_state.acs_mass_flow = acs_power, acs_mass_flow
        
        new_state.mols_air= state.mols_air + (
            new_state.acs_mass_flow /
            jax_utils.DRY_AIR_MOLAR_MASS) * stride
        new_state.mols_air = jnp.max(jnp.array([new_state.mols_air, jnp.astype(0.0, jnp.float64)]))

        ## Step 6: Calculate energy usage and collection, and move coulombs onto
        # and off of the battery as apppropriate. 🔋

        is_day = solar_elevation > jax_utils.MIN_SOLAR_EL_DEG
        new_state.solar_charging = jax.lax.cond(
            is_day,
            lambda op: jax_utils.solar_power(solar_elevation, op),
            lambda _: jnp.astype(0.0, jnp.float64),
            operand=state.pressure,
        )

        # TODO(scandido): Introduce a variable power load for cold upwelling IR?
        new_state.power_load = jnp.astype(jax.lax.cond(
            is_day,
            lambda op: op[0],
            lambda op: op[1],
            operand=[state.daytime_power_load, state.nighttime_power_load],
        ), jnp.float64)


        new_state.power_load += new_state.acs_power

        # We use a simplified model of a battery that is kept at a constant
        # temperature and acts like an ideal energy reservoir.
        new_state.battery_charge = state.battery_charge + (
            new_state.solar_charging - new_state.power_load) * (stride/jax_utils.NUM_SECONDS_PER_HOUR)

        # print("Q: ", new_state.solar_charging, new_state.power_load)

        # energer watts hr
        new_state.battery_charge = jnp.clip(new_state.battery_charge,
                                                0.0,
                                                state.battery_capacity)

        new_state.status = jax.lax.cond(new_state.battery_charge <= 0.0, 
                                    lambda _: JaxBalloonStatus.OUT_OF_POWER,
                                    lambda op: op,
                                    operand=state.status,
                                    )
        
        # This must be updated in the inner loop, since the safety layer and
        # solar calculations rely on the current time.
        new_state.date_time = state.date_time + stride
        new_state.time_elapsed = state.time_elapsed + stride

        return JaxBalloon(new_state)

    # @jax.jit
    def _simulate_step_internal(
            self, 
            wind_vector, 
            atmosphere, 
            action,
            stride) -> 'd_state':
        
        state = self.state
        new_state = JaxBalloonState.from_jax_state(state)
        
        # Step 1: balloon moves with the wind
        new_state.x = state.x + wind_vector[0] * stride
        new_state.y = state.y + wind_vector[1] * stride

        # Step 2: calculate pressure from bouyancy and drag forces
        rho_air = (state.pressure * jax_utils.DRY_AIR_MOLAR_MASS) / (
            jax_utils.UNIVERSAL_GAS_CONSTANT * state.ambient_temperature)
        
        drag = state.envelope_cod * state.envelope_volume**(2.0 / 3.0)
        total_flight_system_mass = (
            jax_utils.HE_MOLAR_MASS * state.mols_lift_gas +
            jax_utils.DRY_AIR_MOLAR_MASS * state.mols_air + state.envelope_mass +
        state.payload_mass)
        
        direction = jax.lax.cond(
            rho_air * state.envelope_volume >= total_flight_system_mass,
            lambda _: 1.0,
            lambda _: -1.0,
            operand=None,
        )
        
        # direction = (1.0 if rho_air * self.envelope_volume >=
        #     total_flight_system_mass else -1.0)
        
        dh_dt = direction * jnp.sqrt(  # [m/s]
            jnp.abs(2 * (rho_air * state.envelope_volume -
                        total_flight_system_mass) * jax_utils.GRAVITY /
                    (rho_air * drag)))
        # print("dh_dt: ", dh_dt)

        dp = 1.0  # [Pa] A small pressure delta.
        height0 = atmosphere.at_pressure(state.pressure).height.meters
        height1 = atmosphere.at_pressure(state.pressure +
                                        direction * dp).height.meters
        dp_dh = direction * dp / (height1 - height0)
        # print("dp_dh: ", dp_dh)
        dp_dt = dp_dh * dh_dt
        # print("P", dh_dt, stride)
        new_state.pressure = state.pressure + dp_dt * stride
        
        # Step 3: calculate internal temp of balloon

        latlng = jax_utils.calculate_jax_latlng_from_offset(state.center_latlng, state.x/1000, state.y/1000)
        # print("C(j): ", latlng)
        solar_elevation, _, solar_flux = jax_utils.solar_calculator(latlng, state.date_time)
        # print("solar_elevation", solar_elevation)

        new_state.ambient_temperature = atmosphere.at_pressure(state.pressure).temperature
        d_internal_temperature = jax_utils.d_balloon_temperature_dt(
            state.envelope_volume, state.envelope_mass, state.internal_temperature,
            state.ambient_temperature, state.pressure, solar_elevation,
            solar_flux, state.upwelling_infrared)
        new_state.internal_temperature = state.internal_temperature + d_internal_temperature * stride

        ## Step 4: Calculate superpressure and volume of the balloon 🎈.
        new_state.envelope_volume, new_state.superpressure = jax_utils.calculate_superpressure_and_volume(
                state.mols_lift_gas,
                state.mols_air,
                state.internal_temperature,
                state.pressure,
                state.envelope_volume_base,
                state.envelope_volume_dv_pressure
            )
        
        new_state.status = jax.lax.cond(
            new_state.superpressure > state.envelope_max_superpressure,
            lambda _: JaxBalloonStatus.BURST,
            lambda op: op,
            operand=state.status,
        )

        new_state.status = jax.lax.cond(
            new_state.superpressure <= 0.0,
            lambda _: JaxBalloonStatus.ZEROPRESSURE,
            lambda op: op,
            operand=state.status,
        )

        ## Step 5: Calculate, based on desired action, whether we'll use the
        # altitude control system (ACS) ⚙️. Adjust power usage accordingly.

        def on_action_up(state: JaxBalloonState):
            state_acs_power = 0.0 # watts
            valve_area = jnp.pi * state.acs_valve_hole_diameter_meters**2 / 4.0
            # Coefficient of drag on the air passing through the ACS from the
            # aperture. A measured quantity.
            default_valve_hole_cd = 0.62  # [.]
            gas_density = (
                state.superpressure +
                state.pressure) * jax_utils.DRY_AIR_MOLAR_MASS / (
                    jax_utils.UNIVERSAL_GAS_CONSTANT * state.internal_temperature)
            state_acs_mass_flow= (
                -1 * default_valve_hole_cd * valve_area * jnp.sqrt(
                    2.0 * state.superpressure * gas_density))
            
            return state_acs_power, state_acs_mass_flow

        def on_action_down(state: JaxBalloonState):
            superpressure = jnp.max(jnp.array([state.superpressure, 0.0]))
            pressure_ratio = (state.pressure + superpressure) / state.pressure
            
            state_acs_power = jax_utils.get_most_efficient_power(pressure_ratio)
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
        
        new_state.acs_power, new_state.acs_mass_flow = jax.lax.cond(
            action == 0,
            lambda op: on_action_down(op),
            lambda op: jax.lax.cond(
                action == 1,
                lambda _:on_action_stay(),
                lambda op1: on_action_up(op1),
                operand=op,
            ),
            operand=state,
        )
        
        new_state.mols_air= state.mols_air + (
            new_state.acs_mass_flow /
            jax_utils.DRY_AIR_MOLAR_MASS) * stride
        new_state.mols_air = jnp.max(jnp.array([new_state.mols_air, 0.0]))
        
        ## Step 6: Calculate energy usage and collection, and move coulombs onto
        # and off of the battery as apppropriate. 🔋

        is_day = solar_elevation > jax_utils.MIN_SOLAR_EL_DEG
        new_state.solar_charging = jax.lax.cond(
            is_day,
            lambda op: jax_utils.solar_power(solar_elevation, op),
            lambda _: jnp.astype(0.0, jnp.float64),
            operand=state.pressure,
        )

        # TODO(scandido): Introduce a variable power load for cold upwelling IR?
        new_state.power_load = jax.lax.cond(
            is_day,
            lambda op: op[0],
            lambda op: op[1],
            operand=[state.daytime_power_load, state.nighttime_power_load],
        )

        new_state.power_load += new_state.acs_power

        # We use a simplified model of a battery that is kept at a constant
        # temperature and acts like an ideal energy reservoir.
        new_state.battery_charge = state.battery_charge + (
            new_state.solar_charging - new_state.power_load) * (stride/jax_utils.NUM_SECONDS_PER_HOUR)
        
        # print("Q: ", new_state.solar_charging, new_state.power_load)

        # energer watts hr
        new_state.battery_charge = jnp.clip(new_state.battery_charge,
                                                jnp.astype(0.0, jnp.float64),
                                                state.battery_capacity)

        new_state.status = jax.lax.cond(new_state.battery_charge <= 0.0, 
                                    lambda _: JaxBalloonStatus.OUT_OF_POWER,
                                    lambda op: op,
                                    operand=state.status,
                                    )

        # This must be updated in the inner loop, since the safety layer and
        # solar calculations rely on the current time.

        new_state.date_time = state.date_time + stride
        new_state.time_elapsed = state.time_elapsed + stride

        return JaxBalloon(new_state)
        
    def tree_flatten(self): 
        return (self.state, ), {}

    @classmethod
    def tree_unflatten(cls, aux_data, children): 
        return JaxBalloon(children[0])

register_pytree_node_class(JaxBalloon)