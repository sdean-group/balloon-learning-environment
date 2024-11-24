"""Compare wind measurements and dynamics model"""
from balloon_learning_environment.env.balloon import balloon
from balloon_learning_environment.env.balloon import control
from balloon_learning_environment.agents.mpc2_agent import JaxBalloon, JaxBalloonState
# from balloon_learning_environment.env.balloon_arena import balloon_arena
from balloon_learning_environment.env.generative_wind_field import generative_wind_field_factory
from balloon_learning_environment.agents.mpc_agent import *
from balloon_learning_environment.env import features
from balloon_learning_environment.env.balloon import stable_init
from balloon_learning_environment.env.balloon import standard_atmosphere

from balloon_learning_environment.utils import units
from balloon_learning_environment.utils import sampling
from atmosnav.utils import alt2p as alt2p_atm
from atmosnav.utils import p2alt
from atmosnav import *
import jax
import time
import math

import jax.numpy as jnp
import datetime as dt

# key = jax.random.PRNGKey(seed=0)


# just use a fixed key for the start time and the atmosphere 
start_date_time = sampling.sample_time(jax.random.PRNGKey(seed=0))
atmosphere = standard_atmosphere.Atmosphere(jax.random.PRNGKey(seed=0))

def initialize_balloon(key=jax.random.PRNGKey(seed=0)):
    alpha = 1.2
    beta = 2.0
    
    # global key
    key, *keys = jax.random.split(key, num=6)

    # Note: Balloon units are in Pa.
    # Sample the starting distance using a beta distribution, within 200km.
    radius = jax.random.beta(keys[0], alpha, beta).item()
    radius = units.Distance(km=200.0 * radius)
    theta = jax.random.uniform(keys[1], (), minval=0.0, maxval=2.0 * jax.numpy.pi)

    x = math.cos(theta) * radius
    y = math.sin(theta) * radius
    latlng = sampling.sample_location(keys[2])

    pressure = sampling.sample_pressure(keys[3], atmosphere)
    upwelling_infrared = sampling.sample_upwelling_infrared(keys[4])
    b_state = balloon.BalloonState(
            center_latlng=latlng,
            x=x,
            y=y,
            pressure=pressure,
            date_time=start_date_time,
            upwelling_infrared=upwelling_infrared)
    stable_init.cold_start_to_stable_params(b_state, atmosphere)
    return b_state

wind_forecast = generative_wind_field_factory()
wind_forecast.reset_forecast(jax.random.PRNGKey(seed=0), start_date_time)
wind_forecast.reset(jax.random.PRNGKey(seed=0), start_date_time)
def get_wind_sample(balloon_state: balloon.BalloonState):
    return wind_forecast.get_ground_truth(balloon_state.x, balloon_state.y, balloon_state.pressure, balloon_state.time_elapsed)

def test_wind_measurements():
    """"tests that wind measurements at the same state are the same """


    balloon_state = initialize_balloon()
    print(get_wind_sample(balloon_state))
    print("____________________")
    print(get_wind_sample(balloon_state))


def test_initialization():
    balloon_state = initialize_balloon()

    ble_balloon =  balloon.Balloon(initialize_balloon())
    jax_balloon = JaxBalloon(JaxBalloonState.from_ble_state(balloon_state))

    print(ble_balloon.state)
    print(jax_balloon.state)

def compare_prints(ble_state, jax_state):
    print(f"Center LatLng: BLE={ble_state.center_latlng}, JAX={jax_state.center_latlng}")
    print(f"x: BLE={ble_state.x}, JAX={jax_state.x}")
    print(f"y: BLE={ble_state.y}, JAX={jax_state.y}")
    print(f"Pressure: BLE={ble_state.pressure}, JAX={jax_state.pressure}")
    print(f"Ambient Temperature: BLE={ble_state.ambient_temperature}, JAX={jax_state.ambient_temperature}")
    print(f"Internal Temperature: BLE={ble_state.internal_temperature}, JAX={jax_state.internal_temperature}")
    print(f"Envelope Volume: BLE={ble_state.envelope_volume}, JAX={jax_state.envelope_volume}")
    print(f"Superpressure: BLE={ble_state.superpressure}, JAX={jax_state.superpressure}")
    print(f"Status: BLE={ble_state.status}, JAX={jax_state.status}")
    print(f"ACS Power: BLE={ble_state.acs_power}, JAX={jax_state.acs_power}")
    print(f"ACS Mass Flow: BLE={ble_state.acs_mass_flow}, JAX={jax_state.acs_mass_flow}")
    print(f"Mols Air: BLE={ble_state.mols_air}, JAX={jax_state.mols_air}")
    print(f"Solar Charging: BLE={ble_state.solar_charging}, JAX={jax_state.solar_charging}")
    print(f"Power Load: BLE={ble_state.power_load}, JAX={jax_state.power_load}")
    print(f"Battery Charge: BLE={ble_state.battery_charge}, JAX={jax_state.battery_charge}")
    print(f"Date Time: BLE={ble_state.date_time}, JAX={jax_state.date_time}")
    print(f"Time Elapsed: BLE={ble_state.time_elapsed}, JAX={jax_state.time_elapsed}")
    print(f"Payload Mass: BLE={ble_state.payload_mass}, JAX={jax_state.payload_mass}")
    print(f"Envelope Mass: BLE={ble_state.envelope_mass}, JAX={jax_state.envelope_mass}")
    print(f"Envelope Max Superpressure: BLE={ble_state.envelope_max_superpressure}, JAX={jax_state.envelope_max_superpressure}")
    print(f"Envelope Volume Base: BLE={ble_state.envelope_volume_base}, JAX={jax_state.envelope_volume_base}")
    print(f"Envelope Volume DV Pressure: BLE={ble_state.envelope_volume_dv_pressure}, JAX={jax_state.envelope_volume_dv_pressure}")
    print(f"Envelope COD: BLE={ble_state.envelope_cod}, JAX={jax_state.envelope_cod}")
    print(f"Daytime Power Load: BLE={ble_state.daytime_power_load}, JAX={jax_state.daytime_power_load}")
    print(f"Nighttime Power Load: BLE={ble_state.nighttime_power_load}, JAX={jax_state.nighttime_power_load}")
    print(f"ACS Valve Hole Diameter Meters: BLE={ble_state.acs_valve_hole_diameter}, JAX={jax_state.acs_valve_hole_diameter_meters}")
    print(f"Battery Capacity: BLE={ble_state.battery_capacity}, JAX={jax_state.battery_capacity}")
    print(f"Upwelling Infrared: BLE={ble_state.upwelling_infrared}, JAX={jax_state.upwelling_infrared}")
    print(f"Mols Lift Gas: BLE={ble_state.mols_lift_gas}, JAX={jax_state.mols_lift_gas}")

def test_simulate_one_step():


    balloon_state = initialize_balloon()

    ble_balloon =  balloon.Balloon(balloon_state)
    jax_balloon = JaxBalloon(JaxBalloonState.from_ble_state(balloon_state))
    
    for i in range(2):
        action = control.AltitudeControlCommand.UP
        wind_vector = get_wind_sample(balloon_state)
        balloon._simulate_step_internal(ble_balloon.state, wind_vector, atmosphere, action, dt.timedelta(seconds=10))

        action = 2
        jax_wind_vector = jnp.array([ wind_vector.u.meters_per_second, wind_vector.v.meters_per_second ])
        jax_balloon._simulate_step_internal(jax_wind_vector, atmosphere.to_jax_atmopshere(), action, 10) 

        ble_state = ble_balloon.state
        jax_state = jax_balloon.state

        print("________________________________________________________")
        print(f"Iteration {i}")
        compare_prints(ble_state, jax_state)


# test_initialization()

test_simulate_one_step()
