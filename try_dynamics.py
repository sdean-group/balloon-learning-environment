"""Compare wind measurements and dynamics model"""
from balloon_learning_environment.env.balloon import balloon
from balloon_learning_environment.env.balloon import control
from balloon_learning_environment.agents import opd
from balloon_learning_environment.agents.mpc_agent import DeterministicAltitudeModel, make_weather_balloon, make_plan
from balloon_learning_environment.agents.mpc2_agent import JaxBalloon, JaxBalloonState
from balloon_learning_environment.agents.mpc4_agent import jax_plan_cost, grad_descent_optimizer, get_initial_plans
# from balloon_learning_environment.env.balloon_arena import balloon_arena
from balloon_learning_environment.env.generative_wind_field import generative_wind_field_factory
from balloon_learning_environment.agents.mpc_agent import *
from balloon_learning_environment.env import features
from balloon_learning_environment.env.wind_field import WindVector
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
import matplotlib.pyplot as plt

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
    
    print("________________________________________________________")
    print(f"Iteration 0")
    compare_prints(ble_balloon.state, jax_balloon.state)

    for i in range(1, 100):
        # update ble balloon
        action = control.AltitudeControlCommand.UP
        wind_vector = get_wind_sample(balloon_state)
        state_changes = balloon._simulate_step_internal(ble_balloon.state, wind_vector, atmosphere, action, dt.timedelta(seconds=10))
        for k, v in state_changes.items():
            setattr(ble_balloon.state, k, v)

        if ble_balloon.state.status != balloon.BalloonStatus.OK:
            print("got not OK status")

        # update jax balloon
        action = 2
        jax_wind_vector = jnp.array([ wind_vector.u.meters_per_second, wind_vector.v.meters_per_second ])
        jax_balloon = jax_balloon._simulate_step_internal(jax_wind_vector, atmosphere.to_jax_atmopshere(), action, 10) 

        print("________________________________________________________")
        print(f"Iteration {i}")
        compare_prints(ble_balloon.state, jax_balloon.state)

def test_simulate_step():
    balloon_state = initialize_balloon()

    ble_balloon =  balloon.Balloon(balloon_state)
    jax_balloon = JaxBalloon(JaxBalloonState.from_ble_state(balloon_state))
    

    # update ble balloon
    action = control.AltitudeControlCommand.UP
    wind_vector = get_wind_sample(balloon_state)
    ble_balloon.simulate_step(wind_vector, atmosphere, action, time_delta=dt.timedelta(seconds=100*10), stride=dt.timedelta(seconds=10))

    if ble_balloon.state.status != balloon.BalloonStatus.OK:
        print("got not OK status")

    # update jax balloon
    action = 2
    jax_wind_vector = jnp.array([ wind_vector.u.meters_per_second, wind_vector.v.meters_per_second ])
    jax_balloon = jax_balloon.simulate_step(jax_wind_vector, atmosphere.to_jax_atmopshere(), action, time_delta=100*10, stride=10) 

    compare_prints(ble_balloon.state, jax_balloon.state)

# test_initialization()
# test_simulate_one_step()
# test_simulate_step()

def run_simulation():
    balloon_state = initialize_balloon()

    bballoon = balloon.Balloon(initialize_balloon())
    gballoon = JaxBalloon(JaxBalloonState.from_ble_state(balloon_state))
    # gballoon = GBalloon(
    #     x=balloon_state.x.meters, 
    #     y=balloon_state.y.meters,
    #     pressure=balloon_state.pressure, 
    #     volume=balloon_state.envelope_volume, 
    #     mass=balloon_state.payload_mass + balloon_state.envelope_mass 
    #     # TODO: i think technically there are more mass values but I'm also guessing they are much smaller 
    # )

    print("run_simulation(): balloon initialized")

    time_steps = 500
    time_delta = 3*60
    stride = 60  # seconds
    pressures = []
    altitudes = []

    pressures1=[]
    altitudes1=[]


    jax_atmopshere = atmosphere.to_jax_atmopshere()

    for t in range(time_steps):
        # print(t)
        # action = jnp.sin(t / 30.0)  # Example control action
        # # action = (t//24%3) - 1

        action = (t < 30) * 1.0

        wind_vector = jnp.array([1.0, 0.0])  # Constant wind to the east
        
        # gballoon = gballoon.step(action, wind_vector, atmosphere.to_jax_atmopshere(), stride)
        gballoon = gballoon.simulate_step_continuous(wind_vector, jax_atmopshere, action, time_delta, stride)

        bballoon.simulate_step(
            WindVector(units.Velocity(mps=1.0), units.Velocity(mps=0.0)), 
            atmosphere, 
            action, 
            dt.timedelta(seconds=time_delta), 
            dt.timedelta(seconds=stride))

        pressures.append(gballoon.state.pressure)
        altitudes.append(jax_atmopshere.at_pressure(pressures[-1]).height.km)

        pressures1.append(bballoon.state.pressure)
        altitudes1.append(atmosphere.at_pressure(pressures1[-1]).height.km)

    # Plot results
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    # plt.plot(range(time_steps), pressures1)
    plt.plot(range(time_steps), altitudes1)
    # plt.title("Balloon Pressure Over Time")
    plt.title("balloon.Balloon")
    plt.xlabel("Time (s)")
    # plt.ylabel("Pressure (Pa)")
    plt.ylabel("Altitude (km)")

    plt.subplot(1, 2, 2)
    # plt.plot(range(time_steps), pressures)
    plt.plot(range(time_steps), altitudes)
    plt.title("JaxBalloon")
    plt.xlabel("Time (s)")
    plt.ylabel("Altitude (km)")

    plt.tight_layout()
    plt.show()

# run_simulation()

def test_mpc_initializations():
    # Balloon configuration
    balloon_state = initialize_balloon()
    x = balloon_state.x.km
    y = balloon_state.y.km
    pressure = balloon_state.pressure
    t = balloon_state.time_elapsed.seconds
    jax_atmosphere = atmosphere.to_jax_atmopshere()
    jax_forecast = wind_forecast.to_jax_wind_field()
    waypoint_time_step = 3*60
    integration_time_step = 10
    

    balloon = make_weather_balloon(x, y, pressure, t, jax_atmosphere, waypoint_time_step, integration_time_step)

    @jax.jit
    def step(balloon, time, plan, wind):
        return balloon.step(time, plan, wind)

    # Plan configuration
    plan_steps = 240

    # plan = balloon.state[2] + np.cumsum(np.random.uniform(-0.5, 0.5, plan_steps)).reshape(-1, 1)
    plan, _ = make_plan(t, 100, plan_steps, balloon, jax_forecast, jax_atmosphere, waypoint_time_step, integration_time_step)

    # Plotting
    time = [t]
    actions = [plan[0]]
    altitude = [balloon.state[2]]

    N = (waypoint_time_step * (len(plan)-1)) // integration_time_step
    for _ in range(N):
        balloon,info = step(balloon, time[-1], plan, jnp.array([ 0.0, 0.0 ]))
        actions.append(info['control_input'])
        time.append(time[-1] + integration_time_step)
        altitude.append(balloon.state[2])

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(time, actions)

    plt.subplot(1, 2, 2)
    plt.plot(time, altitude)

    plt.tight_layout()
    plt.show()

# test_mpc_initializations()

def test_opd():
    jax_atmopshere = atmosphere.to_jax_atmopshere()
    jax_wind_forecast = wind_forecast.to_jax_wind_field()

    balloon_state = initialize_balloon()
    start = opd.ExplorerState(
        balloon_state.x.meters,
        balloon_state.y.meters,
        balloon_state.pressure,
        balloon_state.time_elapsed.seconds)


    search_delta_time = 60*60
    plan_delta_time = 3*60
    best_node, best_node_early = opd.run_opd_search(start, jax_wind_forecast, [0, 1, 2], opd.ExplorerOptions(budget=25_000, planning_horizon=240, delta_time=search_delta_time))
    print(best_node)
    print(best_node_early)

    jax_balloon = JaxBalloon(JaxBalloonState.from_ble_state(balloon_state))

    plan = opd.get_plan_from_opd_node(best_node, search_delta_time=search_delta_time, plan_delta_time=plan_delta_time)
    print(len(plan))
    print(jax_plan_cost(plan, jax_balloon, jax_wind_forecast, jax_atmopshere, plan_delta_time, 60))


    initial_plans =get_initial_plans(jax_balloon, 50, jax_wind_forecast, jax_atmopshere, len(plan), plan_delta_time, 60)
    batched_cost = []
    for i in range(len(initial_plans)):
        batched_cost.append(jax_plan_cost(jnp.array(initial_plans[i]), jax_balloon, jax_wind_forecast, jax_atmopshere, plan_delta_time, 60))
    print(np.min(batched_cost))

test_opd()