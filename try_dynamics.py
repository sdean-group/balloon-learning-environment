"""Compare wind measurements and dynamics model"""
from balloon_learning_environment.env.balloon import balloon
from balloon_learning_environment.env.balloon import control
from balloon_learning_environment.agents import opd
from balloon_learning_environment.agents.mpc_agent import DeterministicAltitudeModel, make_weather_balloon, make_plan
from balloon_learning_environment.agents.mpc2_agent import JaxBalloon, JaxBalloonState
from balloon_learning_environment.agents.mpc4_agent import jax_plan_cost, grad_descent_optimizer, get_initial_plans, sigmoid
# from balloon_learning_environment.env.balloon_arena import balloon_arena
from balloon_learning_environment.env.generative_wind_field import generative_wind_field_factory
from balloon_learning_environment.agents.mpc_agent import *
from balloon_learning_environment.env import features
from balloon_learning_environment.env.wind_field import WindVector
from balloon_learning_environment.env.balloon import stable_init
from balloon_learning_environment.env.balloon import standard_atmosphere


from balloon_learning_environment.utils import units
from balloon_learning_environment.utils import sampling
import json
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
START_DATE_TIME = sampling.sample_time(jax.random.PRNGKey(seed=0))
ATMOPSHERE = standard_atmosphere.Atmosphere(jax.random.PRNGKey(seed=0))

def initialize_balloon(key, start_date_time=START_DATE_TIME, atmosphere=ATMOPSHERE, return_key=False):
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
    if return_key:
        return b_state, key
    else:
        return b_state

wind_forecast = generative_wind_field_factory()
wind_forecast.reset_forecast(jax.random.PRNGKey(seed=0), START_DATE_TIME)
wind_forecast.reset(jax.random.PRNGKey(seed=0), START_DATE_TIME)
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
        state_changes = balloon._simulate_step_internal(ble_balloon.state, wind_vector, ATMOPSHERE, action, dt.timedelta(seconds=10))
        for k, v in state_changes.items():
            setattr(ble_balloon.state, k, v)

        if ble_balloon.state.status != balloon.BalloonStatus.OK:
            print("got not OK status")

        # update jax balloon
        action = 2
        jax_wind_vector = jnp.array([ wind_vector.u.meters_per_second, wind_vector.v.meters_per_second ])
        jax_balloon = jax_balloon._simulate_step_internal(jax_wind_vector, ATMOPSHERE.to_jax_atmosphere(), action, 10) 

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
    ble_balloon.simulate_step(wind_vector, ATMOPSHERE, action, time_delta=dt.timedelta(seconds=100*10), stride=dt.timedelta(seconds=10))

    if ble_balloon.state.status != balloon.BalloonStatus.OK:
        print("got not OK status")

    # update jax balloon
    action = 2
    jax_wind_vector = jnp.array([ wind_vector.u.meters_per_second, wind_vector.v.meters_per_second ])
    jax_balloon = jax_balloon.simulate_step(jax_wind_vector, ATMOPSHERE.to_jax_atmosphere(), action, time_delta=100*10, stride=10) 

    compare_prints(ble_balloon.state, jax_balloon.state)

# test_initialization()
# test_simulate_one_step()
# test_simulate_step()

def test_jax_balloon_fidelity():
    seed = 0
    key = jax.random.PRNGKey(seed=seed)
    # print('A', key)
    key, arena_key = jax.random.split(key, 2)
    # print('B', key, arena_key)
    arena_key, atmosphere_key, time_key = jax.random.split(arena_key, 3)
    # print('C', arena_key, atmosphere_key, time_key)

    atmosphere = standard_atmosphere.Atmosphere(atmosphere_key)
    start_date_time = sampling.sample_time(time_key)

    # local_timezone = dt.datetime.utcnow().astimezone().tzinfo
    # print('D', start_date_time, time_key, local_timezone)

    balloon_state, arena_key = initialize_balloon(arena_key, start_date_time, atmosphere, return_key=True)

    balloon_state: balloon.BalloonState = balloon_state
    jax_balloon_state: JaxBalloonState = JaxBalloonState.from_ble_state(balloon_state)

    google_balloon = balloon.Balloon(balloon_state)
    jax_balloon = JaxBalloon(jax_balloon_state)

    jax_atmosphere = atmosphere.to_jax_atmosphere()
    
    def get_plan(going_up, steps, random_walk, plan_size):
        plan = np.zeros((plan_size, ))
        plan[:steps] = 1 if going_up else -1
        plan[steps:] += np.random.uniform(-random_walk, random_walk, plan_size - steps)

        return plan
        
    plan_size = 240
    plan = get_plan(False, 40, 0.3, plan_size)


    jax_balloon_properties = set(prop for prop in dir(jax_balloon_state) if not prop.startswith("_") and not callable(getattr(jax_balloon_state, prop)))
    google_balloon_properties = set(prop for prop in dir(balloon_state) if not prop.startswith("_") and not callable(getattr(balloon_state, prop)))

    data = {"google": {prop: [] for prop in google_balloon_properties}, "jax": {prop: [] for prop in jax_balloon_properties}}

    for action in plan:
        # NOTE: 10 seconds (or when time_delta == stride), it will only run one inner loop at a time which is better for debugging
            
        # Simulate step for Google balloon
        wind_vector = WindVector(units.Velocity(mps=1.0), units.Velocity(mps=0.0))
        google_balloon.simulate_step(wind_vector, atmosphere, action, dt.timedelta(seconds=10), dt.timedelta(seconds=10))

        # Log Google balloon properties
        for prop in google_balloon_properties:
            data["google"][prop].append(getattr(google_balloon.state, prop))

        # Simulate step for JAX balloon
        wind_vector = jnp.array([1.0, 0.0])
        jax_balloon = jax_balloon.simulate_step_continuous(wind_vector, jax_atmosphere, action, 10, 10)

        # Log JAX balloon properties
        for prop in jax_balloon_properties:
            data["jax"][prop].append(getattr(jax_balloon.state, prop))


    jax_to_plot = (
        "x",
        "y",
        "pressure",
        "ambient_temperature",
        "internal_temperature",
        "envelope_volume",
        "superpressure",
        "status",
        "acs_power",
        "acs_mass_flow",
        "mols_air",
        "solar_charging",
        "power_load",
        "battery_charge",
        "date_time",
        "time_elapsed",
        "payload_mass",
        "envelope_mass",
        "envelope_max_superpressure",
        "envelope_volume_base",
        "envelope_volume_dv_pressure",
        "envelope_cod",
        "daytime_power_load",
        "nighttime_power_load",
        "acs_valve_hole_diameter_meters",
        "battery_capacity",
        "upwelling_infrared",
        "mols_lift_gas")
    
    google_to_plot = (
        "x",
        "y",
        "pressure",
        "ambient_temperature",
        "internal_temperature",
        "envelope_volume",
        "superpressure",
        "status",
        "acs_power",
        "acs_mass_flow",
        "mols_air",
        "solar_charging",
        "power_load",
        "battery_charge",
        "date_time",
        "time_elapsed",
        "payload_mass",
        "envelope_mass",
        "envelope_max_superpressure",
        "envelope_volume_base",
        "envelope_volume_dv_pressure",
        "envelope_cod",
        "daytime_power_load",
        "nighttime_power_load",
        "acs_valve_hole_diameter",
        "battery_capacity",
        "upwelling_infrared",
        "mols_lift_gas")

    google_getters = {
        "status": lambda x: 0,
        "x": lambda x: x.meters,
        "y": lambda y: y.meters,
        "acs_power": lambda p: p.watts,
        "solar_charging": lambda p: p.watts,
        "power_load": lambda p: p.watts,
        "battery_charge": lambda p: p.watt_hours,
        "date_time": lambda t: t.timestamp(),
        "time_elapsed": lambda t: t.total_seconds(),
        "daytime_power_load": lambda p: p.watts,
        "nighttime_power_load": lambda p: p.watts,
        "acs_valve_hole_diameter": lambda d: d.meters,
        "battery_capacity": lambda b: b.watt_hours,
    }

    for google_prop, jax_prop in zip(google_to_plot, jax_to_plot):
        google_values, jax_values = data["google"][google_prop], np.array(data["jax"][jax_prop])

        if google_prop in google_getters:
            converter = google_getters[google_prop]
            google_values = np.array([converter(value) for value in google_values])

        print(google_prop, '=====')

        # Find the first index where the values differ
        for idx, (g_val, j_val) in enumerate(zip(google_values, jax_values)):
            if not np.isclose(g_val, j_val, atol=1e-8):  # Adjust tolerance as needed
                print(f"First difference at index {idx}: Google={g_val}, JAX={j_val}")
                break



    # plt.plot(range(plan_size), np.array(data["google"]["pressure"]) - np.array(data["jax"]["pressure"]))
    # # plt.plot(range(plan_size))
    # plt.show()

    # print(balloon_state.x.km)
    # print(balloon_state.y.km)
    # print(atmosphere.at_pressure(balloon_state.pressure).height.km)


# test_jax_balloon_fidelity()

def test_mpc4_initialization():

    jax_atmopshere = ATMOPSHERE.to_jax_atmosphere()

    for seed in range(10):
        jax_pressures_table = {'up': [], 'down': []}
        jax_altitudes_table = {'up': [], 'down': []}

        google_pressures_table = {'up': [], 'down': []}
        google_altitudes_table = {'up': [], 'down': []}

        for direction in ["up", "down"]:
            going_up = direction == "up"

            jax_pressures = jax_pressures_table[direction]
            jax_altitudes = jax_altitudes_table[direction]

            google_pressures = google_pressures_table[direction]
            google_altitudes = google_altitudes_table[direction]

            balloon_state = initialize_balloon(key=jax.random.PRNGKey(seed=seed))
            google_balloon = balloon.Balloon(balloon_state)
            jax_balloon = JaxBalloon(JaxBalloonState.from_ble_state(balloon_state))

            print("run_simulation(): balloon initialized")

            time_steps = 240
            time_delta = 3*60
            stride = 10  # seconds


            for t in range(time_steps):
                # action = control.AltitudeControlCommand.UP if going_up else control.AltitudeControlCommand.DOWN
                action = 1.0 if going_up else -1.0

                if min(len(google_pressures), len(jax_pressures)) > 0:    
                    if going_up:
                        if min(jax_altitudes[-1], google_altitudes[-1]) >= 19.1:
                            break
                    else:
                        if max(jax_altitudes[-1], google_altitudes[-1]) <= 15.4:
                            break

                wind_vector = jnp.array([1.0, 0.0])  # Constant wind to the east
                
                jax_balloon = jax_balloon.simulate_step_continuous(wind_vector, jax_atmopshere, action, time_delta, stride)
                # jax_balloon = jax_balloon.simulate_step(wind_vector, jax_atmopshere, action, time_delta, stride)

                try:
                    google_balloon.simulate_step(
                        WindVector(units.Velocity(mps=1.0), units.Velocity(mps=0.0)), 
                        ATMOPSHERE, 
                        action, 
                        dt.timedelta(seconds=time_delta), 
                        dt.timedelta(seconds=stride))
                except AssertionError as e:
                    print(f'Seed={seed}: {e}')
                    break

                jax_pressures.append(jax_balloon.state.pressure)
                jax_altitudes.append(jax_atmopshere.at_pressure(jax_pressures[-1]).height.km)

                google_pressures.append(google_balloon.state.pressure)
                google_altitudes.append(ATMOPSHERE.at_pressure(google_pressures[-1]).height.km)

        # Plot results
        
        plt.figure(figsize=(12, 5))
        plt.title(f'Seed {seed}')

        # Left plot: Google up balloon vs JAX up balloon
        plt.subplot(1, 2, 1)
        plt.plot(range(len(google_altitudes_table['up'])), google_altitudes_table['up'], label='Google Up Balloon')
        plt.plot(range(len(jax_altitudes_table['up'])), jax_altitudes_table['up'], label='JAX Up Balloon')
        plt.title("Google Up Balloon vs JAX Up Balloon")
        plt.xlabel("Time Steps")
        plt.ylabel("Altitude (km)")
        plt.legend()

        # Right plot: Google up balloon vs JAX down balloon
        plt.subplot(1, 2, 2)
        plt.plot(range(len(google_altitudes_table['down'])), google_altitudes_table['down'], label='Google Down Balloon')
        plt.plot(range(len(jax_altitudes_table['down'])), jax_altitudes_table['down'], label='JAX Down Balloon')
        plt.title("Google Up Balloon vs JAX Down Balloon")
        plt.xlabel("Time Steps")
        plt.ylabel("Altitude (km)")
        plt.legend()

        plt.tight_layout()
        plt.savefig(f"diagnostics/initialization_fidelity/seed_{seed}.png")


# test_mpc4_initialization()

def test_mpc4_initializations():
    _start = time.time()

    seed = 0
    key = jax.random.PRNGKey(seed=seed)
    # print('A', key)
    key, arena_key = jax.random.split(key, 2)
    # print('B', key, arena_key)
    arena_key, atmosphere_key, time_key = jax.random.split(arena_key, 3)
    # print('C', arena_key, atmosphere_key, time_key)

    atmosphere = standard_atmosphere.Atmosphere(atmosphere_key)
    start_date_time = sampling.sample_time(time_key)

    # local_timezone = dt.datetime.utcnow().astimezone().tzinfo
    # print('D', start_date_time, time_key, local_timezone)

    balloon_state, arena_key = initialize_balloon(arena_key, start_date_time, atmosphere, return_key=True)

    balloon_state: balloon.BalloonState = balloon_state
    jax_balloon_state: JaxBalloonState = JaxBalloonState.from_ble_state(balloon_state)

    google_balloon = balloon.Balloon(balloon_state)
    jax_balloon = JaxBalloon(jax_balloon_state)

    jax_atmosphere = atmosphere.to_jax_atmosphere()
    jax_forecast = wind_forecast.to_jax_wind_field()

    def get_altitude_km(jax_balloon):
        return jax_atmosphere.at_pressure(jax_balloon.state.pressure).height.km

    np.random.seed(seed=0)

    print('Setup:', time.time() - _start)
    _start = time.time()
    
    plans = sigmoid(get_initial_plans(jax_balloon, 500, jax_forecast, jax_atmosphere, 240, 3*60, 10))
    
    print('Generate initial plans:', time.time() - _start)
    _start = time.time()

    altitude_logs = []
    for plan in plans:
        test_balloon = jax_balloon
        altitude_log = [get_altitude_km(test_balloon)]
        for action in plan:
            test_balloon = test_balloon.simulate_step_continuous(jnp.array([ 0.0, 0.0 ]), jax_atmosphere, action, 3*60, 10)
            altitude_log.append(get_altitude_km(test_balloon))
        altitude_logs.append(altitude_log)
    
    print('Simulate initial plans:', time.time() - _start)
    _start = time.time()

    end_altitudes = [ altitude_log[-1] for altitude_log in altitude_logs 
                    #  if altitude_log[-1] <= get_altitude_km(jax_balloon)
                     ]
    plt.hist(end_altitudes)
    plt.show()

    for altitude_log in altitude_logs:
        plt.plot(range(len(altitude_log)), altitude_log)
    print('Plot everything:', time.time() - _start)
    _start = time.time()
    
    plt.show()

    
# test_mpc4_initializations()

def test_mpc_initializations():
    # Balloon configuration
    balloon_state = initialize_balloon()
    x = balloon_state.x.km
    y = balloon_state.y.km
    pressure = balloon_state.pressure
    t = balloon_state.time_elapsed.seconds
    jax_atmosphere = ATMOPSHERE.to_jax_atmosphere()
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
    jax_atmopshere = ATMOPSHERE.to_jax_atmosphere()
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

# test_opd()