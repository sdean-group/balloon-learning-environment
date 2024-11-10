"""Compare wind measurements and dynamics model"""
from balloon_learning_environment.env.balloon import balloon
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

key = jax.random.PRNGKey(seed=0)
alpha = 1.2
beta = 2.0

# just use a fixed key for the start time and the atmosphere 
start_date_time = sampling.sample_time(jax.random.PRNGKey(seed=0))
atmosphere = standard_atmosphere.Atmosphere(jax.random.PRNGKey(seed=0))

def initialize_balloon():
    global key
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
def get_wind_sample(balloon):
    return wind_forecast.get_ground_truth(balloon.state.x, balloon.state.y, balloon.state.pressure, balloon.state.time_elapsed)

class AtmosnavBalloon:
    def __init__(self, state):
        self.state = state

    def convert_action_to_heigh

    def simulate_step(
        self,
        wind_vector: wind_field.WindVector,
        atmosphere: standard_atmosphere.Atmosphere,
        action: control.AltitudeControlCommand,
        time_delta: dt.timedelta,
        stride: dt.timedelta = dt.timedelta(seconds=10)):

        altitude_model = DeterministicAltitudeModel(stride.seconds)

        for stride in time_delta: # run stride to match timedelta
            altitude_model_state = jnp.array([ 
                self.state.x.km, self.state.x.km, atmosphere.at_pressure(self.state.pressure).height.km, 0.0 ])
            
            altitude_model.control_input_to_delta_state(
                self.state.time_elapsed.seconds, altitude_model_state, )
            weather_balloon.step()
            next_balloon, _ = balloon.step(time, plan, wind_vector)


# balloons = [
#     balloon.Balloon(balloon.BalloonState()),
#     make_weather_balloon(init_lat, init_lon, init_pressure, start_time, atmosphere, waypoint_time_step, integration_time_stepz)
# ]

def test_wind_measurements():
    """"tests that wind measurements at the same state are the same """
    class LazyBalloon:
        def __init__(self, state):
            self.state = state

    balloon = LazyBalloon(initialize_balloon()) 
    print(get_wind_sample(balloon))
    print("____________________")
    print(get_wind_sample(balloon))