import jax.numpy as jnp
import decimal
import jax_utils
import s2sphere as s2
from balloon_learning_environment.env.balloon import solar
import numpy as np
import datetime as dt
import time
import jax

def test_drem():
    test_cases = [
        (5.5, 2),
        (-5.5, 2),
        (5.5, -2),
        (-5.5, -2),
        (9, 4),
        (-9, 4),
        (9, -4),
        (-9, -4),
        (0, 2),
        (2, 0.5),
        (2.5, 0.5),
        (-2.5, 0.5),
        (2.5, -0.5),
        (-2.5, -0.5),
    ]
    for x, y in test_cases:
        result_jax = jax_utils.drem(x, y)
        result_ori = s2.drem(x, y)
        print(f"Test case (x={x}, y={y}): drem_jax = {result_jax}, drem_ori = {result_ori}, Match = {jnp.isclose(result_jax, result_ori)}")

def test_seconds_to_date():
    test_cases = [ 
        1719782400.0,
        time.time(),
        0.0,                  # Epoch (1970-01-01 00:00:00 UTC)
        31556926.0,           # 1971-01-01 00:00:00 UTC
        946684800.0,          # Y2K (2000-01-01 00:00:00 UTC)
        1582934400.0,         # 2020-02-29 00:00:00 UTC (leap year day)
        1609459200.0,         # 2021-01-01 00:00:00 UTC
        1625078400.0,         # 2021-07-01 00:00:00 UTC
        1661990400.0,         # 2022-09-01 00:00:00 UTC
        1672531200.0,         # 2023-01-01 00:00:00 UTC
        1704067200.0,         # 2024-01-01 00:00:00 UTC (leap year)
        1719792000.0,         # 2024-07-01 00:00:00 UTC
        1723843200.0,         # 2024-08-01 00:00:00 UTC
        1735689600.0,         # 2024-12-01 00:00:00 UTC
        2524608000.0,         # 2050-01-01 00:00:00 UTC
        4102444800.0,         # 2100-01-01 00:00:00 UTC
        7258118400.0,         # 2200-01-01 00:00:00 UTC
         
    ]
    
    for i, seconds in enumerate(test_cases):
        print(f'Test Case {i+1}')
        date = dt.datetime.fromtimestamp(seconds, tz=dt.timezone.utc)
        print(f"dt.datetime result: {(date.year, date.month, date.day)}")
        print(f"jax_utils result: {jax_utils.timestamp_to_date_components_jax(seconds)}")
        print("-" * 40)

test_seconds_to_date()

def test_solar_calculator():
    jax.config.update("jax_enable_x64", True)
    # Define test cases as tuples of (latitude in radians, longitude in radians, datetime object)
    test_cases = [
        (np.radians(40.7128), np.radians(-74.0060), dt.datetime(2024, 6, 21, 12, 0, 0, tzinfo=dt.timezone.utc)),  # New York, summer solstice
        (np.radians(34.0522), np.radians(-118.2437), dt.datetime(2024, 12, 21, 12, 0, 0, tzinfo=dt.timezone.utc)),  # Los Angeles, winter solstice
        (np.radians(51.5074), np.radians(-0.1278), dt.datetime(2024, 3, 21, 12, 0, 0, tzinfo=dt.timezone.utc)),    # London, spring equinox
        (np.radians(-33.8688), np.radians(151.2093), dt.datetime(2024, 9, 23, 12, 0, 0, tzinfo=dt.timezone.utc)),   # Sydney, fall equinox
        (np.radians(90.0), np.radians(0.0), dt.datetime(2024, 6, 21, 12, 0, 0, tzinfo=dt.timezone.utc)),            # North Pole, summer solstice
        (np.radians(-90.0), np.radians(0.0), dt.datetime(2024, 12, 21, 12, 0, 0, tzinfo=dt.timezone.utc)),          # South Pole, winter solstice
        (np.radians(0.0), np.radians(0.0), dt.datetime(2024, 6, 21, 0, 0, 0, tzinfo=dt.timezone.utc)),              # Equator, summer solstice, midnight
    ]

    # Assuming the functions `solar.solar_calculator` and `jax_utils.solar_calculator` are available and the classes `JaxLatLng` and `s2.LatLng` are defined.
    for i, (lat, lng, dt_obj) in enumerate(test_cases):
        # Convert to JaxLatLng and datetime in seconds for jax_utils.solar_calculator
        jax_latlng = jax_utils.JaxLatLng(lat, lng)
        time_seconds = dt_obj.timestamp()

        # Convert to s2.LatLng for solar.solar_calculator
        s2_latlng = s2.LatLng.from_radians(lat, lng)

        # Get baseline result from original function
        original_result = solar.solar_calculator(s2_latlng, dt_obj)

        # Get new JAX-compatible result
        jax_result = jax_utils.solar_calculator(jax_latlng, time_seconds)

        # Print results for comparison
        print(f"Test Case {i+1}")
        print("Original Result:", original_result)
        print("JAX Result:", jax_result)
        print("Difference:", np.abs(np.array(original_result) - np.array(jax_result)))
        print("-" * 40)

# test_solar_calculator()