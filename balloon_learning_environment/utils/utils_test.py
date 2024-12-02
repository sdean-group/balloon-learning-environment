import jax.numpy as jnp
import decimal
import jax_utils
import s2sphere as s2
from balloon_learning_environment.env.balloon import solar
from balloon_learning_environment.env.balloon import acs
from balloon_learning_environment.env.balloon import balloon
from balloon_learning_environment.env.balloon import thermal
import numpy as np
import datetime as dt
import time
import jax
# jax.config.update("jax_enable_x64", True)


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

# test_seconds_to_date()

def test_solar_calculator():
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
        # jax_result = jax.jit(jax_utils.solar_calculator)(jax_latlng, time_seconds)
        jax_result = jax_utils.solar_calculator(jax_latlng, int(time_seconds))

        # Print results for comparison
        print(f"Test Case {i+1}")
        print("Original Result:", original_result)
        print("JAX Result:", jax_result)
        print("Difference:", np.abs(np.array(original_result) - np.array(jax_result)))
        print("-" * 40)
        # input()

test_solar_calculator()


def test_d_temperature_dt():
    test_cases = [
        {
            "balloon_volume": 300.0,
            "balloon_mass": 50.0,
            "balloon_temperature_k": 300.0,
            "ambient_temperature_k": 290.0,
            "pressure_altitude_pa": 101325.0,
            "solar_elevation_deg": 45.0,
            "solar_flux": 600.0,
            "earth_flux": 240.0,
        },
        {
            "balloon_volume": 500.0,
            "balloon_mass": 60.0,
            "balloon_temperature_k": 270.0,
            "ambient_temperature_k": 250.0,
            "pressure_altitude_pa": 50000.0,
            "solar_elevation_deg": 10.0,
            "solar_flux": 300.0,
            "earth_flux": 200.0,
        },
        {
            "balloon_volume": 400.0,
            "balloon_mass": 55.0,
            "balloon_temperature_k": 310.0,
            "ambient_temperature_k": 320.0,
            "pressure_altitude_pa": 101325.0,
            "solar_elevation_deg": 85.0,
            "solar_flux": 1000.0,
            "earth_flux": 260.0,
        },
        {
            "balloon_volume": 350.0,
            "balloon_mass": 50.0,
            "balloon_temperature_k": 280.0,
            "ambient_temperature_k": 275.0,
            "pressure_altitude_pa": 80000.0,
            "solar_elevation_deg": -10.0,
            "solar_flux": 0.0,
            "earth_flux": 200.0,
        },
        {
            "balloon_volume": 600.0,
            "balloon_mass": 70.0,
            "balloon_temperature_k": 260.0,
            "ambient_temperature_k": 240.0,
            "pressure_altitude_pa": 1000.0,
            "solar_elevation_deg": 50.0,
            "solar_flux": 400.0,
            "earth_flux": 150.0,
        },
    ]

    # Test the function
    for i, case in enumerate(test_cases, 1):
        for modu in [ thermal, jax_utils ]:
            d_temp_dt = modu.d_balloon_temperature_dt(**case)
            print(f"{modu.__name__} on test case {i}:")
            print(f"  d_balloon_temperature/dt: {d_temp_dt:.6f} K/s")
            print("-" * 40)

# test_d_temperature_dt()

def test_calculate_superpressure_and_volume():
    test_cases = [
        {
            "mols_lift_gas": 100.0,
            "mols_air": 200.0,
            "internal_temperature": 300.0,
            "pressure": 101325.0,
            "envelope_volume_base": 50.0,
            "envelope_volume_dv_pressure": 0.001,
        },
        {
            "mols_lift_gas": 80.0,
            "mols_air": 150.0,
            "internal_temperature": 250.0,
            "pressure": 50000.0,
            "envelope_volume_base": 45.0,
            "envelope_volume_dv_pressure": 0.002,
        },
        {
            "mols_lift_gas": 110.0,
            "mols_air": 210.0,
            "internal_temperature": 320.0,
            "pressure": 101325.0,
            "envelope_volume_base": 50.0,
            "envelope_volume_dv_pressure": 0.0015,
        },
        {
            "mols_lift_gas": 90.0,
            "mols_air": 300.0,
            "internal_temperature": 290.0,
            "pressure": 105000.0,
            "envelope_volume_base": 40.0,
            "envelope_volume_dv_pressure": 0.0012,
        },
        {
            "mols_lift_gas": 100.0,
            "mols_air": 50.0,
            "internal_temperature": 270.0,
            "pressure": 1000.0,
            "envelope_volume_base": 60.0,
            "envelope_volume_dv_pressure": 0.002,
        },
    ]

    # Example: Call the function with test cases
    for i, case in enumerate(test_cases, 1):
        for modu in [ balloon, jax_utils ]:
            envelope_volume, superpressure = modu.calculate_superpressure_and_volume(**case)
            print(f"{modu.__name__} under test case {i}:")
            print(f"  Envelope Volume: {envelope_volume:.3f} m^3")
            print(f"  Superpressure: {superpressure:.3f} Pa")
            print("-" * 40)

# test_calculate_superpressure_and_volume()

def test_acs():
    test_cases = [
        {"pressure_ratio": 1.05},
        {"pressure_ratio": 1.20},
        {"pressure_ratio": 1.35},
    ]

    for test in test_cases:
        pressure_ratio = test["pressure_ratio"]
        for modu in [acs, jax_utils]:
            print(modu.__name__, "-"*30)
            power = modu.get_most_efficient_power(pressure_ratio)
            efficiency = modu.get_fan_efficiency(pressure_ratio, power)
            mass_flow = modu.get_mass_flow(power, efficiency)
            print(f"Pressure Ratio: {pressure_ratio}")
            print(f"  Power: {power:.2f} W")
            print(f"  Efficiency: {efficiency:.2f}")
            print(f"  Mass Flow: {mass_flow:.6f} kg/s")
            print("-" * 30)

# test_acs()