import jax.numpy as jnp
from balloon_learning_environment.utils import constants
from balloon_learning_environment.utils import units
import jax
import jax.numpy as jnp
import math
from typing import Tuple
import datetime as dt
import s2sphere as s2
from jax.tree_util import register_pytree_node_class

# --- Physics Constants ---

GRAVITY: float = 9.80665  # [m/s^2]
NUM_SECONDS_PER_HOUR = 3_600
NUM_SECONDS_PER_DAY: int = 86_400
UNIVERSAL_GAS_CONSTANT: float = 8.3144621  # [J/(mol.K)]
DRY_AIR_MOLAR_MASS: float = 0.028964922481160  # Dry Air. [kg/mol]
HE_MOLAR_MASS: float = 0.004002602  # Helium.  [kg/mol]
DRY_AIR_SPECIFIC_GAS_CONSTANT: float = (
UNIVERSAL_GAS_CONSTANT / DRY_AIR_MOLAR_MASS)  # [J/(kg.K)]


# --- RL constants ---
# Amount of time that elapses between agent steps.
AGENT_TIME_STEP: dt.timedelta = dt.timedelta(minutes=3)
# Pressure limits for the Perciatelli features.
PERCIATELLI_PRESSURE_RANGE_MIN: int = 5000
PERCIATELLI_PRESSURE_RANGE_MAX: int = 14000

def drem(x, y):
    """Computes remainder of x / y, rounding to nearest integer."""
    return jnp.remainder(x, y) - jnp.sign(x) * jnp.sign(y) * (jnp.abs(jnp.remainder(x, y)) >= y / 2)

class JaxLatLng:
    def __init__(self, lat, lng):
        self.lat = lat
        self.lng = lng

    @classmethod
    def from_radians(cls, lat, lng):
        return cls(lat, lng)

    def tree_flatten(self): 
        return (self.lat, self.lon), {}
    
    def normalized(self):
        return self.__class__(
            # TODO: convert max(.., min(...,...))) to jax!
            max(-math.pi / 2.0, min(math.pi / 2.0, self.lat().radians)),
            drem(self.lng().radians, 2 * math.pi))

    @classmethod
    def tree_unflatten(cls, aux_data, children): 
        return cls(*children)

register_pytree_node_class(JaxLatLng)

def calculate_jax_latlng_from_offset(center_latlng: JaxLatLng,
                                 x: 'units.Distance',
                                 y: 'units.Distance') -> JaxLatLng:
  """Calculates a new lat lng given an origin and x y offsets.

  Args:
    center_latlng: The starting latitude and longitude.
    x: An offset from center_latlng parallel to longitude.
    y: An offset from center_latlng parallel to latitude.

  Returns:
    A new latlng that is the specified distance from the start latlng.
  """
  # x and y are swapped to give heading with 0 degrees = North.
  # This is equivalent to pi / 2 - atan2(y, x).
  heading = jnp.atan2(x.km, y.km)  # In radians.
  angle = units.relative_distance(x, y) / _EARTH_RADIUS  # In radians.
    # units.relative_distance is a Distance object
    #_EARTH_RADIUS = units.Distance(km=6371)
    # idea is Distance / Distance is 
  cos_angle = jnp.cos(angle)
  sin_angle = jnp.sin(angle)
  sin_from_lat = jnp.sin(center_latlng.lat().radians)
  cos_from_lat = jnp.cos(center_latlng.lat().radians)

  sin_lat = (cos_angle * sin_from_lat +
             sin_angle * cos_from_lat * math.cos(heading))
  d_lng = jnp.atan2(sin_angle * cos_from_lat * math.sin(heading),
                     cos_angle - sin_from_lat * sin_lat)

  new_lat = jnp.asin(sin_lat)
  # TODO: convert following line to jax
  new_lat = min(max(new_lat, -math.pi / 2.0), math.pi / 2.0)
  new_lng = center_latlng.lng().radians + d_lng

  return JaxLatLng.from_radians(new_lat, new_lng).normalized()


def solar_calculator(latlng: JaxLatLng,
                     time: 'float, seconds') -> Tuple[float, float, float]:
    """Computes solar elevation, azimuth, and flux given latitude/longitude/time.

    Args:
        latlng: The latitude and longitude.
        time: Datetime object.

    Returns:
        el_deg: Solar elevation in degrees.
        az_deg: Solar azimuth in degrees.
        flux: Solar flux in W/m^2.
    """
    if not latlng.is_valid:
        raise ValueError(f'solar_calculator: latlng is invalid: {latlng}.')
    if time.tzinfo is None:
        raise ValueError('time parameter needs timezone. Try UTC.')

    fraction_of_day = (int(time.timest amp()) % constants.NUM_SECONDS_PER_DAY) / constants.NUM_SECONDS_PER_DAY
    julian_day_number = (367.0 * time.year - jnp.floor(7.0 * (time.year + jnp.floor(
        (time.month + 9.0) / 12.0)) / 4.0) - jnp.floor(3.0 * (jnp.floor(
        (time.year + (time.month - 9.0) / 7.0) / 100.0) + 1.0) / 4.0) +
                        jnp.floor(275.0 * time.month / 9.0) + time.day + 1721028.5)

    julian_time = julian_day_number + fraction_of_day
    julian_century = (julian_time - 2451545.0) / 36525.0

    geometric_mean_long_sun = math.radians(
        280.46646 + julian_century * (36000.76983 + julian_century * 0.0003032))
    sin2l0 = jnp.sin(2.0 * geometric_mean_long_sun)
    cos2l0 = jnp.cos(2.0 * geometric_mean_long_sun)
    sin4l0 = jnp.sin(4.0 * geometric_mean_long_sun)

    geometric_mean_anomaly_sun = math.radians(
        357.52911 + julian_century * (35999.05029 - 0.0001537 * julian_century))
    sinm0 = jnp.sin(geometric_mean_anomaly_sun)
    sin2m0 = jnp.sin(2.0 * geometric_mean_anomaly_sun)
    sin3m0 = jnp.sin(3.0 * geometric_mean_anomaly_sun)

    mean_obliquity_of_ecliptic = math.radians(23.0 + (26.0 + (
        (21.448 - julian_century *
         (46.815 + julian_century *
          (0.00059 - julian_century * 0.001813)))) / 60.0) / 60.0)

    obliquity_correction = mean_obliquity_of_ecliptic + math.radians(
        0.00256 * jnp.cos(math.radians(125.04 - 1934.136 * julian_century)))

    var_y = jnp.tan(obliquity_correction / 2.0)**2

    eccentricity_earth = 0.016708634 - julian_century * (
        0.000042037 + 0.0000001267 * julian_century)

    equation_of_time = (4.0 *
                        (var_y * sin2l0 - 2.0 * eccentricity_earth * sinm0 +
                         4.0 * eccentricity_earth * var_y * sinm0 * cos2l0 -
                         0.5 * var_y * var_y * sin4l0 -
                         1.25 * eccentricity_earth * eccentricity_earth * sin2m0))

    hour_angle = math.radians(
        math.fmod(
            1440.0 * fraction_of_day + math.degrees(equation_of_time) +
            4.0 * latlng.lng().degrees, 1440.0)) / 4.0
    if hour_angle < 0:
        hour_angle += math.pi
    else:
        hour_angle -= math.pi

    eq_of_center_sun = math.radians(sinm0 *
                                    (1.914602 - julian_century *
                                     (0.004817 + 0.000014 * julian_century)) +
                                    sin2m0 *
                                    (0.019993 - 0.000101 * julian_century) +
                                    sin3m0 * 0.000289)
    true_long_sun = geometric_mean_long_sun + eq_of_center_sun
    apparent_long_sun = true_long_sun - math.radians(
        0.00569 -
        0.00478 * jnp.sin(math.radians(125.04 - 1934.136 * julian_century)))
    declination_sun = jnp.arcsin(
        jnp.sin(obliquity_correction) * jnp.sin(apparent_long_sun))

    zenith_angle = jnp.arccos(
        jnp.sin(latlng.lat().radians) * jnp.sin(declination_sun) +
        jnp.cos(latlng.lat().radians) * jnp.cos(declination_sun) *
        jnp.cos(hour_angle))

    el_uncorrected_deg = 90.0 - math.degrees(zenith_angle)

    if el_uncorrected_deg > 85.0:
        atmospheric_refraction = 0
    elif el_uncorrected_deg > 5.0:
        tan_seu = jnp.tan(math.radians(el_uncorrected_deg))
        atmospheric_refraction = (58.1 / tan_seu - 0.07 / (tan_seu**3) + 0.000086 /
                                  (tan_seu**5))
    elif el_uncorrected_deg > -0.575:
        atmospheric_refraction = (1735.0 + el_uncorrected_deg *
                                  (-518.2 + el_uncorrected_deg *
                                   (103.4 + el_uncorrected_deg *
                                    (-12.79 + el_uncorrected_deg * 0.711))))
    else:
        atmospheric_refraction = -20.772 / jnp.tan(math.radians(el_uncorrected_deg))

    el_deg = el_uncorrected_deg + atmospheric_refraction / 3600.0

    cos_az = ((jnp.sin(latlng.lat().radians) * jnp.cos(zenith_angle) -
               jnp.sin(declination_sun)) /
              (jnp.cos(latlng.lat().radians) * jnp.sin(zenith_angle)))
    az_unwrapped = jnp.arccos(jnp.clip(cos_az, -1.0, 1.0))
    if hour_angle > 0:
        az_deg = math.degrees(az_unwrapped) + 180.0
    else:
        az_deg = 180.0 - math.degrees(az_unwrapped)

    flux = 1366.0 * (1 + 0.5 * (
        ((1 + eccentricity_earth) /
         (1 - eccentricity_earth))**2 - 1) * jnp.cos(geometric_mean_anomaly_sun))

    return el_deg, az_deg, flux