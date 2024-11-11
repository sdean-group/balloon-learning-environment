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

from typing import Tuple

# --- Physics Constants ---

GRAVITY: float = 9.80665  # [m/s^2]
NUM_SECONDS_PER_HOUR = 3_600
NUM_SECONDS_PER_DAY: int = 86_400
UNIVERSAL_GAS_CONSTANT: float = 8.3144621  # [J/(mol.K)]
DRY_AIR_MOLAR_MASS: float = 0.028964922481160  # Dry Air. [kg/mol]
HE_MOLAR_MASS: float = 0.004002602  # Helium.  [kg/mol]
DRY_AIR_SPECIFIC_GAS_CONSTANT: float = (
UNIVERSAL_GAS_CONSTANT / DRY_AIR_MOLAR_MASS)  # [J/(kg.K)]


def drem(x, y):
    """Computes remainder of x / y, rounding to nearest integer."""
    return x - jnp.round(x / y) * y

class JaxLatLng:
    def __init__(self, lat, lng):
        """ expects lat, lng to be in radians """
        self.lat = lat 
        self.lng = lng

    @classmethod
    def from_radians(cls, lat, lng):
        return cls(lat, lng)

    def tree_flatten(self): 
        return (self.lat, self.lon), {}
    
    def normalized(self):
        return self.__class__(
            jnp.clip(self.lat, min=-jnp.pi/2, max=jnp.pi/2),
            drem(self.lng, 2 * math.pi))

    @classmethod
    def tree_unflatten(cls, aux_data, children): 
        return cls(*children)

# s2.LatLng

register_pytree_node_class(JaxLatLng)

_EARTH_RADIUS = 6371 # kilometers
def calculate_jax_latlng_from_offset(center_latlng: JaxLatLng,
                                 x: 'kilometers',
                                 y: 'kilometers') -> JaxLatLng:
  """Calculates a new lat lng given an origin and x y offsets.

  Args:
    center_latlng: The starting latitude and longitude.
    x: An offset from center_latlng parallel to longitude. in kilometers
    y: An offset from center_latlng parallel to latitude. in kilometers

  Returns:
    A new latlng that is the specified distance from the start latlng.
  """
  # x and y are swapped to give heading with 0 degrees = North.
  # This is equivalent to pi / 2 - atan2(y, x).
  heading = jnp.atan2(x, y)  # In radians.
  angle = jnp.linalg.norm(jnp.array([x, y])).item() / _EARTH_RADIUS  # In radians.
    # units.relative_distance is a Distance object
    #_EARTH_RADIUS = units.Distance(km=6371)
    # idea is Distance / Distance is 
  cos_angle = jnp.cos(angle)
  sin_angle = jnp.sin(angle)
  sin_from_lat = jnp.sin(center_latlng.lat)
  cos_from_lat = jnp.cos(center_latlng.lat)

  sin_lat = (cos_angle * sin_from_lat +
             sin_angle * cos_from_lat * jnp.cos(heading))
  d_lng = jnp.atan2(sin_angle * cos_from_lat * jnp.sin(heading),
                     cos_angle - sin_from_lat * sin_lat)

  new_lat = jnp.asin(sin_lat)
  # TODO: convert following line to jax
  new_lat = min(max(new_lat, -jnp.pi / 2.0), jnp.pi / 2.0)
  new_lng = center_latlng.lng + d_lng

  return JaxLatLng.from_radians(new_lat, new_lng).normalized()


def timestamp_to_date_components_jax(seconds: float):
    """
    JAX-compatible function to convert a timestamp (in seconds since epoch)
    into year, month, day, hour, minute, and second components.

    Args:
        seconds (float): Timestamp in seconds since the Unix epoch (1970-01-01 00:00:00 UTC).

    Returns:
        dict: A dictionary containing year, month, day, hour, minute, and second.
    """
    # Days since Unix epoch (1970-01-01)
    days = jnp.floor(seconds / 86400.0)
    
    # Remaining seconds within the current day
    seconds_in_day = seconds % 86400.0
    hour = jnp.floor(seconds_in_day / 3600).astype(int)
    minute = jnp.floor((seconds_in_day % 3600) / 60).astype(int)
    second = jnp.floor(seconds_in_day % 60).astype(int)
    
    # Julian day to Gregorian calendar conversion
    julian_day = days + 2440587.5  # Unix epoch to Julian day
    f = julian_day + 0.5
    z = jnp.floor(f).astype(int)
    a = z + 32044
    b = (4 * a + 3) // 146097
    c = a - (b * 146097) // 4
    d = (4 * c + 3) // 1461
    e = c - (1461 * d) // 4
    m = (5 * e + 2) // 153

    # Calculating Gregorian year, month, day
    day = (e - (153 * m + 2) // 5 + 1).astype(int)
    month = (m + 3 - 12 * (m // 10)).astype(int)
    year = (b * 100 + d - 4800 + (m // 10)).astype(int)

    return year, month, day

def solar_calculator(latlng: JaxLatLng, time_seconds: float) -> Tuple[float, float, float]:
    """
    Computes solar elevation, azimuth, and flux given latitude/longitude and time in seconds.

    Args:
        latlng: JaxLatLng object containing latitude and longitude.
        time_seconds: Time in seconds (UTC) since the epoch, replacing datetime.

    Returns:
        el_deg: Solar elevation in degrees.
        az_deg: Solar azimuth in degrees.
        flux: Solar flux in W/m^2.
    """
    # Check if latitude is within expected range.
    # if not latlng.is_valid:
    #     raise ValueError(f'solar_calculator: latlng is invalid: {latlng}.')

    # Compute fraction_of_day from time in seconds.
    fraction_of_day = (int(time_seconds) % NUM_SECONDS_PER_DAY) / NUM_SECONDS_PER_DAY

    # Compute Julian day number from Gregorian calendar.
    year, month, day = timestamp_to_date_components_jax(time_seconds)
    julian_day_number = (
        367.0 * year - jnp.floor(7.0 * (year + jnp.floor((month + 9.0) / 12.0)) / 4.0)
        - jnp.floor(3.0 * (jnp.floor((year + (month - 9.0) / 7.0) / 100.0) + 1.0) / 4.0)
        + jnp.floor(275.0 * month / 9.0) + day + 1721028.5
    )

    # Compute Julian time (in days and in centuries).
    julian_time = julian_day_number + fraction_of_day
    julian_century = (julian_time - 2451545.0) / 36525.0

    # Compute solar parameters.
    geometric_mean_long_sun = jnp.radians(
        280.46646 + julian_century * (36000.76983 + julian_century * 0.0003032)
    )
    sin2l0 = jnp.sin(2.0 * geometric_mean_long_sun)
    cos2l0 = jnp.cos(2.0 * geometric_mean_long_sun)
    sin4l0 = jnp.sin(4.0 * geometric_mean_long_sun)

    geometric_mean_anomaly_sun = jnp.radians(
        357.52911 + julian_century * (35999.05029 - 0.0001537 * julian_century)
    )
    sinm0 = jnp.sin(geometric_mean_anomaly_sun)
    sin2m0 = jnp.sin(2.0 * geometric_mean_anomaly_sun)
    sin3m0 = jnp.sin(3.0 * geometric_mean_anomaly_sun)

    mean_obliquity_of_ecliptic = jnp.radians(
        23.0 + (26.0 + ((21.448 - julian_century * (46.815 + julian_century * (0.00059 - julian_century * 0.001813))) / 60.0)) / 60.0
    )
    obliquity_correction = mean_obliquity_of_ecliptic + jnp.radians(
        0.00256 * jnp.cos(jnp.radians(125.04 - 1934.136 * julian_century))
    )

    var_y = jnp.tan(obliquity_correction / 2.0) ** 2
    eccentricity_earth = 0.016708634 - julian_century * (0.000042037 + 0.0000001267 * julian_century)

    equation_of_time = 4.0 * (var_y * sin2l0 - 2.0 * eccentricity_earth * sinm0 + 4.0 * eccentricity_earth * var_y * sinm0 * cos2l0 - 0.5 * var_y * var_y * sin4l0 - 1.25 * eccentricity_earth * eccentricity_earth * sin2m0)

    hour_angle = jnp.radians(jnp.fmod(1440.0 * fraction_of_day + jnp.degrees(equation_of_time) + 4.0 * latlng.lng, 1440.0)) / 4.0
    hour_angle = jnp.where(hour_angle < 0, hour_angle + jnp.pi, hour_angle - jnp.pi)

    eq_of_center_sun = jnp.radians(sinm0 * (1.914602 - julian_century * (0.004817 + 0.000014 * julian_century)) + sin2m0 * (0.019993 - 0.000101 * julian_century) + sin3m0 * 0.000289)
    true_long_sun = geometric_mean_long_sun + eq_of_center_sun
    apparent_long_sun = true_long_sun - jnp.radians(0.00569 - 0.00478 * jnp.sin(jnp.radians(125.04 - 1934.136 * julian_century)))
    declination_sun = jnp.arcsin(jnp.sin(obliquity_correction) * jnp.sin(apparent_long_sun))

    zenith_angle = jnp.arccos(
        jnp.sin(latlng.lat) * jnp.sin(declination_sun) + jnp.cos(latlng.lat) * jnp.cos(declination_sun) * jnp.cos(hour_angle)
    )
    el_uncorrected_deg = 90.0 - jnp.degrees(zenith_angle)

    atmospheric_refraction = jnp.where(
        el_uncorrected_deg > 85.0, 0,
        jnp.where(
            el_uncorrected_deg > 5.0,
            (58.1 / jnp.tan(jnp.radians(el_uncorrected_deg)) - 0.07 / jnp.tan(jnp.radians(el_uncorrected_deg))**3 + 0.000086 / jnp.tan(jnp.radians(el_uncorrected_deg))**5),
            jnp.where(
                el_uncorrected_deg > -0.575,
                1735.0 + el_uncorrected_deg * (-518.2 + el_uncorrected_deg * (103.4 + el_uncorrected_deg * (-12.79 + el_uncorrected_deg * 0.711))),
                -20.772 / jnp.tan(jnp.radians(el_uncorrected_deg))
            )
        )
    )

    el_deg = el_uncorrected_deg + atmospheric_refraction / 3600.0

    cos_az = (jnp.sin(latlng.lat) * jnp.cos(zenith_angle) - jnp.sin(declination_sun)) / (jnp.cos(latlng.lat) * jnp.sin(zenith_angle))
    az_unwrapped = jnp.arccos(jnp.clip(cos_az, -1.0, 1.0))
    az_deg = jnp.where(hour_angle > 0, jnp.degrees(az_unwrapped) + 180.0, 180.0 - jnp.degrees(az_unwrapped))

    flux = 1366.0 * (1 + 0.5 * (((1 + eccentricity_earth) / (1 - eccentricity_earth))**2 - 1) * jnp.cos(geometric_mean_anomaly_sun))

    return el_deg, az_deg, flux
