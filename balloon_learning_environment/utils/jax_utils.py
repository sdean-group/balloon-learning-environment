import jax.numpy as jnp
from balloon_learning_environment.utils import constants
from balloon_learning_environment.utils import units
import jax
from jax import lax
import jax.numpy as jnp
from typing import Tuple
import datetime as dt
import s2sphere as s2
from jax.tree_util import register_pytree_node_class

from jax.scipy.interpolate import RegularGridInterpolator

from typing import Tuple

#####
# Solar.py
#####

# --- Physics Constants ---

GRAVITY: float = 9.80665  # [m/s^2]
NUM_SECONDS_PER_HOUR = 3_600
NUM_SECONDS_PER_DAY: int = 86_400
UNIVERSAL_GAS_CONSTANT: float = 8.3144621  # [J/(mol.K)]
DRY_AIR_MOLAR_MASS: float = 0.028964922481160  # Dry Air. [kg/mol]
HE_MOLAR_MASS: float = 0.004002602  # Helium.  [kg/mol]
DRY_AIR_SPECIFIC_GAS_CONSTANT: float = (
    UNIVERSAL_GAS_CONSTANT / DRY_AIR_MOLAR_MASS
)  # [J/(kg.K)]
MIN_SOLAR_EL_DEG = -4.242
_SOLAR_VIEW_FACTOR = 0.25
_EARTH_VIEW_FACTOR = 0.4605
_PE01_REFLECTIVITY = 0.0291
_PE01_ABSORPTIVITY_SOLAR = 0.01435
_PE01_ABSORPTIVITY_IR_BASE = 0.04587
_PE01_ABSORPTIVITY_IR_D_TEMPERATURE = 0.000232  # [1/K]
_PE01_ABSORPTIVITY_IR_REF_TEMPERATURE = 210  # [K]
_PE01_FILM_SPECIFIC_HEAT = 1500  # [J/(kg.K)]

_STEFAN_BOLTZMAN = 0.000000056704  # [W/(m^2.K^4)]
_UNIVERSAL_GAS_CONSTANT = 8.3144621  # [J/(mol.K)]
_DRY_AIR_MOLAR_MASS = 0.028964922481160  # Dry Air. [kg/mol]


def drem(x, y):
    """Computes remainder of x / y, rounding to nearest integer."""
    return x - jnp.round(x / y) * y


class JaxLatLng:
    def __init__(self, lat, lng):
        """expects lat, lng to be in radians"""
        self.lat = lat
        self.lng = lng

    @classmethod
    def from_radians(cls, lat, lng):
        return cls(lat, lng)

    def tree_flatten(self):
        return (self.lat, self.lng), {}

    def normalized(self):
        return self.__class__(
            jnp.clip(self.lat, min=-jnp.pi / 2, max=jnp.pi / 2),
            drem(self.lng, 2 * jnp.pi),
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)
    
    def __str__(self):
        return f"JaxLatLng(lat={self.lat}, lng={self.lng})"

    def __repr__(self): return str(self)

# s2.LatLng

register_pytree_node_class(JaxLatLng)

_EARTH_RADIUS = 6371  # kilometers


def calculate_jax_latlng_from_offset(
    center_latlng: JaxLatLng, x: "kilometers", y: "kilometers"
) -> JaxLatLng:
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
    angle = jnp.linalg.norm(jnp.array([x, y])) / _EARTH_RADIUS  # In radians.
    # units.relative_distance is a Distance object
    # _EARTH_RADIUS = units.Distance(km=6371)
    # idea is Distance / Distance is
    cos_angle = jnp.cos(angle)
    sin_angle = jnp.sin(angle)
    sin_from_lat = jnp.sin(center_latlng.lat)
    cos_from_lat = jnp.cos(center_latlng.lat)

    sin_lat = cos_angle * sin_from_lat + sin_angle * cos_from_lat * jnp.cos(heading)
    d_lng = jnp.atan2(
        sin_angle * cos_from_lat * jnp.sin(heading), cos_angle - sin_from_lat * sin_lat
    )

    new_lat = jnp.asin(sin_lat)
    # TODO: convert following line to jax
    # new_lat = jnp.min(jnp.array([jnp.max(jnp.array([new_lat, -jnp.pi / 2.0])), jnp.pi / 2.0]))
    new_lat = jnp.clip(new_lat, -jnp.pi / 2.0, jnp.pi / 2.0)
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


def solar_calculator(
    latlng: JaxLatLng, time_seconds: float
) -> Tuple[float, float, float]:
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
    # fraction_of_day = (time_seconds.astype(int) % NUM_SECONDS_PER_DAY) / NUM_SECONDS_PER_DAY
    fraction_of_day = (
        time_seconds % NUM_SECONDS_PER_DAY
    ) / NUM_SECONDS_PER_DAY  # no jit
    # print('A', fraction_of_day)

    # Compute Julian day number from Gregorian calendar.
    year, month, day = timestamp_to_date_components_jax(time_seconds)
    # print('B', year, month, day)
    julian_day_number = (
        367.0 * year
        - jnp.floor(7.0 * (year + jnp.floor((month + 9.0) / 12.0)) / 4.0)
        - jnp.floor(3.0 * (jnp.floor((year + (month - 9.0) / 7.0) / 100.0) + 1.0) / 4.0)
        + jnp.floor(275.0 * month / 9.0)
        + day
        + 1721028.5
    )
    # print('C', julian_day_number)

    # Compute Julian time (in days and in centuries).
    julian_time = julian_day_number + fraction_of_day
    julian_century = (julian_time - 2451545.0) / 36525.0
    # print('D', julian_time, julian_century)

    # Compute solar parameters.
    geometric_mean_long_sun = jnp.radians(
        280.46646 + julian_century * (36000.76983 + julian_century * 0.0003032)
    )
    sin2l0 = jnp.sin(2.0 * geometric_mean_long_sun)
    cos2l0 = jnp.cos(2.0 * geometric_mean_long_sun)
    sin4l0 = jnp.sin(4.0 * geometric_mean_long_sun)
    # print('E', geometric_mean_long_sun, sin2l0, cos2l0, sin4l0)

    geometric_mean_anomaly_sun = jnp.radians(
        357.52911 + julian_century * (35999.05029 - 0.0001537 * julian_century)
    )
    sinm0 = jnp.sin(geometric_mean_anomaly_sun)
    sin2m0 = jnp.sin(2.0 * geometric_mean_anomaly_sun)
    sin3m0 = jnp.sin(3.0 * geometric_mean_anomaly_sun)
    # print('F', geometric_mean_anomaly_sun, sinm0, sin2m0, sin3m0)

    mean_obliquity_of_ecliptic = jnp.radians(
        23.0
        + (
            26.0
            + (
                (
                    21.448
                    - julian_century
                    * (46.815 + julian_century * (0.00059 - julian_century * 0.001813))
                )
                / 60.0
            )
        )
        / 60.0
    )
    obliquity_correction = mean_obliquity_of_ecliptic + jnp.radians(
        0.00256 * jnp.cos(jnp.radians(125.04 - 1934.136 * julian_century))
    )
    # print('G', mean_obliquity_of_ecliptic, obliquity_correction)

    var_y = jnp.tan(obliquity_correction / 2.0) ** 2
    eccentricity_earth = 0.016708634 - julian_century * (
        0.000042037 + 0.0000001267 * julian_century
    )
    # print('H', var_y, eccentricity_earth)

    equation_of_time = 4.0 * (
        var_y * sin2l0
        - 2.0 * eccentricity_earth * sinm0
        + 4.0 * eccentricity_earth * var_y * sinm0 * cos2l0
        - 0.5 * var_y * var_y * sin4l0
        - 1.25 * eccentricity_earth * eccentricity_earth * sin2m0
    )
    # print('I', equation_of_time)

    hour_angle = (
        jnp.radians(
            jnp.fmod(
                1440.0 * fraction_of_day
                + jnp.degrees(equation_of_time)
                + 4.0 * jnp.degrees(latlng.lng),
                1440.0,
            )
        )
        / 4.0
    )
    hour_angle = jnp.where(hour_angle < 0, hour_angle + jnp.pi, hour_angle - jnp.pi)
    # print('J', hour_angle)

    eq_of_center_sun = jnp.radians(
        sinm0 * (1.914602 - julian_century * (0.004817 + 0.000014 * julian_century))
        + sin2m0 * (0.019993 - 0.000101 * julian_century)
        + sin3m0 * 0.000289
    )
    true_long_sun = geometric_mean_long_sun + eq_of_center_sun
    # print('K', eq_of_center_sun, true_long_sun)
    apparent_long_sun = true_long_sun - jnp.radians(
        0.00569 - 0.00478 * jnp.sin(jnp.radians(125.04 - 1934.136 * julian_century))
    )
    declination_sun = jnp.arcsin(
        jnp.sin(obliquity_correction) * jnp.sin(apparent_long_sun)
    )
    # print("L", apparent_long_sun, jnp.sin(obliquity_correction) * jnp.sin(apparent_long_sun), declination_sun)

    zenith_angle = jnp.arccos(
        jnp.sin(latlng.lat) * jnp.sin(declination_sun)
        + jnp.cos(latlng.lat) * jnp.cos(declination_sun) * jnp.cos(hour_angle)
    )
    el_uncorrected_deg = 90.0 - jnp.degrees(zenith_angle)
    # print(f'{zenith_angle=}, {el_uncorrected_deg=}')

    atmospheric_refraction = jnp.where(
        el_uncorrected_deg > 85.0,
        0,
        jnp.where(
            el_uncorrected_deg > 5.0,
            (
                58.1 / jnp.tan(jnp.radians(el_uncorrected_deg))
                - 0.07 / jnp.tan(jnp.radians(el_uncorrected_deg)) ** 3
                + 0.000086 / jnp.tan(jnp.radians(el_uncorrected_deg)) ** 5
            ),
            jnp.where(
                el_uncorrected_deg > -0.575,
                1735.0
                + el_uncorrected_deg
                * (
                    -518.2
                    + el_uncorrected_deg
                    * (
                        103.4
                        + el_uncorrected_deg * (-12.79 + el_uncorrected_deg * 0.711)
                    )
                ),
                -20.772 / jnp.tan(jnp.radians(el_uncorrected_deg)),
            ),
        ),
    )
    # print("el_uncorrected_deg", el_uncorrected_deg)
    # print("atmospheric_refraction", atmospheric_refraction)

    el_deg = el_uncorrected_deg + atmospheric_refraction / 3600.0
    # print("el_deg", el_deg)

    # print(f'({latlng.lat:0.3f}*cos({zenith_angle:0.3f})-sin({declination_sun:0.3f})) / cos({latlng.lat:0.3f}) * sin({zenith_angle:0.3f})')
    # print(f'({jnp.sin(latlng.lat):0.5f}*{jnp.cos(zenith_angle):0.5f}-{jnp.sin(declination_sun):0.5f})) / ({jnp.cos(latlng.lat)} * {jnp.sin(zenith_angle):0.5f})')
    # print(f"In jax solar, latlng.lat: {latlng.lat}, zenith_angle: {zenith_angle}, declination_sun: {declination_sun}")
    # print(f"In jax solar: ({jnp.sin(latlng.lat)*jnp.cos(zenith_angle)}-{jnp.sin(declination_sun)}) / ({jnp.cos(latlng.lat) * jnp.sin(zenith_angle)})")
    # print(f'in jax solar: ({(jnp.sin(latlng.lat)*jnp.cos(zenith_angle)-jnp.sin(declination_sun))}) / ({jnp.cos(latlng.lat) * jnp.sin(zenith_angle)})')
    def compute_cos_az(latlng, zenith_angle, declination_sun):
        epsilon = 1e-5
        near_pole_condition = jnp.abs(jnp.pi / 2 - jnp.abs(latlng.lat)) < epsilon

        cos_az_regular = (
            jnp.sin(latlng.lat) * jnp.cos(zenith_angle) - jnp.sin(declination_sun)
        ) / (jnp.cos(latlng.lat) * jnp.sin(zenith_angle))

        cos_az_limiting = (
            -0.5 * jnp.cos(latlng.lat) * jnp.cos(zenith_angle)
        ) / (jnp.sin(latlng.lat) * jnp.sin(zenith_angle))

        cos_az = lax.cond(
            near_pole_condition,
            lambda _: cos_az_limiting,
            lambda _: cos_az_regular,
            operand=None
        )
        return cos_az
    
    cos_az = compute_cos_az(latlng, zenith_angle, declination_sun)
    
    # print("ccos_az", cos_az)
    az_unwrapped = jnp.arccos(jnp.clip(cos_az, -1.0, 1.0))
    # print("az_unwrapped", az_unwrapped)
    az_deg = jnp.where(
        hour_angle > 0,
        jnp.degrees(az_unwrapped) + 180.0,
        180.0 - jnp.degrees(az_unwrapped),
    )
    # print("az_deg", az_deg)

    flux = 1366.0 * (
        1
        + 0.5
        * (((1 + eccentricity_earth) / (1 - eccentricity_earth)) ** 2 - 1)
        * jnp.cos(geometric_mean_anomaly_sun)
    )

    return el_deg, az_deg, flux


####
# Thermal.py
####


def solar_atmospheric_attenuation(el_deg: float, pressure_altitude_pa: float) -> float:
    """Computes atmospheric attenuation of incoming solar radiation.

    Args:
    el_deg: Solar elevation in degrees.
    pressure_altitude_pa: Balloon's pressure altitude in Pascals.

    Returns:
    attenuation_factor: Solar atmospheric attenuation factor in range [0, 1].
    """

    # Check if solar elevation is within range [-90, 90] deg.
    # if el_deg > 90.0 or el_deg < -90.0:
    #     raise ValueError('solar_atmospheric_attenuation: '
    #                     'Solar elevation out of expected range [-90, 90] deg.')

    # # Check if pressure altitude [Pa] is within range [0, 101325] Pa.
    # if pressure_altitude_pa > 101325.0 or pressure_altitude_pa < 0.0:
    #     raise ValueError('solar_atmospheric_attenuation: '
    #                     'Pressure altitude out of expected range [0, 101325] Pa.')

    # If solar elevation is below min solar horizon return 0.
    # if el_deg < MIN_SOLAR_EL_DEG:
    #     return 0.0

    # Compute airmass.
    tmp_sin_elev = 614.0 * jnp.sin(jnp.radians(el_deg))
    airmass = (
        0.34764
        * (pressure_altitude_pa / 101325.0)
        * (jnp.sqrt(1229.0 + tmp_sin_elev * tmp_sin_elev) - tmp_sin_elev)
    )

    # Compute atmospheric attenuation factor.
    return jnp.where(
        el_deg < MIN_SOLAR_EL_DEG,
        0.0,
        0.5 * (jnp.exp(-0.65 * airmass) + jnp.exp(-0.95 * airmass)),
    )

def balloon_shadow(el_deg: float, panel_height_below_balloon_m: float) -> float:
  """Computes shadowing factor on solar panels due to balloon film.

  Args:
    el_deg: Solar elevation in degrees.
    panel_height_below_balloon_m: Panel location below balloon in meters.

  Returns:
    shadow_factor: Balloon shadowing factor in range [0, 1].
  """
  balloon_radius = 8.69275
  balloon_height = 10.41603

  shadow_el_deg = jnp.degrees(
      jnp.arctan2(
          jnp.sqrt(panel_height_below_balloon_m *
                    (balloon_height + panel_height_below_balloon_m)),
          balloon_radius))

  return jax.lax.cond(el_deg >= shadow_el_deg, lambda _: 0.4392, lambda _: 1.0, operand=None)


def solar_power(el_deg: float, pressure_altitude_pa: float) -> 'power in watts, float':
  """Computes solar power produced by panels on the balloon.

  Args:
    el_deg: Solar elevation in degrees.
    pressure_altitude_pa: Balloon's pressure altitude in Pascals.

  Returns:
    solar_power: Solar power from panels on the balloon [W].
  """

  # Get atmospheric attenuation factor.
  attenuation = solar_atmospheric_attenuation(el_deg, pressure_altitude_pa)

  # Loon balloons have 4 main solar panels mounted at 35deg and hanging at 3.3m
  # below the balloon. There are an additional 2 panels mounted at 65deg
  # hanging at 2.7m below the balloon. All panels have a max power of 210 W.
  power = 210.0 * attenuation * (
      4 * jnp.cos(jnp.radians(el_deg - 35)) * balloon_shadow(el_deg, 3.3) +
      2 * jnp.cos(jnp.radians(el_deg - 65)) * balloon_shadow(el_deg, 2.7))

  return power


def total_absorptivity(absorptivity: float, reflectivity: float) -> float:
    """Compute total internal balloon absorptivity/emissivity factor.

    This function computes total absorptivity or total emissivity. For the
    absorptivity process, the dynamics are as follows:

    --> From the radiation hitting the surface, R is reflected outward, A is
        absorbed, and the rest, T, is "transmitted" through the surface,
        where T = 1 - R - A.
    --> From the amount transmitted through the surface, T, a portion TA is
        absorbed, a portion TR is reflected back into the sphere, and the
        rest (TT) leaves the sphere.
    --> From the amount reflected into the sphere, TR, a portion TRA is
        absorbed, a portion TRR is reflected back into the sphere, and the
        rest (TRT) is lost.
    --> Continuing this process to infinity and adding up all the aborbed
        amounts gives the following:

           A_total = A + TA + TRA + TR^2 A + TR^3 A + ...
                   = A + TA (1 + R + R^2 + R^3 + ...)
                   = A (1 + T / (1 - R))

    Similarly, we can analyze the emissivity process:

    --> From the radiation emitted by the surface, A is emitted outwards where
        E = A = emissivity, and A is emitted inwards (double radiation).
    --> From the inwards amount, AA is re-absorbed, AR is internally reflected
        and AT is emitted through the film.
    --> From the internally reflected amount, ARA is re-absorbed, ARR is
        internally reflected, and ART is emitted through the film.
    --> Continuing this process to infinity and adding up all the outwards
        emissions gives the following:

           E_total = A + AT + ART + AR^2T + AR^3 T + ...
                   = A + AT (1 + R + R^2 + R^3 + ...)
                   = A (1 + T / (1 - R))

    Noting that E_total and A_total are equivalent, we can use this function
    for both incoming and outgoing emissions.

    Args:
      absorptivity: Balloon film's absorptivity/emissivity.
      reflectivity: Balloon film's reflectivity.

    Returns:
      total_absorptivity_factor: Factor of radiation absorbed/emitted by balloon.
    """
    transmisivity = 1.0 - absorptivity - reflectivity
    total_absorptivity_factor = absorptivity * (
        1.0 + transmisivity / (1.0 - reflectivity)
    )
    #   if total_absorptivity_factor < 0.0 or total_absorptivity_factor > 1.0:
    #     raise ValueError(
    #         'total_absorptivity: '
    #         'Computed total absorptivity factor out of expected range [0, 1].')

    return total_absorptivity_factor


def absorptivity_ir(object_temperature_k: float) -> float:
    """Compute balloon IR absorptivity/emissivity given black body temperature.

    This function computes IR absorptivity/emissivity given the radiative object's
    black body temperature. We assume PE01 balloon film and use a linear model.

    Args:
      object_temperature_k: Object's black body temperature [K].

    Returns:
      absorptivity: Absorptivity factor in IR spectrum.
    """
    return _PE01_ABSORPTIVITY_IR_BASE + _PE01_ABSORPTIVITY_IR_D_TEMPERATURE * (
        object_temperature_k - _PE01_ABSORPTIVITY_IR_REF_TEMPERATURE
    )


def black_body_flux_to_temperature(flux: float) -> float:
    """Compute corresponding temperature given black body flux.

    Args:
      flux: Black body's flux [W/m^2].

    Returns:
      temperature_k: Black body's temperature [K].
    """
    return (flux / _STEFAN_BOLTZMAN) ** 0.25


def black_body_temperature_to_flux(temperature_k: float) -> float:
    """Compute corresponding flux given black body temperature.

    Args:
      temperature_k: Black body's temperature [K].

    Returns:
      flux: Black body's flux [W/m^2].
    """
    return _STEFAN_BOLTZMAN * temperature_k**4


def convective_heat_air_factor(
    balloon_radius: float,
    balloon_temperature_k: float,
    ambient_temperature_k: float,
    pressure_altitude_pa: float,
) -> float:
    """Convective heat air factor."""
    viscosity = (
        1.458e-6 * (ambient_temperature_k**1.5) / (ambient_temperature_k + 110.4)
    )
    conductivity = 0.0241 * ((ambient_temperature_k / 273.15) ** 0.9)
    prandtl = 0.804 - 3.25e-4 * ambient_temperature_k
    air_density = (
        pressure_altitude_pa
        * _DRY_AIR_MOLAR_MASS
        / (_UNIVERSAL_GAS_CONSTANT * ambient_temperature_k)
    )

    grashof = (
        9.80665
        * (air_density**2)
        * ((2 * balloon_radius) ** 3)
        / (ambient_temperature_k * (viscosity**2))
    ) * jnp.abs(ambient_temperature_k - balloon_temperature_k)
    rayleigh = prandtl * grashof
    nusselt = 2 + 0.457 * (rayleigh**0.25) + ((1 + 2.69e-8 * rayleigh) ** (1.0 / 12.0))
    k_heat_transfer = nusselt * conductivity / (2 * balloon_radius)

    return k_heat_transfer * (ambient_temperature_k - balloon_temperature_k)


def convective_heat_air_factor(
    balloon_radius: float,
    balloon_temperature_k: float,
    ambient_temperature_k: float,
    pressure_altitude_pa: float,
) -> float:
    """Convective heat air factor."""
    viscosity = (
        1.458e-6 * (ambient_temperature_k**1.5) / (ambient_temperature_k + 110.4)
    )
    conductivity = 0.0241 * ((ambient_temperature_k / 273.15) ** 0.9)
    prandtl = 0.804 - 3.25e-4 * ambient_temperature_k
    air_density = (
        pressure_altitude_pa
        * _DRY_AIR_MOLAR_MASS
        / (_UNIVERSAL_GAS_CONSTANT * ambient_temperature_k)
    )

    grashof = (
        9.80665
        * (air_density**2)
        * ((2 * balloon_radius) ** 3)
        / (ambient_temperature_k * (viscosity**2))
    ) * jnp.abs(ambient_temperature_k - balloon_temperature_k)
    rayleigh = prandtl * grashof
    nusselt = 2 + 0.457 * (rayleigh**0.25) + ((1 + 2.69e-8 * rayleigh) ** (1.0 / 12.0))
    k_heat_transfer = nusselt * conductivity / (2 * balloon_radius)

    return k_heat_transfer * (ambient_temperature_k - balloon_temperature_k)


def d_balloon_temperature_dt(
    balloon_volume: float,
    balloon_mass: float,
    balloon_temperature_k: float,
    ambient_temperature_k: float,
    pressure_altitude_pa: float,
    solar_elevation_deg: float,
    solar_flux: float,
    earth_flux: float,
) -> float:
    """Compute d_balloon_temperature / dt. Assumes PE01 film.

    Args:
      balloon_volume: Balloon volume [m^3].
      balloon_mass: Balloon envelope mass [kg].
      balloon_temperature_k: Balloon temperature [K].
      ambient_temperature_k: Ambient temperature around the balloon [K].
      pressure_altitude_pa: Balloon's pressure altitude [Pa].
      solar_elevation_deg: Solar elevation [deg].
      solar_flux: Solar flux [W/m^2].
      earth_flux: Earth radiation experienced at balloon [W/m^2].

    Returns:
      d_balloon_temperature / dt: Derivative of balloon temperature [K/s].
    """

    # Compute balloon radius and surface area. Assumes spherical balloon.
    balloon_radius = (3 * balloon_volume / (4 * jnp.pi)) ** (1 / 3)
    balloon_area = 4 * jnp.pi * balloon_radius * balloon_radius

    # Compute atmospheric attenuation.
    atm_attenuation = solar_atmospheric_attenuation(
        solar_elevation_deg, pressure_altitude_pa
    )
    # Compute solar radiative heat.
    q_solar = (
        solar_flux
        * atm_attenuation
        * _SOLAR_VIEW_FACTOR
        * balloon_area
        * total_absorptivity(_PE01_ABSORPTIVITY_SOLAR, _PE01_REFLECTIVITY)
    )

    # Compute earth radiative heat.
    q_earth = (
        earth_flux
        * _EARTH_VIEW_FACTOR
        * balloon_area
        * total_absorptivity(
            absorptivity_ir(black_body_flux_to_temperature(earth_flux)),
            _PE01_REFLECTIVITY,
        )
    )

    # Compute balloon emissions given balloon temperature.
    q_emitted = (
        black_body_temperature_to_flux(balloon_temperature_k)
        * balloon_area
        * total_absorptivity(absorptivity_ir(balloon_temperature_k), _PE01_REFLECTIVITY)
    )

    # Compute external convective heat given balloon temperature.
    q_convective = balloon_area * convective_heat_air_factor(
        balloon_radius,
        balloon_temperature_k,
        ambient_temperature_k,
        pressure_altitude_pa,
    )

    # Compute derivative of balloon temperature give total heat loads and mass.
    return (q_solar + q_earth + q_convective - q_emitted) / (
        _PE01_FILM_SPECIFIC_HEAT * balloon_mass
    )


####
# Balloon.py
####


def calculate_superpressure_and_volume(
    mols_lift_gas: float,
    mols_air: float,
    internal_temperature: float,
    pressure: float,
    envelope_volume_base: float,
    envelope_volume_dv_pressure: float,
) -> Tuple[float, float]:
    """Calculates the current superpressure and volume of the balloon.

    Args:
      mols_lift_gas: Mols of helium within the balloon envelope [mols].
      mols_air: Mols of air within the ballonet [mols].
      internal_temperature: The temperature of the gas in the envelope.
      pressure: Ambient pressure of the balloon [Pa].
      envelope_volume_base: The y-intercept for the balloon envelope volume
        model [m^3].
      envelope_volume_dv_pressure: The slope for the balloon envelope volume
        model.

    Returns:
      An (envelope_volume, superpressure) tuple.
    """
    envelope_volume = 0.0
    superpressure = 0.0

    # Compute the unconstrained volume of the balloon which is
    # (n_gas + n_air) * R * T_gas / P_amb. This is the volume the balloon would
    # expand to if the material holding the lift gas didn't give any resistence,
    # e.g., to a first-order approximation a latex weather ballon.
    unconstrained_volume = (
        (mols_lift_gas + mols_air)
        * UNIVERSAL_GAS_CONSTANT
        * internal_temperature
        / pressure
    )

    # if unconstrained_volume <= envelope_volume_base:
    #     # Not fully inflated.
    #     envelope_volume = unconstrained_volume
    #     superpressure = 0.0
    # else:
    #     # System of equations for a fully inflated balloon:
    #     #
    #     #  V = V0 + dv_dp * (P_gas - P_amb)
    #     #  P_gas = n * R * T_gas / V
    #     #
    #     # Solve the quadratic equation for volume:
    #     b = -(envelope_volume_base - envelope_volume_dv_pressure * pressure)
    #     c = -(envelope_volume_dv_pressure * unconstrained_volume * pressure)

    #     envelope_volume = 0.5 * (-b + jnp.sqrt(b * b - 4 * c))
    #     superpressure = pressure * unconstrained_volume / envelope_volume - pressure

    def not_fully_inflated(unconstrained_volume):
        # Case where the balloon is not fully inflated
        return unconstrained_volume, 0.0

    def fully_inflated(unconstrained_volume, envelope_volume_base, envelope_volume_dv_pressure, pressure):
        # Case where the balloon is fully inflated
        b = -(envelope_volume_base - envelope_volume_dv_pressure * pressure)
        c = -(envelope_volume_dv_pressure * unconstrained_volume * pressure)
        
        envelope_volume = 0.5 * (-b + jnp.sqrt(b * b - 4 * c))
        superpressure = pressure * unconstrained_volume / envelope_volume - pressure
        return envelope_volume, superpressure

    # Condition to determine if balloon is not fully inflated
    condition = unconstrained_volume <= envelope_volume_base

    # Use lax.cond for branching
    envelope_volume, superpressure = lax.cond(
        condition,
        lambda op: not_fully_inflated(op[0]),
        lambda op: fully_inflated(op[0], op[1], op[2], op[3]),
        operand = (unconstrained_volume, envelope_volume_base, envelope_volume_dv_pressure, pressure)
    )

    return envelope_volume, superpressure


##########
# acs.py
##########

# Constants
NUM_SECONDS_PER_HOUR = 3600

# Interpolator for pressure ratio to power
_PRESSURE_RATIO_TO_POWER_INTERPOLATOR = RegularGridInterpolator(
    (jnp.array([1.0, 1.05, 1.2, 1.25, 1.35]),),  # Pressure ratio grid
    jnp.array([100.0, 100.0, 300.0, 400.0, 400.0]),  # Power values
    bounds_error=False,
    fill_value=None,  # Extrapolate
)

# Interpolator for pressure ratio and power to efficiency
_PRESSURE_RATIO_POWER_TO_EFFICIENCY_INTERPOLATOR = RegularGridInterpolator(
    (
        jnp.linspace(1.05, 1.35, 13),  # Pressure ratio grid
        jnp.linspace(100.0, 400.0, 4),  # Power grid
    ),
    jnp.array([
        0.4, 0.4, 0.3, 0.2, 0.2, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000,
        0.4, 0.3, 0.3, 0.30, 0.25, 0.23, 0.20, 0.15, 0.12, 0.10, 0.00000, 0.00000, 0.00000,
        0.00000, 0.3, 0.25, 0.25, 0.25, 0.20, 0.20, 0.20, 0.2, 0.15, 0.13, 0.12, 0.11, 0.00000,
        0.23, 0.23, 0.23, 0.23, 0.23, 0.20, 0.20, 0.20, 0.18, 0.16, 0.15, 0.13
    ]).reshape(13, 4),  # Efficiency values reshaped to 2D grid
    bounds_error=False,
    fill_value=None,  # Extrapolate
)

# Functions
def get_most_efficient_power(pressure_ratio: float):
    """Lookup the optimal operating power from static tables.

    Args:
        pressure_ratio: Ratio of (balloon pressure + superpressure) to balloon
        pressure.

    Returns:
        Optimal ACS power at current pressure ratio.
    """
    power = _PRESSURE_RATIO_TO_POWER_INTERPOLATOR(jnp.array([pressure_ratio]))
    return power[0]  # Extract scalar from single-element array


def get_fan_efficiency(pressure_ratio: float, power: float) -> float:
    """Compute efficiency of air flow from current pressure ratio and power.

    Args:
        pressure_ratio: The pressure ratio.
        power: The current power (watts).

    Returns:
        Efficiency of the air flow.
    """
    efficiency = _PRESSURE_RATIO_POWER_TO_EFFICIENCY_INTERPOLATOR(
        jnp.array([[pressure_ratio, power]])
    )
    return efficiency[0] # Extract scalar from single-element array


def get_mass_flow(power: float, efficiency: float) -> float:
    """Compute mass flow based on power and efficiency.

    Args:
        power: The current power (watts).
        efficiency: The fan efficiency.

    Returns:
        Mass flow rate.
    """
    return efficiency * power / NUM_SECONDS_PER_HOUR

