from balloon_learning_environment.env.balloon.standard_atmosphere import Atmosphere, JaxAtmosphere
from balloon_learning_environment.utils import units
from atmosnav.utils import alt2p as alt2p_atm
from atmosnav.utils import p2alt
import jax
import time

key = jax.random.PRNGKey(seed=0)
atmosphere = Atmosphere(key)
def alt2p_ble(altitutde):
    return atmosphere.at_height(units.Distance(km=altitutde)).pressure

jaxmosphere = atmosphere.to_jax_atmopshere()
def alt2p_jax_ble(altitutde):
    return jaxmosphere.at_height(units.Distance(km=altitutde).meters).pressure

def p2alt_ble(pressure):
    return atmosphere.at_pressure(pressure).height.km

def p2alt_jax_ble(pressure):
    return jaxmosphere.at_pressure(pressure).height.km

"""
 alt2p_ble: 108746.899  alt2p_jax_ble: 108746.930 
 alt2p_ble: 55093.658  alt2p_jax_ble: 55093.668 
 alt2p_ble: 36782.261  alt2p_jax_ble: 36782.262 
 alt2p_ble: 31916.909  alt2p_jax_ble: 31916.912 
 alt2p_ble: 27586.348  alt2p_jax_ble: 27586.354 
 alt2p_ble: 23744.393  alt2p_jax_ble: 23744.391 
 alt2p_ble: 20347.699  alt2p_jax_ble: 20347.697 
 alt2p_ble: 17355.661  alt2p_jax_ble: 17355.662 
 alt2p_ble: 12436.246  alt2p_jax_ble: 12436.250 
 alt2p_ble: 8712.367  alt2p_jax_ble: 8712.366 
 alt2p_ble: 5134.426  alt2p_jax_ble: 5134.426 
"""
def test_altitudes():
    tests=[
        alt2p_ble,
        jax.jit(alt2p_jax_ble)]
    altitudes = [ -0.6, 5, 8, 9 , 10, 11, 12, 13, 15, 17 ,20]

    for altitude in altitudes:
        for test in tests:
            print(f" {test.__name__}: {test(altitude):.3f}", end=" ")
            
        print()
    
def test_pressures():
    tests = [
        p2alt_ble,
        jax.jit(p2alt_jax_ble)
    ]

    pressures = [108746.930, 55093.668, 36782.262, 31916.912, 27586.354, 23744.391, 20347.697, 17355.662, 12436.250, 8712.366, 5134.426 ]
    for pressure in pressures:
        for test in tests:
            print(f" {test.__name__}: {test(pressure):.3f}", end=" ")
        print()

test_altitudes()
test_pressures()
