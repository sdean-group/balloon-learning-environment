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
    return jaxmosphere.at_height(units.Distance(km=altitutde)).pressure


tests=[
    # jax.jit(alt2p_atm),
      alt2p_ble, 
      jax.jit(alt2p_jax_ble)]
altitudes = [ -0.6, 5, 8, 9 , 10, 11, 12, 13, 15, 17 ,20]

for altitude in altitudes:
    for test in tests:
        start = time.time()
        pressure = test(altitude)
        took = time.time() - start
        # print(f"[{took}s] {test.__name__}: {test(altitude):.3f}", end=" ")
        print(f" {test.__name__}: {test(altitude):.3f}", end=" ")
        # print(f"{p2alt(pressure)}", end=" ")
    print()