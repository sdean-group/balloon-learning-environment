import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

def numpy_version(obliquity_correction, apparent_long_sun):
    declination_sun = np.arcsin(
      np.sin(obliquity_correction) * np.sin(apparent_long_sun))
    return declination_sun

def jnp_version(obliquity_correction, apparent_long_sun):
    declination_sun = jnp.arcsin(
      jnp.sin(obliquity_correction) * jnp.sin(apparent_long_sun))
    return declination_sun

tests = [ (0.409080651134202, 157.10346628729434), (0.4090810491912244, 160.237885151924), (0.4090807783600018, 161.79407168447165), (0.40908100951762, 158.66099390508788) ] 
functions = [numpy_version, jnp_version ]

for i, test in enumerate(tests):
    print(f'Test {i+1}:')
    print(f"\tinput = {test}")
    for func in functions:
        print(f'\t{func.__name__}={func(*test)}')