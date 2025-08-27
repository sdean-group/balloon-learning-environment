import jax.numpy as jnp
from balloon_learning_environment.env.balloon.jax_balloon import JaxBalloon, JaxBalloonState
from balloon_learning_environment.env.wind_field import JaxWindField

class ValueNetworkFeature:
    """ interface for computing feature vectors given a balloon and wind forecast for a value network """

    def __init__(self):
        pass

    @property
    def num_input_dimensions(self):
        raise NotImplementedError('Implement num_input_dimensions')

    def compute(self, balloon: JaxBalloon, wind_forecast: JaxWindField):
        raise NotImplementedError('Implement computing')

class BasicValueNetworkFeature(ValueNetworkFeature):
    """ this is for testing that all the jax features work """
    def __init__(self): pass

    @property
    def num_input_dimensions(self):
        return 5

    def compute(self, balloon: JaxBalloon, wind_forecast: JaxWindField) -> jnp.ndarray:
        wind_vector = wind_forecast.get_forecast(balloon.state.x/1000, balloon.state.y/1000, balloon.state.pressure, balloon.state.time_elapsed)
        return jnp.array([ 
            balloon.state.x/1000, 
            balloon.state.y/1000, 
            balloon.state.pressure, 
            wind_vector[0], 
            wind_vector[1] ])
