import jax.numpy as jnp
from flax import linen as nn
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.core import freeze, unfreeze
import numpy as np
import jax
import pickle
from balloon_learning_environment.utils import units
from balloon_learning_environment.utils import constants
from balloon_learning_environment.env.balloon.jax_balloon import JaxBalloon, JaxBalloonState
from balloon_learning_environment.env.wind_field import JaxWindField



def load_pretrained_params(npz_path, jax_params):
    """
    Loads pretrained TensorFlow weights from an NPZ file and maps them into the JAX model's parameter tree.
    
    Args:
      npz_path: Path to the NPZ file containing TF weights.
      jax_params: The Flax parameter tree obtained by initializing the model.
    
    Returns:
      A frozen parameter tree with the pretrained weights.
    """
    loaded = np.load(npz_path, allow_pickle=True).item()
    params = unfreeze(jax_params)
    
    # Map the TensorFlow weights to the Flax parameters.
    # The TensorFlow variable names come from the frozen graph and are assumed to be:
    # "Online/fully_connected/weights:0" and "Online/fully_connected/biases:0" for the first layer, etc.
    
    params['params']['fc0']['kernel'] = jnp.array(loaded['Online/fully_connected/weights:0'])
    params['params']['fc0']['bias']   = jnp.array(loaded['Online/fully_connected/biases:0'])
    
    params['params']['fc1']['kernel'] = jnp.array(loaded['Online/fully_connected_1/weights:0'])
    params['params']['fc1']['bias']   = jnp.array(loaded['Online/fully_connected_1/biases:0'])
    
    params['params']['fc2']['kernel'] = jnp.array(loaded['Online/fully_connected_2/weights:0'])
    params['params']['fc2']['bias']   = jnp.array(loaded['Online/fully_connected_2/biases:0'])
    
    params['params']['fc3']['kernel'] = jnp.array(loaded['Online/fully_connected_3/weights:0'])
    params['params']['fc3']['bias']   = jnp.array(loaded['Online/fully_connected_3/biases:0'])
    
    params['params']['fc4']['kernel'] = jnp.array(loaded['Online/fully_connected_4/weights:0'])
    params['params']['fc4']['bias']   = jnp.array(loaded['Online/fully_connected_4/biases:0'])
    
    params['params']['fc5']['kernel'] = jnp.array(loaded['Online/fully_connected_5/weights:0'])
    params['params']['fc5']['bias']   = jnp.array(loaded['Online/fully_connected_5/biases:0'])
    
    params['params']['fc6']['kernel'] = jnp.array(loaded['Online/fully_connected_6/weights:0'])
    params['params']['fc6']['bias']   = jnp.array(loaded['Online/fully_connected_6/biases:0'])
    
    params['params']['fc7']['kernel'] = jnp.array(loaded['Online/fully_connected_7/weights:0'])
    params['params']['fc7']['bias']   = jnp.array(loaded['Online/fully_connected_7/biases:0'])
    
    return freeze(params)



class Perciatelli44Network(nn.Module):
    num_actions: int = 3
    num_quantiles: int = 51

    @nn.compact
    def __call__(self, x):
        # x shape: (batch, 1099)
        x = nn.Dense(600, name='fc0')(x)
        x = nn.relu(x)
        x = nn.Dense(600, name='fc1')(x)
        x = nn.relu(x)
        x = nn.Dense(600, name='fc2')(x)
        x = nn.relu(x)
        x = nn.Dense(600, name='fc3')(x)
        x = nn.relu(x)
        x = nn.Dense(600, name='fc4')(x)
        x = nn.relu(x)
        x = nn.Dense(600, name='fc5')(x)
        x = nn.relu(x)
        x = nn.Dense(600, name='fc6')(x)
        x = nn.relu(x)
        x = nn.Dense(153, name='fc7')(x)  # 153 = 3 x 51
        # Reshape output to (batch, 3, 51)
        x = x.reshape((-1, self.num_actions, self.num_quantiles))
        return x

def get_q_values(quantiles):
    """
    Compute expected Q-values by averaging over the quantile estimates.
    
    Args:
      quantiles: jnp.array with shape (batch, num_actions, num_quantiles)
      
    Returns:
      Q-values with shape (batch, num_actions)
    """
    return jnp.mean(quantiles, axis=-1)

def write_weights(sess, path='perciatelli_weights'):
    """ 
    sess = load_perciatelli_session()
    write_weights(sess)
    """
    weights = {}
    for op in sess.graph.get_operations():
        name = f'{op.name}:0'
        try:
            weights_tensor = sess.graph.get_tensor_by_name(name)
            weights_value = sess.run(weights_tensor)
            weights[name] = weights_value
            print('found weights for', name)
        except:
            print('no weights for',name)

    np.save(path, weights, allow_pickle=True)

def get_perciatelli_params_network(path='perciatelli_weights.npy'):
    perciatelli_network = Perciatelli44Network()
    
    # Load pretrained weights (ensure "perciatelli_weights.npz" contains the correct TF variable names)
    jax_params = load_pretrained_params(path, perciatelli_network.init(jax.random.PRNGKey(0), jnp.ones((1, 1099))))

    return jax_params, perciatelli_network

def jax_construct_feature_vector(balloon: JaxBalloon, wind_forecast: JaxWindField, input_size, num_wind_layers):
    feature_vector = jnp.zeros((input_size,))

    x_km = balloon.state.x/1000
    y_km = balloon.state.y/1000

    distance = jnp.sqrt(x_km**2 + y_km**2)
    angle_heading_to_station = jnp.atan2(-x_km, -y_km)

    feature_vector = feature_vector.at[0].set(balloon.state.pressure)
    feature_vector = feature_vector.at[1].set(distance)
    feature_vector = feature_vector.at[2].set(angle_heading_to_station)
    feature_vector = feature_vector.at[3].set(balloon.state.battery_charge/balloon.state.battery_capacity)
    
    # Fill in wind values
    pressure_levels = jnp.linspace(constants.PERCIATELLI_PRESSURE_RANGE_MIN,
                                   constants.PERCIATELLI_PRESSURE_RANGE_MAX,
                                   num_wind_layers)

    def compute_wind_features(i, feature_vector):
        wind_vector = wind_forecast.get_forecast(x_km, y_km, pressure_levels[i], balloon.state.time_elapsed)
        feature_vector = feature_vector.at[4 + i * 3 + 0].set(jnp.sqrt(wind_vector[0]**2 + wind_vector[1]**2))
        feature_vector = feature_vector.at[4 + i * 3 + 1].set(jnp.arctan2(wind_vector[1], wind_vector[0]))
        feature_vector = feature_vector.at[4 + i * 3 + 2].set(pressure_levels[i])
        return feature_vector

    feature_vector = jax.lax.fori_loop(0, num_wind_layers, compute_wind_features, feature_vector)
    return feature_vector

class DistilledNetwork(nn.Module):
    hidden_size: int = 128
    num_actions: int = 3

    @nn.compact
    def __call__(self, x):
        for i in range(6):
            x = nn.Dense(self.hidden_size)(x)
            x = nn.relu(x)
        
        x = nn.Dense(self.num_actions)(x) 
        return x

def get_distilled_model_input_size(num_wind_levels):
    return 4 + 3 * num_wind_levels

def get_distilled_perciatelli(num_wind_levels, filepath='q_training/distilled_model_params-1200.pkl'):
    # Initialize the distilled model
    distilled_model = DistilledNetwork()

    rng = jax.random.PRNGKey(0)
    dummy_input = jnp.ones((1, get_distilled_model_input_size(num_wind_levels)))  # Example observation batch of size 1
    distilled_params = distilled_model.init(rng, dummy_input)
    
    # Load pretrained weights (ensure "distilled_model_params.pkl" contains)
    with open(filepath, 'rb') as f:
      distilled_params = pickle.load(f)

    return distilled_params, distilled_model


if __name__ == "__main__":
    params, model = get_perciatelli_params_network()
    
    x = np.load('observation.npy')
    q_values = get_q_values(model.apply(params, x))
    print(q_values) # Should see "[[127.65921 132.6324  131.29575]]"

    # Is jax-able
    def reward_fn(x, params): 
        model = Perciatelli44Network()
        return jnp.sum(get_q_values(model.apply(params, x)))

    # reward_fn = lambda x, params: jnp.sum(get_q_values(model.apply(params, x)))
    grad_fn = jax.jit(jax.grad(reward_fn, argnums=0))

    print(reward_fn(x, params))
    for i in range(100):
        dx = grad_fn(x, params)
        x += dx/jnp.linalg.norm(dx)
        print(reward_fn(x, params))
