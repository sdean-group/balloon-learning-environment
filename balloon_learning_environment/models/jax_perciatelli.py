
from balloon_learning_environment.agents import agent
from balloon_learning_environment.agents.perciatelli44 import load_perciatelli_session
from balloon_learning_environment.models import models
import tensorflow as tf

import jax.numpy as jnp
from flax import linen as nn
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.core import freeze, unfreeze
import numpy as np
import jax


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

def get_perciatelli_params_network(batch_size=1, path='perciatelli_weights.npy'):
    rng = jax.random.PRNGKey(0)
    dummy_input = jnp.ones((batch_size, 1099))  # Example observation batch of size 1
    perciatelli_network = Perciatelli44Network()
    
    # Initialize the model parameters
    jax_params = perciatelli_network.init(rng, dummy_input)
    # Load pretrained weights (ensure "perciatelli_weights.npz" contains the correct TF variable names)
    jax_params = load_pretrained_params(path, jax_params)

    return jax_params, perciatelli_network



if __name__ == "__main__":
    params, model = get_perciatelli_params_network()
    
    x = np.load('observation.npy')
    q_values = get_q_values(model.apply(params, x))
    print(q_values) # Should see "[[127.65921 132.6324  131.29575]]"

    # Is jax-able
    reward_fn = lambda x: jnp.sum(get_q_values(model.apply(params, x)))
    grad_fn = jax.jit(jax.grad(reward_fn))

    print(reward_fn(x))
    for i in range(10):
        dx = grad_fn(x)
        x += dx/jnp.linalg.norm(dx)
        print(reward_fn(x))
