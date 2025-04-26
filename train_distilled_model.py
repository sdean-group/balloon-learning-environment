import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
import numpy as np
from balloon_learning_environment.agents.perciatelli44 import get_distilled_model_features
from balloon_learning_environment.models.jax_perciatelli import DistilledNetwork, get_distilled_model_input_size
from balloon_learning_environment.models.jax_perciatelli import get_perciatelli_params_network
import pickle

def create_train_state(rng, model, learning_rate, input_shape):
    params = model.init(rng, jnp.ones(input_shape))
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


@jax.jit
def train_step(state, batch_inputs, batch_targets):
    def loss_fn(params):
        loss = jnp.mean((state.apply_fn(params, batch_inputs) - batch_targets)**2)

        return loss

    grads = jax.grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads)

def train_distilled_model(num_wind_levels, X_train, y_train, X_val, y_val, num_epochs=1000, batch_size=32, seed=42):
    distilled_input_dim = get_distilled_model_input_size(num_wind_levels)
    distilled_model = DistilledNetwork()
    
    rng = jax.random.PRNGKey(0)
    state = create_train_state(rng, distilled_model, 1e-4, (1, distilled_input_dim))

    np.random.seed(seed)

    num_samples = X_train.shape[0]
    for epoch in range(num_epochs):
        perm = np.random.permutation(num_samples)
        for i in range(0, num_samples, batch_size):
            idx = perm[i:i + batch_size]
            batch_inputs = X_train[idx]
            batch_targets = y_train[idx]
            state = train_step(state, batch_inputs, batch_targets)

        if epoch % 100 == 0 or epoch == num_epochs - 1:
            preds = distilled_model.apply(state.params, X_val)
            loss_val = jnp.mean((preds - y_val) ** 2)
            
            # same = 0
            # for pred, y in zip(preds, y_val):
            #     selection = np.argmax(pred)
            #     real_selection = np.argmax(y)
            #     same += selection == real_selection

            print(f"Epoch {epoch}: Loss = {loss_val:.4f}")

            with open(f'q_training/distilled_model_params-{epoch}.pkl', 'wb') as f:
                pickle.dump(state.params, f)

    return state.params, distilled_model

def load_training_data():
    # Load pickled file
    filepath = 'q_training/perciatelli-training-data'
    X_train, y_train = [], []
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
        for observation, q_vals in data:
            X_train.append(observation)
            y_train.append(q_vals.T.squeeze())
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    return X_train, y_train

if __name__ == "__main__":
    num_wind_levels = 181
    
    training = False

    if training:
        batch_size = 64

        distilled_input_dim = get_distilled_model_input_size(num_wind_levels)

        X_all, y_all = load_training_data()

        split_idx = int(0.8 * len(X_all))
        X_train, X_val = X_all[:split_idx], X_all[split_idx:]
        y_train, y_val = y_all[:split_idx], y_all[split_idx:]

        distilled_params, distilled_model = train_distilled_model(
            num_wind_levels=num_wind_levels,
            X_train=X_train,
            y_train=y_train,
            X_val=X_train,
            y_val=y_train,
            num_epochs=1500,
            batch_size=batch_size,
        )

        #Save params
        with open('q_training/distilled_model_params.pkl', 'wb') as f:
            pickle.dump(distilled_params, f)
    else:
        # Load the distilled model parameters
        
        # Initialize the distilled model
        distilled_model = DistilledNetwork()

        dummy_input = jax.random.uniform(jax.random.PRNGKey(seed=0), (1, get_distilled_model_input_size(num_wind_levels)))
        with open('q_training/distilled_model_params.pkl', 'rb') as f:
            distilled_params = pickle.load(f)

        print(distilled_params.keys())
        print(distilled_params['params'].keys())
        print(distilled_params['params']['Dense_0']['kernel'].shape)
        print(distilled_params['params']['Dense_0']['bias'].shape)
        print(distilled_params['params']['Dense_1']['kernel'].shape) 
        print(distilled_params['params']['Dense_1']['bias'].shape)
        print(distilled_params['params']['Dense_6']['kernel'].shape)
        print(distilled_params['params']['Dense_6']['bias'].shape)

        # print(dummy_input, distilled_model.apply(distilled_params, dummy_input))