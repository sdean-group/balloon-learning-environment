import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
import numpy as np
from balloon_learning_environment.models import jax_perciatelli
import pickle

class DistilledNetwork(nn.Module):
    hidden_size: int = 128
    num_actions: int = 3
    # num_quantiles: int = 51

    @nn.compact
    def __call__(self, x):
        
        for i in range(6):
            x = nn.Dense(self.hidden_size)(x)
            x = nn.relu(x)
        

        # NOT USING QUANTILES
        # x = nn.Dense(self.num_actions * self.num_quantiles)(x)
        # return x.reshape((-1, self.num_actions, self.num_quantiles))
        
        x = nn.Dense(self.num_actions)(x) 
        return x

def get_distilled_model_input_size(num_wind_levels):
    return 4 + 3 * num_wind_levels

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

def train_distilled_model(num_wind_levels, X_train, y_train, X_val, y_val, num_epochs=1000, batch_size=32):
    distilled_input_dim = get_distilled_model_input_size(num_wind_levels)
    distilled_model = DistilledNetwork()
    
    rng = jax.random.PRNGKey(0)
    state = create_train_state(rng, distilled_model, 1e-4, (1, distilled_input_dim))

    num_samples = X_train.shape[0]
    for epoch in range(num_epochs):
        perm = np.random.permutation(num_samples)
        for i in range(0, num_samples, batch_size):
            idx = perm[i:i + batch_size]
            batch_inputs = X_train[idx]
            batch_targets = y_train[idx]
            state = train_step(state, batch_inputs, batch_targets)

        if epoch % 100 == 0:
            preds = distilled_model.apply(state.params, X_val)
            loss_val = jnp.mean((preds - y_val) ** 2)
            
            same = 0
            for pred, y in zip(preds, y_val):
                selection = np.argmax(pred)
                real_selection = np.argmax(y)
                same += selection == real_selection
            
            print(f"Epoch {epoch}: Loss = {loss_val:.4f}, matched = {same/len(y_val):.4f}")

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
    batch_size = 64

    num_wind_levels = 181
    distilled_input_dim = get_distilled_model_input_size(num_wind_levels)

    X_train, y_train = load_training_data()
    print(X_train.shape, y_train.shape)

    distilled_params, distilled_model = train_distilled_model(
        num_wind_levels=num_wind_levels,
        X_train=X_train,
        y_train=y_train,
        X_val=X_train,
        y_val=y_train,
        num_epochs=1000,
        batch_size=batch_size,
    )
