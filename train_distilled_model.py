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

import argparse

from functools import partial

import os
import time

def create_train_state(rng, model, learning_rate, input_shape):
    params = model.init(rng, jnp.ones(input_shape))
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def train_step(state, batch_inputs, batch_targets):
    
    def loss_fn(params):
        loss = jnp.mean((state.apply_fn(params, batch_inputs) - batch_targets)**2)

        return loss

    grads = jax.grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads)

@partial(jax.jit, static_argnames=('batch_size',), donate_argnums=(1, ))
def train_epoch(key, state, X_train, y_train, batch_size):
    num_samples = X_train.shape[0]

    # Split the key for reproducibility
    key, subkey = jax.random.split(key)
    perm = jax.random.permutation(subkey, num_samples)

    def body_fn(state, i):
        # Get batch indices from perm[i:i+batch_size] using dynamic_slice
        idx = jax.lax.dynamic_slice(perm, (i,), (batch_size,))

        # Gather inputs and targets using jnp.take with axis=0
        batch_inputs = jnp.take(X_train, idx, axis=0)
        batch_targets = jnp.take(y_train, idx, axis=0)

        state = train_step(state, batch_inputs, batch_targets)
        return state, None

    num_batches = (num_samples + batch_size - 1) // batch_size
    batch_starts = np.arange(0, num_batches * batch_size, batch_size)

    state, _ = jax.lax.scan(body_fn, state, batch_starts)
    return key, state

@jax.jit
def evaluate_model(state, X_train, y_train, X_val, y_val):
    # Predictions
    # train_preds = state.apply_fn(state.params, X_train)
    val_preds = state.apply_fn(state.params, X_val)

    # MSE loss
    # train_loss= jnp.mean((train_preds - y_train) ** 2)
    val_loss = jnp.mean((val_preds - y_val) ** 2)

    return val_loss, val_loss

def train_distilled_model(
    num_wind_levels,
    X_train_host, y_train_host,  # host-side (numpy)
    X_val_host, y_val_host,
    output_dir="./",
    learning_rate=1e-4,
    num_epochs=1000,
    batch_size=128,
    seed=42
):
    import math

    # Set up model and training state
    input_dim = get_distilled_model_input_size(num_wind_levels)
    model = DistilledNetwork()
    rng = jax.random.PRNGKey(seed)
    state = create_train_state(rng, model, learning_rate, (1, input_dim))

    # Convert validation data ONCE
    X_val = jax.device_put(X_val_host)
    y_val = jax.device_put(y_val_host)

    num_samples = X_train_host.shape[0]
    num_batches = math.ceil(num_samples / batch_size)

    for epoch in range(num_epochs):
        start_time = time.time()

        # Shuffle indices ON CPU
        rng, subkey = jax.random.split(rng)
        perm = np.random.permutation(num_samples)

        # Train over mini-batches
        for i in range(num_batches):
            idx = perm[i * batch_size : (i + 1) * batch_size]

            # Slice & move to device
            batch_x = jax.device_put(X_train_host[idx])
            batch_y = jax.device_put(y_train_host[idx])

            state = train_step(state, batch_x, batch_y)

        epoch_time = time.time() - start_time

        # Evaluate every N epochs
        if epoch % 1 == 0 or epoch == num_epochs - 1:
            val_loss = evaluate_model(state.params, state.apply_fn, X_val, y_val)
            print(f"[Epoch {epoch}] val_loss={val_loss:.4f}  (epoch took {epoch_time:.1f}s)")

            save_path = os.path.join(output_dir, f"distilled_model_params-{epoch}.pkl")
            with open(save_path, 'wb') as f:
                pickle.dump(state.params, f)

    return state.params, model


def load_small_training_data():
    """ this function is for the old training data on 100 seeds 
    that was collected with an older feature vector and file structure """
    # Load pickled file
    filepath = 'q_training/perciatelli-training-data'
    X_train, y_train = [], []
    with open(filepath, 'rb') as f:
        data = pickle.load(f)

    observations, q_vals = zip(*data)
    X_train = np.stack(observations)
    y_train = np.stack([q.T.squeeze() for q in q_vals])

    # Extract balloon magnitudes and directions
    balloon_mag = X_train[:, 1]
    balloon_dir = X_train[:, 2]

    # Compute to_origin_vec for each sample (shape: [num_samples, 2])
    to_origin_vec = -np.stack([
        balloon_mag * np.cos(balloon_dir),
        balloon_mag * np.sin(balloon_dir)
    ], axis=-1)

    # Normalize
    to_origin_vec /= np.linalg.norm(to_origin_vec, axis=1, keepdims=True)

    # Extract wind magnitudes and directions (shape: [num_samples, 181])
    wind_mag = X_train[:, 4::3]
    wind_dir = X_train[:, 5::3]

    # Compute wind vectors (shape: [num_samples, 181, 2])
    wind_vec = np.stack([
        wind_mag * np.cos(wind_dir),
        wind_mag * np.sin(wind_dir)
    ], axis=-1)

    # Normalize wind vectors
    wind_vec /= np.linalg.norm(wind_vec, axis=-1, keepdims=True)

    # Compute dot product with to_origin_vec (broadcasted)
    # to_origin_vec shape: [num_samples, 1, 2] -> broadcasted to [num_samples, 181, 2]
    dot_products = np.sum(wind_vec * to_origin_vec[:, None, :], axis=-1)

    # Clip and compute arccos
    relative_angles = np.arccos(np.clip(dot_products, -1.0, 1.0))

    # Replace wind_dir in X_train with the computed relative angles
    X_train[:, 5::3] = relative_angles
 
    return X_train, y_train

def load_training_data(X_train_filepath, y_train_filepath):
    X_train_file = open(X_train_filepath, 'rb')
    X_train = pickle.load(X_train_file)
    X_train_file.close()

    y_train_file = open(y_train_filepath, 'rb')
    y_train = pickle.load(y_train_file)
    y_train_file.close()

    return np.array(X_train), np.array(y_train).squeeze()

def load_and_concatenate_training_data(file_pairs):
    X_list = []
    y_list = []

    for X_file, y_file in file_pairs:
        X, y = load_training_data(X_file, y_file)
        X_list.append(X)
        y_list.append(y)

    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)

    return X_all, y_all

def parse_args():
    parser = argparse.ArgumentParser(description="Train a model with a specified learning rate.")
    parser.add_argument(
        '--learning-rate',
        type=float,
        required=True,
        help='Learning rate for the optimizer (e.g., 0.001)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        required=True,
        help='Batch size for the optimizer (e.g. )'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory for checkpoint models'
    )
    return parser.parse_args()

if __name__ == "__main__":

    # Read arguments (and print)
    args = parse_args()
    print("Arguments:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")

    print("Starting train distilled model...")
    num_wind_levels = 181
    
    training = True
    if training:

        distilled_input_dim = get_distilled_model_input_size(num_wind_levels)

        print("Loading training data...")
        start = time.time()

        training_data = [
            ("q_training/1746236230344-X_train.pkl", "q_training/1746236230344-y_train.pkl"),
            # ("q_training/1746237737825-X_train.pkl", "q_training/1746237737825-y_train.pkl"),
            # ("q_training/1746240501445-X_train.pkl", "q_training/1746240501445-y_train.pkl"),
            # ("q_training/1746242980434-X_train.pkl", "q_training/1746242980434-y_train.pkl"),
            # ("q_training/1746245471029-X_train.pkl", "q_training/1746245471029-y_train.pkl"),
            # ("q_training/1746248364122-X_train.pkl", "q_training/1746248364122-y_train.pkl"),
            # ("q_training/1746249380619-X_train.pkl", "q_training/1746249380619-y_train.pkl"),
            # ("q_training/1746249603584-X_train.pkl", "q_training/1746249603584-y_train.pkl"),
            # ("q_training/1746249883782-X_train.pkl", "q_training/1746249883782-y_train.pkl"),
            # ("q_training/1746254247520-X_train.pkl", "q_training/1746254247520-y_train.pkl"),
        ]
        X_all, y_all = load_and_concatenate_training_data(training_data)
        print("Data found: ", X_all.shape, y_all.shape)

        print("took ", time.time() - start, "s" )

        split_idx = int(0.85 * len(X_all))
        X_train, X_val = X_all[:split_idx], X_all[split_idx:]
        y_train, y_val = y_all[:split_idx], y_all[split_idx:]

        print("training")
        distilled_params, distilled_model = train_distilled_model(
            num_wind_levels=num_wind_levels,
            X_train_host=X_train,
            y_train_host=y_train,
            X_val_host=X_val,
            y_val_host=y_val,
            learning_rate=args.learning_rate,
            num_epochs=10000,
            batch_size=args.batch_size,
            output_dir=args.output_dir
        )

        #Save params
        final_save_path = os.path.join(args.output_dir, 'distilled_model_params-final.pkl')
        with open(final_save_path, 'wb') as f:
            pickle.dump(distilled_params, f)
    else:
        
        # print("loading")
        # start = time.time()
        # X_train, y_train = load_training_data("q_training/1746236230344-X_train.pkl", "q_training/1746236230344-y_train.pkl")
        # print("took ", time.time() - start, "s" )

        # Load the distilled model parameters
        
        # Initialize the distilled model
        distilled_model = DistilledNetwork()

        rng = jax.random.PRNGKey(seed=0)
        dummy_input = jax.random.uniform(rng, (1, get_distilled_model_input_size(num_wind_levels)))
        params = distilled_model.init(rng, dummy_input)
        # with open('q_training/distilled_model_params.pkl', 'rb') as f:
        #     distilled_params = pickle.load(f)


        print(sum(jnp.size(p) for p in jax.tree_util.tree_leaves(params)))

        # print(distilled_params.keys())
        # print(distilled_params['params'].keys())
        # print(distilled_params['params']['Dense_0']['kernel'].shape)
        # print(distilled_params['params']['Dense_0']['bias'].shape)
        # print(distilled_params['params']['Dense_1']['kernel'].shape) 
        # print(distilled_params['params']['Dense_1']['bias'].shape)
        # print(distilled_params['params']['Dense_6']['kernel'].shape)
        # print(distilled_params['params']['Dense_6']['bias'].shape)

        # print(dummy_input, distilled_model.apply(distilled_params, dummy_input))