import jax
import jax.numpy as jnp
from flax.training import train_state
import optax
import numpy as np
from balloon_learning_environment.env import balloon_env, features
from balloon_learning_environment.env.balloon.jax_balloon import JaxBalloonState
from balloon_learning_environment.agents import agent, networks
from balloon_learning_environment.eval import suites
from balloon_learning_environment.env.wind_field import JaxWindField
from balloon_learning_environment.utils import run_helpers
import pickle
import gym
from absl import flags
import sys
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import matplotlib.pyplot as plt


@jax.jit
def train_step(state, batch_inputs, batch_targets):
    def mse(params):
        # Define the squared loss for a single pair (x,y)
        def squared_error(x, y):
            pred = state.apply_fn(params, x)
            #jax.debug.print("pred: {pred}, target: {target}", pred=pred, target=y)
            return jnp.inner(y-pred, y-pred) / 2.0
        # Vectorize the previous to compute the average of the loss on all samples.
        return jnp.mean(jax.vmap(squared_error)(batch_inputs,batch_targets), axis=0)


    grads = jax.grad(mse)(state.params)
    params= state.params
    loss_grad_fn = jax.value_and_grad(mse)
    opt_state = state.opt_state
    for i in range(11):
        loss_val, grads = loss_grad_fn(params)
        updates, opt_state = state.tx.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        #if i % 10 == 0:
            #jax.debug.print('Loss step {i}: {loss_val}', i=i, loss_val=loss_val)

    
    # jax.debug.print('---------------------------------------------------')
    # jax.debug.print("grad: grads = {grads}", grads=grads)
    return state.apply_gradients(grads=grads)


def create_train_state(rng, model, input_shape, tx):
    params = model.init(rng, jnp.ones(input_shape))
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

def train_behavioral_clone(X_train, y_train, X_val, y_val, num_layers, hidden_dim,bc_input_dim,batch_size:int, num_epochs=100, seed=42, learning_rate=0.01, momentum=0.9 ):
    bc_model = networks.PolicyNetwork(num_layers=num_layers, hidden_dim=hidden_dim, input_dim=bc_input_dim)
    
    rng = jax.random.PRNGKey(0)

    np.random.seed(seed)

    num_samples = X_train.shape[0]
    print(num_samples)
    print(batch_size)
    print(y_train.shape)
    print(X_train.shape)
    print(y_val.shape)
    print(X_val.shape)
    # 1e-10 is too slow. 1e-3 loss goes down at first but then goes up
    lr = optax.linear_schedule(
        init_value=1e-6,
        end_value=1e-7,
        transition_steps=num_samples/batch_size * num_epochs,
    )
    tx = optax.adam(lr)
    state = create_train_state(rng, bc_model, (1, bc_input_dim),tx )
    
    
    for epoch in range(num_epochs):
        perm = np.random.permutation(num_samples)
        # train on batches
        for i in range(0, num_samples, batch_size):
            idx = perm[i:i + batch_size]
            batch_inputs = X_train[idx]
            batch_targets = y_train[idx]
            state = train_step(state, batch_inputs, batch_targets)
        #jax.debug.print('Done training for epoch {epoch}', epoch=epoch)
        # evaluate again on whole training set
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            state = train_step(state, batch_inputs, batch_targets)
            
            # Predict on train and val sets
            val_preds = bc_model.apply(state.params, X_val)
            train_preds = bc_model.apply(state.params, X_train)

            if epoch == num_epochs-1:
                # Training set
                plt.figure(figsize=(6, 4))
                plt.plot(train_preds, label="Predictions")
                plt.plot(y_train, label="Actual")
                plt.xlabel('Steps')
                plt.ylabel('Predicted Action')
                plt.title(f'Training Predictions vs Actuals after {num_epochs} epochs')
                plt.grid(True)
                plt.legend()
                plt.tight_layout()
                plt.show()

                if (len(X_train) > 1):
                    # Validation set
                    plt.figure(figsize=(6, 4))
                    plt.plot(val_preds, label="Predictions")
                    plt.plot(y_val, label="Actual")
                    plt.xlabel('Steps')
                    plt.ylabel('Predicted Action')
                    plt.title(f'Validation Predictions vs Actuals after {num_epochs} epochs')
                    plt.grid(True)
                    plt.legend()
                    plt.tight_layout()
                    plt.show()




            if (len(X_val) < 1):
                print('using training loss')
                loss_val = jnp.mean((val_preds - y_val) ** 2)
            else:
                print('using validation loss')
                loss_val = jnp.mean((train_preds - y_train) ** 2)
            print(f"Epoch {epoch}: Loss = {loss_val:.4f}")

            # with open(f'bc_model_params-{epoch}.pkl', 'wb') as f:
            #     pickle.dump(state.params, f)

    return state.params, bc_model

def observation_to_feature(observation, wind_field):
    """ Need to make observation from JaxBalloonState to np.array and also cutting out constants so we don't learn them"""
    modified_obs = JaxBalloonState.get_jax_features(observation)
    #print(f"type of balloon state is {type(balloon_state)}")

    # add wind data to each point. [lat, lon] as jax floats from grid_based_wind_field.py get_forecast
    # NOTE: use get_forecast_column instead

    #pressure is 0 - 2380 in general
    # more specifically use pressure_range_builder.py
    #winds = wind_field.get_col

    wind = wind_field.get_forecast(modified_obs[2]/1000, modified_obs[3]/1000,modified_obs[4],modified_obs[17])
    final_obs = jnp.concatenate((modified_obs, wind))
    #print(f"length of final obs {final_obs}, {len(final_obs)}")
    return final_obs
    # TODO: add info about safety violations to observation? maybe only do this if it is consistently violating thigns. balloon status tracks if burst or out of power but not violations
    # but my mpc version ensures 0 pretty much always so it would be a constant. maybe track the agent's calculated expected charge variable instead, since thats technically not an agent thing thats a balloon stat

def collect_expert_pairs(expert: agent.Agent, env: balloon_env.BalloonEnv, episode_length: int, seed):
  """ Similar to run_one_episode in train_lib.py but also tracks state.
        
        NOTE: dimension of observations is 20
  """
  env.seed(seed)
  observation = env.reset()
  expert.update_forecast(env.get_wind_forecast())
  expert.update_atmosphere(env.get_atmosphere())
  wind_field: JaxWindField = env.get_wind_forecast().to_jax_wind_field()
  total_reward = 0
  step_count = 0
  observations = []
  actions = []
  # normal states are BalloonState so it has a bit more info but observations is JaxBalloonState
  action = expert.begin_episode(observation)
  print(f"{observation.x, observation.y}")
  balloon_state = env.get_simulator_state().balloon_state
  print(f"{balloon_state.x, balloon_state.y}")
  final_obs = observation_to_feature(observation, wind_field)
  print(f"{final_obs[2], final_obs[3]}")
  print("---------------------")
  observations.append(final_obs)
  actions.append(action)

  while step_count < episode_length:
    observation, reward, is_done, info = env.step(action)
    action = expert.step(reward, observation)
    
    final_obs = observation_to_feature(observation, wind_field)
    observations.append(final_obs) 
    actions.append(action)

    total_reward += reward
    step_count += 1

    if is_done:
      break

  expert.end_episode(reward, is_done)
  print(f"Collected {len(actions)} action/state pairs from expert.")
#   print(actions[:3])
#   print(observations[:3])
  states_np = jnp.array(observations, dtype=jnp.float32)
  actions_np = jnp.array(actions, dtype=jnp.float32)
  return total_reward, states_np, actions_np

def collect_training_data(expert: agent.Agent, env: balloon_env.BalloonEnv, eval_suite: suites.EvaluationSuite):
  """ Collect expert demonstrations over multiple seeds to create a varied training set. """
  expert_total_reward_list = [] # does this do anything
  expert_states_list = []
  expert_actions_list = []
  for seed in eval_suite.seeds:
      expert_total_reward, expert_states_np, expert_actions_np = collect_expert_pairs(expert, env, eval_suite.max_episode_length, seed)

      with open("expert_states.txt", "w") as f:
        f.writelines([str(state) + "\n" for state in expert_states_np] )
        f.write(f"EXPERT TOTAL REWARD IS {expert_total_reward}")
      with open("expert_actions.txt", "w") as f:
        f.writelines([str(action) + "\n" for action in expert_actions_np])

      
      expert_total_reward_list.append(expert_total_reward)
      expert_states_list.append(expert_states_np)
      expert_actions_list.append(expert_actions_np)

  expert_states_list = np.array(expert_states_list)
  expert_actions_list = np.array(expert_actions_list)
  print('YOOOOOOOOOO')
  print(expert_states_list.shape)
  print(expert_actions_list.shape)
#   expert_states_np = np.concatenate(expert_states_list) # X_train
#   print(expert_states_np.shape)
#   expert_actions_np = np.concatenate(expert_actions_list) # Y_train
#   print(expert_actions_np.shape)

  return expert_states_list, expert_actions_list

def run_training_bc(batch_size:int, num_layers:int, hidden_dim:int, bc_input_dim:int, suite_name:str=None, num_epochs:int=100):
    try:
        data = np.load('tiny_expert_demos.npz')
        X_train = data['X_train']
        y_train = data['y_train']
        X_val = data['X_val']
        y_val = data['y_val']
    except OSError:
        print('training data file doesnt exist')
        X_train = []
        y_train = []
        X_val = []
        y_val = []

    if suite_name is not None:
        fc_factory = features.MPC2Features
        wf_factory = run_helpers.get_wind_field_factory('generative')
        env = gym.make('BalloonLearningEnvironment-v0',
                        wind_field_factory=wf_factory,
                        renderer=None,
                        feature_constructor_factory=fc_factory)

        mpc4_agent = run_helpers.create_agent(
        'mpc4',
        env.action_space.n,
        observation_shape=env.observation_space.shape)
        suite = suites.get_eval_suite(suite_name)

        X_all, y_all = collect_training_data(mpc4_agent, env, suite)

        split_idx = int(0.8 * len(X_all))
        print(f"split indx is {split_idx}")
        # train and validation set
        X_train_new, X_val_new = X_all[:split_idx], X_all[split_idx:]
        y_train_new, y_val_new = y_all[:split_idx], y_all[split_idx:]
        print('SPLITTED')
        print(X_train_new.shape)
        print(y_train_new.shape)
        print(X_val_new.shape)
        print(y_val_new.shape)

        X_train = jnp.concatenate(X_train_new)
        y_train = jnp.concatenate(y_train_new)
        X_val = jnp.concatenate(X_val_new)
        y_val = jnp.concatenate(y_val_new)
        print('JOINED')
        print(X_train.shape)
        print(y_train.shape)
        print(X_val.shape)
        print(y_val.shape)

        # X_train= X_train_new if len(X_train) == 0 else jnp.concatenate([X_train, X_train_new])
        # y_train= y_train_new if len(y_train) == 0 else jnp.concatenate([y_train, y_train_new])
        # X_val= X_val_new if len(X_val) == 0 else jnp.concatenate([X_val, X_val_new])
        # y_val= y_val_new if len(y_val) == 0 else jnp.concatenate([y_val, y_val_new])

        np.savez('tiny_expert_demos.npz', X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)

    #normalize inputs
    feature_means = jnp.mean(X_train, axis=0)
    feature_std_devs = jnp.std(X_train, axis=0)
    feature_std_devs = feature_std_devs.at[feature_std_devs==0].set(0.01)
    X_train = (X_train - feature_means) / feature_std_devs

    feature_means = jnp.mean(X_val, axis=0)
    feature_std_devs = jnp.std(X_val, axis=0)
    feature_std_devs = feature_std_devs.at[feature_std_devs==0].set(0.01)
    X_val = (X_val - feature_means) / feature_std_devs


    bc_params, bc_model = train_behavioral_clone(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        bc_input_dim=bc_input_dim,
        batch_size=batch_size,
        num_epochs=num_epochs,
    )

    #Save params
    with open('bc_model_params2.pkl', 'wb') as f:
        pickle.dump(bc_params, f)

    print('Finished training behavior cloning network')
    return bc_params,bc_model

if __name__ == "__main__":

    for name in list(flags.FLAGS):
        delattr(flags.FLAGS,name)
    flags.DEFINE_string('train_suite', None, 'suite name to train mpc expert on')
    FLAGS = flags.FLAGS

    FLAGS(sys.argv)

    #doing the paper versions has the same down n up behavior for loss
    
    bc_params, bc_model = run_training_bc(64,4, 128,20, num_epochs=1000, suite_name=FLAGS.train_suite)

    
    # print(f"KERNEL SHAPE {bc_params['params']['Dense_0']['kernel'].shape}")
    # print(f"KERNEL SHAPE {bc_params['params']['Dense_1']['kernel'].shape}")
    # print(f"KERNEL SHAPE {bc_params['params']['Dense_2']['kernel'].shape}")
    # print(f"KERNEL SHAPE {bc_params['params']['Dense_3']['kernel'].shape}")


    #if state aliasing happens, then need new state representation or new network architecture like RNNs

    """
    with 10 repeated gradients:
    Epoch 0: Loss = 1.5888
    Epoch 10: Loss = 1.5082
    Epoch 20: Loss = 1.4309
    Epoch 30: Loss = 1.3962
    Epoch 40: Loss = 1.3482
    Epoch 50: Loss = 1.2845
    Epoch 60: Loss = 1.1781
    Epoch 70: Loss = 1.0408
    Epoch 80: Loss = 0.8270
    Epoch 90: Loss = 0.5414
    Epoch 99: Loss = 0.5013
Finished training behavior 
    
    """

    """
    with 5. 2 does bad
    Epoch 0: Loss = 1.3043
Epoch 10: Loss = 1.2654
Epoch 20: Loss = 1.1776
Epoch 30: Loss = 1.1302
Epoch 40: Loss = 1.0814
Epoch 50: Loss = 1.0276
Epoch 60: Loss = 0.9084
Epoch 70: Loss = 0.7628
Epoch 80: Loss = 0.5527
Epoch 90: Loss = 0.5083
Epoch 99: Loss = 0.5086
    
    
    """
    