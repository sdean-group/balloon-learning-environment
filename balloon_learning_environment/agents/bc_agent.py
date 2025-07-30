from balloon_learning_environment.agents import agent
from balloon_learning_environment.agents import networks
from balloon_learning_environment.env import balloon_env
from balloon_learning_environment.env.balloon.jax_balloon import JaxBalloonState
from train_bc import run_training_bc, observation_to_feature
import train_lstm

import gin
import jax
import jax.numpy as jnp
import numpy as np
import functools
import time
from typing import Any, Sequence, Union
import optax
import pickle

@functools.partial(jax.jit, static_argnums=0)
def select_action(
    network_def: Any, network_params: np.ndarray, state: Any
) -> int:
  """Select an action from network."""
  # since the NN should only return one action just return it?
  return (network_def.apply(network_params, state))

#@gin.configurable
class BCAgent(agent.Agent):
  """An behavior cloning agent based off of MPC4."""

  def __init__(
      self,
      num_actions: int,
      observation_shape: Sequence[int],
      num_layers=4,
      hidden_dims: int= 128,
      input_dims:int = 20,
      seed: Union[int, None] = None,
      isTraining:bool = False
  ):
    super().__init__(num_actions, observation_shape)
    self.num_layers = num_layers
    self.hidden_dims = hidden_dims
    self.input_dims = input_dims
    self.seed = seed
    #it shouldnt really be training
    self.isTraining = isTraining
    self.wind_field = None
    self.network_def = networks.PolicyNetwork(
            num_layers=self.num_layers,
            hidden_dim=self.hidden_dims,
            input_dim=self.input_dims
        )
        
    
    if (isTraining):
       print(f"Start BC agent training.")
       self.network_params, self.network_def = run_training_bc(32,self.num_layers, self.hidden_dims,self.input_dims,'micro_eval')
    else:
        try:
            with open('bc_model_params2.pkl', 'rb') as f:
                bc_params = pickle.load(f)
                self.network_params = bc_params
                """
                0 - is 20,64
                1 - 64,64
                2 - 64,1
                """
                print(f"KERNEL SHAPE {bc_params['params']['Dense_0']['kernel'].shape}")
        except OSError:
            # Training file doesn't exist so we go with random init params
            seed = int(time.time() * 1e6) if self.seed is None else self.seed
            rng = jax.random.PRNGKey(seed)
            dummy_state = jax.random.uniform(rng, (1, 20)) # starting state??
            self.network_params = self.network_def.init(rng, dummy_state)
   
    # self._mode = agent.AgentMode('train')
    self.training_mean, self.training_std = self.get_training_stats()

  def begin_episode(self, observation: np.ndarray) -> int:
    obs = observation_to_feature(observation, self.wind_field)
    final_obs = (obs - self.training_mean)/self.training_std
    action = select_action(self.network_def, self.network_params, final_obs) # returns a jax array for some reason
    return action.item()

  def step(self, reward: float, observation: np.ndarray) -> int:
    obs = observation_to_feature(observation, self.wind_field)
    final_obs = (obs - self.training_mean)/self.training_std
    action = select_action(self.network_def, self.network_params, final_obs)
    return action.item()

  def end_episode(self, reward: float, terminal: bool) -> None:
    pass
  #maybe modify this since i might train on multiple episodes?

  def set_mode(self, mode: Union[agent.AgentMode, str]) -> None:
    self._mode = agent.AgentMode(mode)

  def update_forecast(self, forecast: agent.WindField): 
    self.wind_field = forecast.to_jax_wind_field()

  def get_training_stats(self):
    try:
        data = np.load('micro_expert_demos.npz')
        X_train = data['X_train']
        X_val = data['X_val']
    except OSError:
        print('training data file doesnt exist')
        X_train = []
        X_val = []
    X = jnp.vstack((X_train, X_val))

    #normalize inputs
    feature_means = jnp.mean(X, axis=0)
    feature_std_devs = jnp.std(X, axis=0)
    feature_std_devs = feature_std_devs.at[feature_std_devs==0].set(0.01)
    print(f'MEANS: {feature_means}')
    print(f'STD DEV: {feature_std_devs}')
    return feature_means, feature_std_devs





class BCAgentLSTM(agent.Agent):
  """An behavior cloning agent based off of MPC4."""

  def __init__(
      self,
      num_actions: int,
      observation_shape: Sequence[int],
      features=128,
      seq_len=5,
      seed: Union[int, None] = None,
  ):
    super().__init__(num_actions, observation_shape)
    self.seed = seed
    self.wind_field = None
    self.network_def = networks.LSTM(
            features=features
        )
    self.obs_buffer = []
    self.seq_len = seq_len

    self.apply_fn = jax.jit(self.network_def.apply)
        
    
    try:
            with open('bc_model_params_lstm.pkl', 'rb') as f:
                bc_params = pickle.load(f)
                self.network_params = bc_params
                print(f"KERNEL SHAPE {bc_params['params']['Dense_0']['kernel'].shape}")
    except OSError:
            # Training file doesn't exist so we go with random init params
            seed = int(time.time() * 1e6) if self.seed is None else self.seed
            rng = jax.random.PRNGKey(seed)
            dummy_state = jax.random.uniform(rng, (1, self.seq_len,20)) # starting state??
            self.network_params = self.network_def.init(rng, dummy_state)
   
    # self._mode = agent.AgentMode('train')
    self.training_mean, self.training_std = self.get_training_stats()

  def begin_episode(self, observation: np.ndarray) -> int:
    # obs = observation_to_feature(observation, self.wind_field)
    # final_obs = (obs - self.training_mean)/self.training_std
    # if (len(self.obs_buffer) < self.seq_len):
    #    self.obs_buffer.append(final_obs)
    #    return 0.0 #dont do anything for the first few moves??
    # action = select_action(self.network_def, self.network_params, final_obs)
    # return action
    return 0.0

  def step(self, reward: float, observation: np.ndarray) -> int:
    obs = observation_to_feature(observation, self.wind_field)
    final_obs = (obs - self.training_mean)/self.training_std
    self.obs_buffer.append(final_obs)
    if (len(self.obs_buffer) < self.seq_len):
       return 0.0 #dont do anything for the first few moves??
    sequence = jnp.array(self.obs_buffer[-self.seq_len:], dtype=jnp.float32)
    
    action = self.apply_fn(self.network_params, sequence.reshape(1, self.seq_len, 20))
    #print(f"ACTION IS  {action}")
    return action.item()

  def end_episode(self, reward: float, terminal: bool) -> None:
    self.obs_buffer = []
  #maybe modify this since i might train on multiple episodes?

  def set_mode(self, mode: Union[agent.AgentMode, str]) -> None:
    self._mode = agent.AgentMode(mode)

  def update_forecast(self, forecast: agent.WindField): 
    self.wind_field = forecast.to_jax_wind_field()

  def get_training_stats(self):
    try:
        data = np.load('expert_demos.npz')
        X_train = data['X_train']
        X_val = data['X_val']
    except OSError:
        print('training data file doesnt exist')
        X_train = []
        X_val = []
    X = jnp.vstack((X_train, X_val))

    #normalize inputs
    feature_means = jnp.mean(X, axis=0)
    feature_std_devs = jnp.std(X, axis=0)
    feature_std_devs = feature_std_devs.at[feature_std_devs==0].set(0.01)
    print(f'MEANS: {feature_means}')
    print(f'STD DEV: {feature_std_devs}')
    return feature_means, feature_std_devs



