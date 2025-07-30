import jax
import jax.numpy as jnp
from flax.training import train_state
import optax
import numpy as np
from balloon_learning_environment.env import balloon_env, features
from balloon_learning_environment.env.balloon.jax_balloon import JaxBalloonState
from balloon_learning_environment.agents import agent, networks
from balloon_learning_environment.agents.bc_agent import BCAgent
from balloon_learning_environment.eval import suites
from balloon_learning_environment.env.wind_field import JaxWindField
from balloon_learning_environment.utils import run_helpers
import pickle
import gym
from absl import flags
import sys
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


#Strongly relies on BC being good for first iteration
# Parameters
dagger_iterations       = 3
episodes_per_iteration  = 1
train_epochs_per_iter   = 5
batch_size              = 64
suite_name = "micro_eval"

# Assumes we have collected expert trajectories and warm started policy with BC
try:
    data = np.load('expert_demos.npz')
    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']

    with open('bc_model_params.pkl', 'rb') as f:
        bc_params = pickle.load(f)
   
except OSError:
    print('Training data file or BC parameters file doesnt exist')
    sys.exit(0)

# Initialize trained BC agent and expert MPC agent
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

bc_agent = BCAgent(
    env.action_space.n,
    observation_shape=env.observation_space.shape,
    num_layers=2, # input layer, one hidden layer, output layer
    hidden_dims=128,
    input_dims=20
)
# this will also initialize the network params internally

suite = suites.get_eval_suite(suite_name)


for i in range(dagger_iterations):
    new_states = []
    new_actions = []

    for ep in range(episodes_per_iteration):
        state = env.reset()
        bc_agent.update_forecast(env.get_wind_forecast())
        mpc4_agent.update_forecast(env.get_wind_forecast())
        wind_field: JaxWindField = env.get_wind_forecast().to_jax_wind_field()

        action_pred = bc_agent.begin_episode(state)
        new_states.append(state)
        expert_act = mpc4_agent.begin_episode(state)
        new_actions.append(expert_act)


        for step in range(suite.max_episode_length):
            # Step environment with student action
            state, reward, done, info = env.step(action_pred)

            # Student policy action
            action_pred  = bc_agent.step(reward, state)
            new_states.append(state)
            # Record state & expert correction
            expert_act  = mpc4_agent.step(reward, state)
            new_actions.append(expert_act)

            
            if done:
                print(f"\nEpisode terminated: {info}")
                break

        print(f"Episode {ep+1}/{episodes_per_iteration} in DAgger Iteration {it+1}/{dagger_iterations} is done. \n Time: {end-start:.2f} seconds")