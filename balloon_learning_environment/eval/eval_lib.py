# coding=utf-8
# Copyright 2022 The Balloon Learning Environment Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Evaluation library for Balloon Learning Environment agents."""

import dataclasses
import datetime as dt
import json
from typing import Any, List, Sequence

from absl import logging
from balloon_learning_environment.agents import agent as base_agent
from balloon_learning_environment.env import balloon_env
from balloon_learning_environment.env import simulator_data
from balloon_learning_environment.env.balloon import balloon
from balloon_learning_environment.eval import suites
from balloon_learning_environment.utils import units
from jax import numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm
import os

import json


class EvalResultEncoder(json.JSONEncoder):
  """A JSON encoder for encoding EvaluationResult objects.

  e.g. `json.dumps(eval_result_object, cls=EvalResultEncoder)`.
  """

  def default(self, o: Any):
    if isinstance(o, SimpleBalloonState):
      return {
          'x': o.x.kilometers,
          'y': o.y.kilometers,
          'pressure': o.pressure,
          'superpressure': o.superpressure,
          'elapsed_seconds': o.time_elapsed.total_seconds(),
          'power': o.battery_soc,
      }
    elif dataclasses.is_dataclass(o):
      # Note: don't use dataclasses.asdict since it recurses.
      return o.__dict__
    elif isinstance(o, (np.ndarray, jnp.ndarray)) and o.size == 1:
      return o.item()
    else:
      return json.JSONEncoder.default(self, o)


@dataclasses.dataclass
class SimpleBalloonState:
  """A class for keeping track of a balloon state during evaluation."""
  x: units.Distance
  y: units.Distance
  pressure: float
  superpressure: float
  time_elapsed: dt.timedelta
  battery_soc: float

  @classmethod
  def from_balloon_state(
      cls,
      balloon_state: balloon.BalloonState) -> 'SimpleBalloonState':
    """Creates a SimpleBalloonState from a BalloonState."""
    return cls(balloon_state.x,
               balloon_state.y,
               balloon_state.pressure,
               balloon_state.superpressure,
               balloon_state.time_elapsed,
               balloon_state.battery_soc)


# TODO(joshgreaves): Add some notion of wind difficulty.
@dataclasses.dataclass
class EvaluationResult:
  """A class that holds the results of a single evaluation flight.

  Attributes:
    seed: The seed the controller was evaluated on.
    cumulative_reward: The total reward received by the agent during its flight.
    time_within_radius: The proportion of time the agent spent within the
      station keeping radius. This will be in [0, 1].
    out_of_power: True if the environment terminated because the balloon ran
      out of power.
    envelope_burst: True if the environment terminated because the envelope
      burst.
    zeropressure: True if the environment ended because the balloon
      zeropressured.
    final_timestep: The index of the final timestep. May be used to detect
      whether the balloon reached a terminal state.
    flight_path: The flight path the balloon took.
  """
  seed: int
  cumulative_reward: float
  time_within_radius: float
  out_of_power: bool
  envelope_burst: bool
  zeropressure: bool
  final_timestep: int
  flight_path: Sequence[SimpleBalloonState]

  def __str__(self) -> str:
    return (f'EvaluationResult(seed={self.seed}, '
            f'cumulative_reward={self.cumulative_reward}, '
            f'time_within_radius={self.time_within_radius}, '
            f'out_of_power={self.out_of_power}, '
            f'envelope_burst={self.envelope_burst}, '
            f'zeropressure={self.zeropressure}, '
            f'final_timestep={self.final_timestep})')


def _balloon_is_within_radius(balloon_state: balloon.BalloonState,
                              radius: units.Distance) -> bool:
  return units.relative_distance(balloon_state.x, balloon_state.y) <= radius


def eval_agent(agent: base_agent.Agent,
               env: balloon_env.BalloonEnv,
               eval_suite: suites.EvaluationSuite,
               *,
               collect_diagnostics=False,
               render_period: int = 10,
               calculate_flight_path: bool = True) -> List[EvaluationResult]:
  """Evaluates an agent on a given test suite.

  If the agent being evaluated is deterministic, the result of this function
  will also be deterministic.

  Args:
    agent: The agent to evaluate.
    env: The environment to use for evaluation.
    eval_suite: The evaluation suite to evaluate the agent on.
    render_period: The period with which to render the environment.
      Only has an effect if the environment as a renderer.
    calculate_flight_path: Whether to calculate flight path.

  Returns:
    A list of evaluation results, corresponding to the seeds passed in by
      the eval_suite.
  """
  assert eval_suite.max_episode_length > 0, 'max_episode_length must be > 0.'

  results = list()

  logging.info('Starting evaluation of %s on %s', agent.get_name(), eval_suite)
  agent.set_mode(base_agent.AgentMode.EVAL)


  
  def simulator_write_diagnostics(diagnostic, simulator_state: simulator_data.SimulatorState, start=False):
    state = simulator_state.balloon_state
    if 'simulator' not in diagnostic:
      diagnostic['simulator'] = {'x':[],'y': [], 'z': [], 'wind':[], 'plan': [], 'power_soc': [], 'alt_safety': [], 'env_safety': [], 'power_safety': []}
    
    diagnostic['simulator']['x'].append(state.x.km)
    diagnostic['simulator']['y'].append(state.y.km)
    diagnostic['simulator']['z'].append(simulator_state.atmosphere.at_pressure(state.pressure).height.kilometers)
    
    if not start:
      diagnostic['simulator']['plan'].append(state.last_command)

    diagnostic['simulator']['power_soc'].append(state.battery_soc)
    diagnostic['simulator']['alt_safety'].append(state.altitude_safety_layer.safety_triggered)
    diagnostic['simulator']['env_safety'].append(state.envelope_safety_layer.safety_triggered)
    diagnostic['simulator']['power_safety'].append(state.power_safety_layer._triggered)

    wind_vector = simulator_state.wind_field.get_ground_truth(state.x, state.y, state.pressure, state.time_elapsed)
    diagnostic['simulator']['wind'].append([wind_vector.u.meters_per_second, wind_vector.v.meters_per_second])

  if collect_diagnostics:
    diagnostics = {}

  #aggregate statistics
  avg_reward = 0
  avg_twr = 0
  twr_list = [0] * len(eval_suite.seeds)
  aggregate_avg_iters = 0
  aggregate_avg_opt_time = 0
  safety_violations = [0] * len(eval_suite.seeds)


  for seed_idx, seed in enumerate(eval_suite.seeds):
    print(f'-----------------------------using seed {seed} now-------------------------')
    total_reward = 0.0
    steps_within_radius = 0
    flight_path = list()

    step_count = 0
    agent.avg_iters = 0
    agent.avg_opt_time = 0

    env.seed(seed)
    observation = env.reset()
    agent.update_forecast(env.get_wind_forecast())
    agent.update_atmosphere(env.get_atmosphere())

    if collect_diagnostics:
      diagnostic = {}
      agent.write_diagnostics_start(observation, diagnostic)
    
    action = agent.begin_episode(observation)
    
    if collect_diagnostics:
      agent.write_diagnostics(diagnostic)
      simulator_write_diagnostics(diagnostic, env.get_simulator_state(), start=True)

    # Implement json debugging

    out_of_power = False
    envelope_burst = False
    zeropressure = False

    trajectory = []
  

    with tqdm(total=eval_suite.max_episode_length) as pbar:
      while step_count < eval_suite.max_episode_length:
        observation, reward, is_done, info = env.step(action)
        action = agent.step(reward, observation)

        if collect_diagnostics:
          agent.write_diagnostics(diagnostic)

        total_reward += reward
        avg_reward += reward
        balloon_state = env.get_simulator_state().balloon_state
        trajectory.append((balloon_state.x.m, balloon_state.y.m))
        if calculate_flight_path:
          flight_path.append(
              SimpleBalloonState.from_balloon_state((balloon_state)))
        steps_within_radius += _balloon_is_within_radius(balloon_state,
                                                        env.radius)
        
        #print(f"observation is {observation.x.item(), observation.y.item()}")

        if collect_diagnostics:
          simulator_write_diagnostics(diagnostic, env.get_simulator_state())

        if step_count % render_period == 0:
          env.render()  # No-op if renderer is None.

        step_count += 1

        if is_done:
          out_of_power = info.get('out_of_power', False)
          envelope_burst = info.get('envelope_burst', False)
          zeropressure = info.get('zeropressure', False)
          break
        
        pbar.update(1)

    print(trajectory)
    twr = steps_within_radius / step_count
    avg_twr += twr
    times = eval_suite.max_episode_length//23
    #Only MPC4 has this attribute
    avg_opt_time = agent.avg_opt_time/times
    avg_iters = agent.avg_iters/times

    aggregate_avg_opt_time += avg_opt_time
    aggregate_avg_iters += avg_iters
    print(f"adding violations to {seed_idx}")
    safety_violations[seed_idx] = env.arena.get_balloon_state().power_safety_layer._triggered + env.arena.get_balloon_state().altitude_safety_layer.safety_triggered + env.arena.get_balloon_state().envelope_safety_layer.safety_triggered
    twr_list[seed_idx] = twr

    # if collect_diagnostics:
    #   agent.write_diagnostics_end(diagnostic)
    #   simulator_write_diagnostics(diagnostic, env.get_simulator_state())
    #   #hard coded for the time horizon of 23 steps
    #   times = eval_suite.max_episode_length//23
    #   #Only MPC4 has this attribute
    #   avg_opt_time = agent.avg_opt_time/times
    #   aggregate_avg_opt_time += avg_opt_time
    #   diagnostics[seed]={'seed': seed, 'twr': twr, 'avg_opt_time': avg_opt_time,'reward': total_reward, 'steps': step_count, 'power_violations':env.arena.get_balloon_state().power_safety_layer._triggered, 'altitude_violations':env.arena.get_balloon_state().altitude_safety_layer.safety_triggered, 'envelope_violations':env.arena.get_balloon_state().envelope_safety_layer.safety_triggered,'rollout': diagnostic}
    #   safety_violations[seed_idx] = env.arena.get_balloon_state().power_safety_layer._triggered + env.arena.get_balloon_state().altitude_safety_layer.safety_triggered + env.arena.get_balloon_state().envelope_safety_layer.safety_triggered
    
    agent.end_episode(reward, is_done)

    eval_result = EvaluationResult(
        seed=seed,
        cumulative_reward=total_reward,
        time_within_radius=twr,
        out_of_power=out_of_power,
        envelope_burst=envelope_burst,
        zeropressure=zeropressure,
        final_timestep=step_count,
        flight_path=flight_path)

    # This logs the fraction of seeds evaluated, the seed, and the eval result.
    # e.g. "10 / 100: (seed 10) EvalResult(cumulative_reward=...)"
    logging.info('%d / %d: (seed %d) %s',
                 seed_idx + 1,
                 len(eval_suite.seeds),
                 seed,
                 eval_result)
    logging.info('Power safety layer violations: %d', env.arena.get_balloon_state().power_safety_layer._triggered)
    logging.info('Altitude safety layer violations: %d', env.arena.get_balloon_state().altitude_safety_layer.safety_triggered)
    logging.info('Envelope safety layer violations: %d', env.arena.get_balloon_state().envelope_safety_layer.safety_triggered)

    # Print this to a .err file to make diagnose.mpc4 work
    # write eval_diagnostics to a json file
    lines = [f'{seed_idx + 1} / {len(eval_suite.seeds)}: (seed {seed}) {eval_result}\n', f'Power safety layer violations: {env.arena.get_balloon_state().power_safety_layer._triggered}\n',f'Altitude safety layer violations: {env.arena.get_balloon_state().altitude_safety_layer.safety_triggered}\n', f'Envelope safety layer violations: {env.arena.get_balloon_state().envelope_safety_layer.safety_triggered}\n']

    # Create the path relative to the current script
    # script_dir = os.path.dirname(os.path.abspath(__file__))
    # project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    # base_dir = os.path.join(project_root, "saved_results", f"{type(agent).__name__}")
    # os.makedirs(base_dir, exist_ok=True)

    # filename = f"{type(agent).__name__}-{int(dt.datetime.now().timestamp() * 1000)}.err"
    # datafile = os.path.join(base_dir, filename)

    # print(f"Data in {datafile}")
    # with open(datafile, 'w', encoding='utf-8') as f:
    #   f.writelines(lines)
    
    results.append(eval_result)

  avg_reward /= len(eval_suite.seeds)
  avg_twr /= len(eval_suite.seeds)
  aggregate_avg_opt_time /= len(eval_suite.seeds)
  aggregate_avg_iters /= len(eval_suite.seeds)
  total_violations = sum(safety_violations)

  if collect_diagnostics:
    agent.write_diagnostics_end(diagnostic)
    simulator_write_diagnostics(diagnostic, env.get_simulator_state())
    diagnostics[0]={'seeds': eval_suite.seeds, 'avg_twr': avg_twr, 'twr_list': twr_list, 'avg_opt_time': aggregate_avg_opt_time, 'avg_iters': aggregate_avg_iters, 'avg_reward': avg_reward, 'violations': safety_violations, 'total_violations': total_violations}
   

  if collect_diagnostics:
    return results, diagnostics
  else:
    return results, {}


def graph_optimizer_results(
  name: str,
  agent: base_agent.Agent,
  env: balloon_env.BalloonEnv,
  eval_suite: suites.EvaluationSuite,
  *,
  collect_diagnostics=False,
  render_period: int = 10,
  calculate_flight_path: bool = True):

  agent.optimizer = 'newton'
  #alphas = np.array([0.0001,0.0005,0.001,0.01,0.05,0.1,0.5,1,10,50,100])
  freqs = np.arange(5,16, 5)
  freqlst = np.array([])
  #alphas = np.geomspace(0.0001, 10, num=10)
  #betas = [0.8,0.9,0.99]
  iters = np.arange(10,110,10)
  twr_list = np.array([])
  violations = np.array([])
  time_violations = np.array([])
  for iteramt in iters:
    for freq in freqs:
      lst = agent.hyperparams
      lst[0] = 1
      lst[1] = 0.99
      lst[2] = iteramt
      lst[3] = freq
      agent.hyperparams = lst
      results, diagnositics = eval_agent(agent,env,eval_suite,collect_diagnostics=collect_diagnostics,render_period=render_period,calculate_flight_path=calculate_flight_path)
      twr_list = np.append(twr_list,results[0].time_within_radius)

      freqlst= np.append(freqlst, freq)

      if (results[0].out_of_power or results[0].envelope_burst or results[0].zeropressure or
          env.arena.get_balloon_state().power_safety_layer._triggered
          or env.arena.get_balloon_state().altitude_safety_layer.safety_triggered
          or env.arena.get_balloon_state().envelope_safety_layer.safety_triggered):
        violations = np.append(violations, True)
      else:
        violations = np.append(violations, False)
      if (collect_diagnostics and diagnositics[0]['avg_opt_time'] > 4):
        time_violations = np.append(time_violations, True)
      else:
        time_violations=np.append(time_violations,False)

  #plt.xscale('log')

  # mask = beta == 0.8
  # plt.scatter(iters, twr_list[mask], color=['#360000' if violation else 'purple' for violation in violations[mask]], label='beta=0.8', alpha=0.6)
  print(time_violations)
  mask = freqlst == 5
  plt.scatter(iters, twr_list[mask], color=["#FF00E1" if violation else 'purple' for violation in violations[mask]], label='frequency=5', alpha=0.7,marker='o')
  # mask = time_violations == True
  # plt.scatter(iters[mask], twr_list[mask], color=["#FF00E1" if violation else 'purple' for violation in violations[mask]], label='Time Violation', alpha=0.7,marker='x')

  mask = freqlst == 10
  plt.scatter(iters, twr_list[mask], color=["#00D5FF" if violation else 'blue' for violation in violations[mask]], label='frequency=10', alpha=0.7,marker='o')
  mask = freqlst == 15
  plt.scatter(iters, twr_list[mask], color=["#00FF80" if violation else '#005246' for violation in violations[mask]], label='frequency=15', alpha=0.7,marker='o')

  plt.xlabel("Max iteration values")
  plt.ylabel("Time within radius (percentage)")
  plt.title("Newton's method with SGD on seed 0 of micro_eval", wrap=True)
  # Create manual legend entries
  legend_handles = [
      mpatches.Patch(color='purple', label='freq=5 (Safe)'),
      mpatches.Patch(color="#FF00E1", label='freq=5 (Unsafe)'),
      mpatches.Patch(color='blue', label='freq=10 (Safe)'),
      mpatches.Patch(color="#00D5FF", label='freq=10 (Unsafe)'),
      mpatches.Patch(color="#005246", label='freq=15 (Safe)'),
      mpatches.Patch(color="#00FF80", label='freq=15 (Unsafe)'),
  ]
  plt.legend(handles=legend_handles)
  plt.grid(True)
  plt.figtext(0.5, 0.01, "Shows performance based on value of hyperparameters. Lighter dots indicate safety violations. SGD params were alpha=0.1,beta=0.99,max_steps=40,isNormalized=1", wrap=True, horizontalalignment='center', fontsize=7)
  plt.tight_layout(rect=[0, 0.05, 1, 1])
  plt.savefig('./eval/graphs/newt-iters2.jpg')

def graph_all_optimizer_results2(
  name: str,
  agent: base_agent.Agent,
  env: balloon_env.BalloonEnv,
  eval_suite: suites.EvaluationSuite,
  *,
  collect_diagnostics=False,
  render_period: int = 10,
  calculate_flight_path: bool = True):
    names = ['adam', 'grad', 'momentum', 'newton']
    time_violations = np.array([])
    twr_list = np.array([])
    violations = np.array([])
    for name in names:
      agent.optimizer = name
 
      
      
      results, diagnositics = eval_agent(agent,env,eval_suite,collect_diagnostics=collect_diagnostics,render_period=render_period,calculate_flight_path=calculate_flight_path)
      twr_list = np.append(twr_list,results[0].time_within_radius)
      if (results[0].out_of_power or results[0].envelope_burst or results[0].zeropressure or
            env.arena.get_balloon_state().power_safety_layer._triggered
            or env.arena.get_balloon_state().altitude_safety_layer.safety_triggered
            or env.arena.get_balloon_state().envelope_safety_layer.safety_triggered):
          violations = np.append(violations, True)
      else:
        violations = np.append(violations, False)
      time_violations = np.append(time_violations, diagnositics[0]['avg_opt_time'])

      


    print("HIIIIIIIIIIIIIIIIIIIII")
    print(time_violations)
    print(violations)
    print(twr_list)
    def trunc(values, decs=0):
      return np.trunc(values*10**decs)/(10**decs)
    adj_list = trunc(time_violations, 3)
    print(adj_list)

    plt.bar(names,twr_list)
    for i in range(len(names)):
        plt.text(i, adj_list[i] // 2, adj_list[i], ha='center')

    plt.xlabel("Optimizers")
    plt.ylabel("Time within radius (percentage)")
    plt.title("Performance of different optimizers", wrap=True)
 
    plt.grid(True)
    plt.figtext(0.5, 0.01, "Shows performance for tuned optimizers. The number in the bars is the average optimization time.", wrap=True, horizontalalignment='center', fontsize=8)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig('./eval/graphs/final2.jpg')


def graph_all_optimizer_results(
  name: str,
  agent: base_agent.Agent,
  env: balloon_env.BalloonEnv,
  eval_suite: suites.EvaluationSuite,
  *,
  collect_diagnostics=False,
  render_period: int = 10,
  calculate_flight_path: bool = True):
    names = ['adam', 'adabelief', 'grad', 'momentum', 'sgd', 'newton']
    times =[1.2705, 0.8443, 2.0103, 1.4894, 1.0961, 2.3820]
    twr_list = [0.7802, 0.8542, 0.7208, 0.7896,0.775,0.8073 ]
 
    plt.bar(names,twr_list)
    

    plt.xlabel("Optimizers")
    plt.ylabel("Time within radius (percentage)")
    plt.title("Objective value of different optimizers", wrap=True)
    plt.axhline(y=0.715625,linewidth=1, color='r', label='Baseline objective value')
    plt.legend()
    plt.grid(True)
    #plt.figtext(0.5, 0.01, "Shows performance for tuned optimizers. The number in the bars is the average optimization time.", wrap=True, horizontalalignment='center', fontsize=8)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig('./eval/graphs/final2.jpg')
    plt.cla()
    plt.xlabel("Optimizers")
    plt.ylabel("Average optimization time (s)")
    plt.title("Computational Performance of different optimizers", wrap=True)
    plt.bar(names, times, color='orange')
    #plt.plot(names, [1.4199] * len(times), color='r', linestyle='-', label='Baseline time')
    plt.axhline(y=1.4199,linewidth=1, color='r', label='Baseline time')
    plt.legend()
    plt.savefig('./eval/graphs/final3.jpg')

def graph_initalizer_results():
    names = ['random', 'best altitude', 'opd search', 'previous', 'zeros']
    twr_list = [0.53125, 0.8072916666666666, 0.20208333333333334, 0.4479166666666667,0.196875 ]
 
    plt.bar(names,twr_list)
    

    plt.xlabel("Initalization methods")
    plt.ylabel("Time within radius (percentage)")
    plt.title("Performance of different initializon methods on seed 0 (with SGD)", wrap=True)
    #plt.figtext(0.5, 0.01, "Shows performance for tuned optimizers. The number in the bars is the average optimization time.", wrap=True, horizontalalignment='center', fontsize=8)
    plt.tight_layout()
    plt.savefig('./eval/graphs/final4.jpg')

def graph_horizon_results(
  agent: base_agent.Agent,
  env: balloon_env.BalloonEnv,
  eval_suite: suites.EvaluationSuite,
  *,
  collect_diagnostics=False,
  render_period: int = 10,
  calculate_flight_path: bool = True
):
    # horizon = [7,15,23, 47,71,95]
    # plan_steps = [60,80,120,160,240,320,400]
    horizon = [95]
    plan_steps = [400]
 
    
    twr_list = []
    violations = []
    times =[]
    iters = []
    for h in horizon:
      twrs = []
      v = []
      t = []
      i = []
      for p in plan_steps:
        env.reset()
        agent.plan_steps = p
        agent.horizon = h
        results, diagnositics = eval_agent(agent,env,eval_suite,collect_diagnostics=collect_diagnostics,render_period=render_period,calculate_flight_path=calculate_flight_path)
        twrs.append(diagnositics[0]['avg_twr'])
        v.append(diagnositics[0]['total_violations'])
        t.append(diagnositics[0]['avg_opt_time'])
        i.append(diagnositics[0]['avg_iters'])
      twr_list.append(twrs)
      violations.append(v)
      times.append(t)
      iters.append(i)

    #plt.xscale('log')
    fig = plt.figure()
    plt.title("Various metrics with seed 0 (MPC with SGD)", wrap=True)
    ax = fig.add_subplot(2,2,1,projection='3d')

    color = iter(plt.cm.rainbow(np.linspace(0, 1, len(horizon))))
    print(f"---------------TWR STATS---------------------")
    for i,h in enumerate(horizon):
      c = next(color)
      ax.bar(plan_steps, twr_list[i], zs=h, zdir='y', color=c, alpha=0.8, width=1)
      print(f"Horizon {h}: {twr_list[i]}")
      
    ax.set_xlabel('Plan size')
    ax.set_ylabel('Horizon')
    ax.set_zlabel('TWR')

    print('-------------VIOLATIONS-----------------')
    for i,h in enumerate(horizon):
      print(f"Horizon {h}: {violations[i]}")

    ax = fig.add_subplot(2,2,3,projection='3d')
    
    color = iter(plt.cm.rainbow(np.linspace(0, 1, len(horizon))))
    print('-------------AVG TIMES-----------------')
    for i,h in enumerate(horizon):
      c = next(color)
      ax.bar(plan_steps, times[i], zs=h, zdir='y', color=c, alpha=0.8, width=1)
      print(f"Horizon {h}: {times[i]}")
    ax.set_xlabel('Plan size')
    ax.set_ylabel('Horizon')
    ax.set_zlabel('Avg opt times')

    ax = fig.add_subplot(2,2,4,projection='3d')
    color = iter(plt.cm.rainbow(np.linspace(0, 1, len(horizon))))
    print('-------------AVG ITERS-----------------')
    for i,h in enumerate(horizon):
      c = next(color)
      ax.bar(plan_steps, iters[i], zs=h, zdir='y', color=c, alpha=0.8, width=1)
      print(f"Horizon {h}: {iters[i]}")
    ax.set_xlabel('Plan size')
    ax.set_ylabel('Horizon')
    ax.set_zlabel('Avg iters')
   
    # plt.legend(handles=legend_handles)
    # plt.grid(True)
    # plt.figtext(0.5, 0.01, "Shows performance based on value of hyperparameters. Lighter dots indicate safety violations. SGD params were alpha=0.1,beta=0.99,max_steps=40,isNormalized=1", wrap=True, horizontalalignment='center', fontsize=7)


    # plt.xlabel("Initalization methods")
    # plt.ylabel("Time within radius (percentage)")
    
    plt.show()
    #plt.figtext(0.5, 0.01, "Shows performance for tuned optimizers. The number in the bars is the average optimization time.", wrap=True, horizontalalignment='center', fontsize=8)
    plt.tight_layout()
    plt.savefig('./eval/graphs/horizonstats.jpg')


def graph_trajectories():
  mpc4 = [(29878.966855266495, 25581.814693164542), (29713.600679092873, 25416.8913882432), (29488.34052712752, 25194.1499575413), (29204.600864354157, 25036.0104502641), (28864.03185150247, 24962.586530598437), (28472.800418672952, 24910.64373845112), (28044.824552758597, 24927.123830351415), (27583.852829798452, 24993.60046499043), (27104.33057803272, 25099.130353465585), (26616.9544144362, 25212.33722615692), (26118.353945428826, 25329.380866483985), (25633.153266460802, 25426.104128580624), (25144.003880506272, 25524.260045576517), (24680.68300038408, 25551.28453433466), (24244.526892909904, 25534.918609166878), (23805.11557651557, 25517.363335117447), (23399.698501810497, 25400.73072736801), (22994.13515642484, 25275.605875338104), (22586.961373286933, 25146.709082438505), (22219.24074761488, 25029.716934834065), (21850.586113112913, 24910.85756314141), (21521.774300391546, 24768.21472037415), (21193.51446218865, 24616.63693539619), (20911.009838144113, 24305.108022331213), (20663.158435812788, 23988.08144408958), (20414.828806749334, 23668.386343123722), (20165.809856780397, 23345.89778019273), (19948.106305355606, 23026.12649954487), (19758.46979839846, 22748.373175030563), (19569.856491953047, 22467.62705849294), (19403.56783462377, 22226.783083636623), (19238.47315968514, 21983.179216521316), (19075.0007355782, 21737.181811278115), (18926.75655744753, 21489.51396909478), (18791.07883109335, 21203.91287243312), (18660.568639760975, 20907.36240590853), (18541.034894254008, 20590.59090080937), (18418.933271762602, 20287.343252464798), (18310.945140724645, 19963.12031074968), (18204.498392708338, 19641.412158677736), (18097.435020253564, 19327.218231473507), (17990.769131690147, 19020.640734140292), (17884.71499920496, 18722.107717158604), (17780.468088168658, 18432.27332819034), (17678.973974670655, 18147.83746615215), (17578.097484030222, 17876.827478480915), (17479.39001005185, 17611.96053642838), (17389.695889152652, 17329.80878431399), (17285.13725202765, 17076.955216793413), (17147.609261064572, 16763.22264193645), (16980.319796050273, 16418.558105641245), (16781.40703628092, 16040.941749870888), (16541.85957364821, 15656.38761265902), (16262.405199530273, 15381.581817223398), (15943.451140314792, 15178.510141040899), (15588.221262043622, 14896.837362844633), (15204.452710344412, 14567.155844410236), (14792.794061784709, 14269.9163680172), (14352.435760896125, 14020.57710817128), (13891.123834279195, 13782.62351706394), (13412.597188007607, 13543.993575885675), (12910.466658589216, 13316.512750242835), (12396.944610620165, 13098.536232523651), (11872.65491854442, 12890.318306933748), (11366.786714982067, 12649.907449597476), (10894.38675670433, 12371.916091812063), (10416.502278678068, 12089.73873075786), (9938.628867199754, 11800.541721769432), (9462.009584695315, 11503.524569231276), (8979.103676259638, 11204.874900916306), (8545.37645131295, 10871.552431332611), (8182.97445832508, 10444.872644232704), (7818.625903956797, 10012.872491200074), (7525.796233512437, 9658.705047555652), (7302.6713658207345, 9246.044007756695), (7094.12673485345, 8804.346778263438), (6885.325411555504, 8358.77609622268), (6746.960857678875, 7909.53701704024), (6674.947531975785, 7496.677352734663), (6652.497601696995, 7109.356958161101), (6630.452963174869, 6717.420318376226), (6610.443355582637, 6323.909484763753), (6588.628630711965, 5922.646366733294), (6604.781304976764, 5554.3088409600205), (6658.662887104755, 5270.75260764431), (6716.1720950426, 4978.906177817731), (6795.102586745344, 4668.950804202863), (6882.474476483377, 4385.556752526007), (6969.486208211624, 4136.693126640528), (7048.106356030869, 3987.893785422869), (7127.928145773284, 3835.7387167393663), (7196.37504951993, 3659.946053661334), (7266.8120636480135, 3479.9815436228528), (7338.775554578798, 3296.2004714225595), (7401.463230011536, 3109.600757691923), (7479.233039447175, 2915.057027127963), (7565.515134238692, 2709.5654566022276), (7646.98080206851, 2371.3748094109997), (7708.368586526548, 2008.106031652611), (7740.328318088575, 1644.828980483299), (7732.322266550659, 1229.1062166488541), (7682.300148746456, 733.7957400809864), (7596.1109583272, 185.537712675765), (7468.971606819493, -398.7434510812834), (7296.570007028896, -978.6359827402794), (7077.3348167735285, -1590.8336522275408), (6809.187493222545, -2198.6598733439946), (6491.72261561791, -2787.959498628274), (6127.307118687003, -3423.561741732097), (5718.658958814439, -4030.4152202282007), (5273.443528882209, -4561.838127745739), (4797.162309533423, -5061.178794488915), (4289.020253799498, -5547.202812275891), (3764.881140126715, -6038.161381075066), (3219.5845479244226, -6547.038317810966), (2664.273694917124, -7065.616288256917), (2131.5464734653624, -7561.187159778103), (1641.0375052185752, -8067.518569796708), (1149.0620764778362, -8575.354965904875), (691.5859650908799, -9159.995375712517), (290.41953417735795, -9830.673104452699), (-94.91552954325525, -10492.798884477264), (-450.1420394768493, -11151.012854144605), (-749.821995648975, -11825.106617369112), (-998.8936545088504, -12576.659523107113), (-1216.3801997743406, -13332.09813285135), (-1386.3110094188603, -14021.721415080583), (-1498.261368426189, -14611.637871361081), (-1585.8764882423282, -15153.236170019967), (-1626.161059985344, -15719.089590250825), (-1648.5223862442392, -16295.316432326637), (-1658.6877714818527, -16867.86773760069), (-1651.3913721011343, -17417.570081299004), (-1624.2059813168132, -17858.04226016478), (-1569.841888478605, -18170.77882140501), (-1503.4847557865166, -18478.052235885545), (-1416.2655234894144, -18800.370278381954), (-1316.988912036987, -19130.456111923464), (-1210.3150535830534, -19443.27112895582), (-1084.8162845798408, -19654.65165928909), (-934.5319208373296, -19885.658131290736), (-756.4675581755935, -20160.39116922766), (-614.1589156554883, -20355.638757971858), (-509.52042731269506, -20673.29585521039), (-441.6889539774424, -20971.2777423819), (-413.3329535838173, -21297.396114172254), (-392.12819340794937, -21645.18376515044), (-378.9752348657228, -22025.957364479684), (-361.5782304211328, -22379.478955198894), (-355.21170120793494, -22783.21825959759), (-358.3402786757479, -23229.326863126655), (-365.5074487821785, -23688.045951274107), (-376.69353261315035, -24156.722228823724), (-390.4402366256965, -24628.2398470071), (-410.4761702640979, -25108.653499267974), (-437.7864502424689, -25591.348919055876), (-470.76747665082866, -26069.106850178614), (-485.1125332119753, -26520.04407514124), (-499.74069641916395, -26965.316659574564), (-522.3847204809376, -27415.189501092187), (-532.4098662875049, -27837.8409874117), (-544.4032778041433, -28258.25375492015), (-559.8483322661157, -28677.524871757458), (-577.0907388870273, -29091.65391357817), (-579.2092742728084, -29470.17413578729), (-562.1935372231421, -29778.377731357723), (-534.1845653199348, -30043.218393094176), (-507.5655824188021, -30307.256343744593), (-483.12422129611144, -30573.04105833073), (-464.93418097748787, -30854.304636792403), (-424.267396138573, -31070.576381782572), (-370.35650480677174, -31261.856892357107), (-324.23875153236753, -31459.80574253408), (-265.1405975490928, -31634.8232903295), (-207.74996491328972, -31807.379945937028), (-126.94719851931313, -31951.22001776495), (-26.91086031016891, -32090.15065396196), (68.38518410839369, -32222.078698783887), (178.48682844128078, -32360.960153006024), (283.6142055289299, -32490.078061895612), (383.00800156431114, -32610.196505152482), (496.2187165711289, -32741.037900427444), (618.6355891033119, -32875.27635798667), (732.3932697096894, -32999.76994640814), (857.8506321169481, -33124.60248030798), (995.4640698600366, -33224.566126496036), (1126.6262880334384, -33332.2154006412), (1254.9430113397261, -33439.14158033504), (1392.629107510742, -33519.700313622365), (1483.4319905112152, -33605.66626148818), (1522.0668193436577, -33739.122816691684), (1507.003328727214, -33971.62225876545), (1476.2073491112142, -34220.69185506173), (1432.0054560257195, -34471.39742550977), (1394.222114322652, -34720.38521410931), (1350.4563420969585, -34968.70221840266), (1293.9038602715164, -35213.152646799776), (1252.8089319408227, -35460.543883789345), (1204.9396430025395, -35707.041904500365), (1151.3169336696624, -35951.779999529215), (1098.7500021824985, -36197.44789457279), (1061.868864739205, -36445.27187536114), (1017.0111307536595, -36695.17392333351), (968.4658937331471, -36945.665405440595), (915.6478488425687, -37196.123129229), (880.550940573112, -37446.47477677863), (861.7226605945359, -37674.76769510459), (841.0566474428815, -37906.42619483368), (831.1971330582658, -38121.30177613713), (819.0482698089481, -38340.43586465894), (813.4340721035815, -38549.39742384176), (830.4165692416364, -38703.09228860073), (852.5083087077614, -38838.79636256078), (895.8974869636327, -38929.16927772651), (951.8490445210225, -39022.22639352264), (999.5963790073354, -39114.36399152492), (1060.0932034529285, -39221.206522195615), (1113.091004875493, -39325.11330164391), (1168.163622587382, -39436.311171000096), (1228.57315185107, -39556.74730633636), (1304.4177565561197, -39668.81151585237), (1373.786941089624, -39790.34330308493), (1456.741524819644, -39885.03866332407), (1555.4770319303893, -39969.83811563815), (1645.9243271433284, -40056.626646936915), (1748.8718677811817, -40163.69376087439), (1846.146470290465, -40267.55761044436), (1934.278897075646, -40366.67242606911), (2036.393574110467, -40501.08898085254), (2134.937835972431, -40635.200524914675), (2235.150346070462, -40784.20803289396), (2334.4374412550665, -40940.33953187957), (2427.5169239018064, -41087.430158647025), (2522.6713800786542, -41250.00251987648), (2610.2954572840767, -41399.56195011386), (2691.915567437379, -41542.30085974862), (2824.123112937134, -41822.17867686224), (2966.364848427715, -42119.32117945072), (3098.0533828268103, -42408.756965602166), (3216.9276737671166, -42679.11686826883), (3320.9478917064484, -42919.489393919706), (3411.5344895213725, -43126.493954357764), (3493.823044545858, -43314.24470669891)]

  bc= [(29878.966855266495, 25581.814693164542), (29751.48362877917, 25393.005060716187), (29604.877590605924, 25240.497013489436), (29430.019556938372, 25055.6360020491), (29223.238814985685, 24820.452804813565), (28988.667655094796, 24590.688931527915), (28732.877726940773, 24357.106332285748), (28468.567308410216, 24120.871344472875), (28222.279817260613, 23881.14239687839), (28003.380488576644, 23631.445753241485), (27815.213265276943, 23418.320339139525), (27654.155701393123, 23240.125203095242), (27521.941362250647, 22989.59118370226), (27414.06360230945, 22733.759242029333), (27321.209537390656, 22469.226666691593), (27235.889080112363, 22217.37841251744), (27153.28739653602, 21976.95534913208), (27070.786101272555, 21737.363637404564), (26987.98179779925, 21495.909244233222), (26904.987264478754, 21252.373706249815), (26821.92808238508, 21006.587767930734), (26738.932334704732, 20758.374112468653), (26656.131556978347, 20507.575143637605), (26573.6751280698, 20254.125870144093), (26491.724796430484, 19998.03784689079), (26410.44868756902, 19739.36461534406), (26330.017278872503, 19478.187637013896), (26250.602896916986, 19214.614624198828), (26172.3792603873, 18948.777115576417), (26095.532916141965, 18680.891746581732), (26020.2556582592, 18411.216267494554), (25946.739176453433, 18140.031934961535), (25875.171763250983, 17867.626808597317), (25805.734589915133, 17594.276239966337), (25738.607859466094, 17320.28042544866), (25673.97960804525, 17045.990629060387), (25612.061918826097, 16771.85890203124), (25553.102205147057, 16498.458590351856), (25497.375452546316, 16226.461079810786), (25445.151556983423, 15956.528633310147), (25396.661607237154, 15689.23074894752), (25352.046464272884, 15424.84444839383), (25311.45479033248, 15163.719066607948), (25275.006445939653, 14906.128348153768), (25242.804727330084, 14652.311761882578), (25214.900801738564, 14402.297427612848), (25191.344591284804, 14156.1246534659), (25172.181693270075, 13913.841102108745), (25157.45378163259, 13675.499045517037), (25147.189706860736, 13441.11956285681), (25141.407011065487, 13210.697830634417), (25140.11108368865, 12984.18949328976), (25143.292250261497, 12761.500503747058), (25150.931288157517, 12542.519579911572), (25163.002895204634, 12327.129247076646), (25179.472870339865, 12115.191323929408), (25200.29746668133, 11906.54218474244), (25225.422186158263, 11700.985448221836), (25254.46920053241, 11497.14644491355), (25287.371749660226, 11294.940305595213), (25324.05119870571, 11094.272872536407), (25364.417966827263, 10895.04045989694), (25409.66088232008, 10701.098506750106), (25459.008832302003, 10507.07529571711), (25511.942441843814, 10310.978573163513), (25568.152964186353, 10112.706745182255), (25627.42009896691, 9912.891145908026), (25689.58005060748, 9712.36067406568), (25754.502837757856, 9511.815016878823), (25822.083711874424, 9311.713776313867), (25892.231338919388, 9112.265058622652), (25964.861787444584, 8913.472849742026), (26039.892684123457, 8715.191913880433), (26117.239329517797, 8517.189697357353), (26196.813739676596, 8319.17664648184), (26278.5250936414, 8120.821925698077), (26362.274828679285, 7921.798727649573), (26447.96499310009, 7721.73359857186), (26535.495315941247, 7520.230910935876), (26624.760773497175, 7316.892630891283), (26715.653785803486, 7111.303823960241), (26808.062096417227, 6903.044949955225), (26901.86926721778, 6691.699785157174), (26996.955375721613, 6476.855693383455), (27093.195828872667, 6258.104479916203), (27190.463896845446, 6035.043364898414), (27288.629761822376, 5807.275978825219), (27387.561555760585, 5574.413491931675), (27487.12591139182, 5336.0755097129), (27587.18859315882, 5091.891465638675), (27687.614744343933, 4841.501659373702), (27788.26941083907, 4584.558525951028), (27889.017617466525, 4320.727612220349), (27989.72600791561, 4049.6809371571035), (28090.262323066316, 3771.105763653871), (28190.495411417556, 3484.707209031637), (28290.29664976221, 3190.2092191286483), (28389.541459605876, 2887.355137921897), (28488.10682655467, 2575.90821882409), (28585.87477232165, 2255.652268254671), (28682.73144610919, 1926.392202853979), (28778.568661846803, 1587.965456146915), (28873.28476086371, 1240.2350712994546), (28966.782811774465, 883.0851510179466), (29058.973611888534, 516.4220648181038), (29149.772778658655, 140.17517946979393), (29239.10861517345, -245.67809345811912), (29326.87042820207, -641.1360805092677), (29412.976333532133, -1046.1492544556068), (29497.380226165646, -1460.645069725952), (29580.044229348772, -1884.5310397659432), (29660.93837222735, -2317.693535250764), (29740.033140725507, -2760.02924607619), (29817.30621804347, -3211.412572475017), (29892.745541071534, -3671.6919341291828), (29966.346298946562, -4140.6902735849935), (30038.11451143086, -4618.20570187891), (30108.06591282672, -5104.005096712478), (30176.242001539853, -5597.778170750135), (30242.69092655355, -6099.200606179974), (30307.469226426736, -6607.929002085236), (30370.06299595695, -7122.882785857787), (30430.578995090553, -7643.567059264734), (30489.08270568293, -8169.61658530482), (30545.64270998374, -8700.660370047295), (30600.327218627295, -9236.323876814258), (30653.156449657443, -9776.393419115451), (30704.17075215352, -10320.590559385982), (30753.418941218035, -10868.614441697304), (30800.962027732945, -11420.124693166556), (30846.884027911292, -11974.713154928006), (30891.269045183028, -12531.968549017853), (30934.196749223738, -13091.49872002942), (30975.776052289795, -13652.834946115698), (31016.126144535818, -14215.446565846498), (31055.36855252066, -14778.811027296833), (31093.63994924364, -15342.35654818781), (31131.07906979566, -15905.49992652663), (31167.82221710788, -16467.66340537712), (31204.002198222115, -17028.275969190057), (31239.7467166665, -17586.77850658381), (31275.171882481263, -18142.646142416117), (31310.38875615813, -18695.365436551467), (31345.504838028886, -19244.4342009186), (31380.671969464696, -19789.192513651455), (31416.05814770297, -20328.91456633786), (31451.826563806615, -20862.90869178786), (31488.124123756606, -21390.467147096417), (31525.079291322254, -21910.970453675844), (31562.80104432154, -22423.861920836272), (31601.380203607878, -22928.64932891765), (31640.91590194373, -23424.811350888864), (31681.5012055154, -23911.84379975721), (31723.216570754885, -24389.29470906152), (31766.124624316937, -24856.784880388695), (31810.267963549355, -25314.018781476356), (31855.669617305328, -25760.78610490336), (31902.336015901485, -26196.958148199446), (31950.25566682349, -26622.480953602913), (31999.40326870478, -27037.366826235862), (32049.74015764516, -27441.68594059484), (32101.214303981018, -27835.557994550116), (32153.76360597337, -28219.14330914044), (32207.320716367343, -28592.620029045793), (32261.816145189245, -28956.166222165433), (32317.17562559519, -29309.974666880516), (32373.317276576454, -29654.287990531186), (32430.15164689268, -29989.330271697785), (32487.582302504277, -30315.348226899237), (32545.507677253627, -30632.6130343627), (32603.82300848952, -30941.395881949415), (32662.4206707614, -31241.971863357903), (32721.182540733033, -31534.643095016618), (32779.975837937345, -31819.747814934046), (32838.66544908306, -32097.621251737917), (32897.11817496975, -32368.593628724753), (32955.20570053753, -32632.963438753643), (33012.80528809624, -32891.02928783197), (33069.80032355297, -33143.07829821815), (33126.07824232308, -33389.388453122265), (33181.53167712721, -33630.23073248911), (33236.05647882095, -33865.87168600213), (33289.55137887142, -34096.57531055537), (33341.907871201685, -34322.62544286713), (33393.00995865951, -34544.32514273434), (33442.74056821947, -34761.98502884116), (33490.98729664046, -34975.91558291093), (33537.646567447875, -35186.418114989465), (33582.62338231842, -35393.79021241633), (33625.830076881386, -35598.32382002574), (33667.186650812255, -35800.30759067633), (33706.61841734526, -36000.02822929419), (33744.056790550945, -36197.77120051947), (33779.43896470915, -36393.82135183685), (33812.7059866499, -36588.46317628076), (33843.808341446194, -36781.97199120916), (33881.637174095835, -36957.45630104931), (33925.22981284551, -37113.29973711073), (33974.262731344184, -37246.91554982149), (34028.30112310939, -37358.31125985299), (34086.9192192696, -37450.148099304504), (34149.27964679979, -37527.787894960355), (34215.385449405665, -37595.75837221543), (34284.93189707456, -37659.3621279748), (34357.675761440194, -37723.15252554812), (34433.31584841881, -37789.74245319039), (34511.39173389448, -37859.53862231649), (34591.535527004664, -37931.60429898374), (34673.39029084861, -38004.34383396856), (34756.629117077035, -38076.10621090036), (34840.944670938516, -38145.53673086437), (34926.05145789544, -38211.720206117716), (35011.68428922808, -38274.20232527263), (35097.59773998778, -38332.94696839782), (35183.56533818851, -38388.252880176566), (35269.332893308885, -38440.74959659621), (35354.65941972871, -38491.222993370895), (35439.32709774096, -38540.504400958605), (35523.138450001985, -38589.417433390765), (35605.91405797314, -38638.73859421556), (35687.44849604589, -38689.23243084917), (35767.52012043974, -38741.60316717011), (35845.9776847635, -38796.37682567405), (35922.666271409755, -38854.00965479938), (35997.397548041285, -38914.915449024345), (36070.07352231217, -38979.36162967292), (36140.59713675742, -39047.56986953498), (36208.82484929295, -39119.75300882545), (36274.627078199694, -39196.07522533052), (36337.84725839996, -39276.69063058045), (36398.36215141853, -39361.71736583337), (36455.988651310625, -39451.311523095756), (36510.342964342104, -39545.80143921427), (36560.75963966009, -39645.88939392362), (36606.41625262994, -39752.80125696734), (36647.17888650184, -39867.27513221274), (36683.45556228747, -39989.1525218051), (36715.68422220529, -40117.96471229927), (36744.257104239696, -40253.17012027739), (36769.5126578541, -40394.250950516)]

  lstm = [(29878.966855266495, 25581.814693164542), (29772.78090422768, 25368.633345511724), (29667.284684270242, 25151.619176612894), (29561.75533285211, 24931.0583854491), (29455.95713622817, 24707.033973721616), (29350.00237963911, 24479.538445058508), (29239.964584010144, 24249.437279025708), (29126.491812274173, 24016.360240137587), (29010.14649240629, 23780.08942950144), (28891.708499676795, 23540.505299776883), (28771.934410221395, 23297.565882119965), (28651.393462848864, 23051.305400314257), (28530.462420608237, 22801.800222930633), (28409.368339115048, 22549.147491602398), (28288.4655916791, 22293.478041977734), (28168.263061172038, 22034.965875209647), (28049.130922965327, 21773.7626389958), (27931.34533482491, 21509.993929223903), (27815.13044521099, 21243.763369043834), (27700.67989372825, 20975.152931432003), (27588.167567532408, 20704.226462975214), (27477.756848467456, 20431.037055266057), (27369.601203621627, 20155.638719007056), (27263.847496555347, 19878.10107882554), (27160.63783333545, 19598.525184843205), (27060.108694226532, 19317.058831971914), (26962.39183543926, 19033.90946250087), (26867.61631555599, 18749.35304372656), (26775.906186756045, 18463.737935633206), (26687.3817332853, 18177.48282666403), (26602.159709802952, 17891.068911547252), (26520.352505852454, 17605.026864803436), (26442.067503863385, 17319.918258019523), (26367.40756413761, 17036.318727438647), (26296.470457948424, 16754.79776935399), (26229.348850452203, 16475.90264531347), (26166.129776525246, 16200.146758388428), (26106.893596588183, 15927.994365727443), (26051.714653766765, 15659.84192814382), (26000.661439319003, 15396.003749264031), (25953.794012120445, 15136.702737622612), (25911.166489151037, 14882.067124319192), (25872.82569798593, 14632.13445867897), (25838.80986564081, 14386.856901700843), (25809.150136928532, 14146.109004396467), (25783.869943665028, 13909.697501273198), (25762.984917727896, 13677.375759882112), (25746.502920555067, 13448.861913171759), (25734.42293359941, 13223.852932900334), (25726.73680248543, 13002.035544785404), (25723.42880325744, 12783.09452675627), (25724.47466860633, 12566.71942489924), (25729.843457647417, 12352.611398576215), (25739.496684951002, 12140.489776055152), (25753.387982062188, 11930.097914589422), (25771.465521783874, 11721.208687168435), (25793.667734959337, 11513.627750693659), (25819.930567144656, 11307.193604630096), (25850.18103767404, 11101.757527270567), (25884.339676250725, 10897.223392174305), (25922.323302626333, 10693.515297194861), (25964.042062364446, 10490.568043502653), (26009.406900379712, 10288.32925266999), (26058.36244538615, 10086.72833371128), (26110.849039942237, 9885.626005999877), (26166.802008529652, 9684.838703082993), (26226.164166705683, 9484.1553240587), (26288.826425198575, 9283.384066579865), (26354.670194076556, 9082.32925883704), (26423.57000910928, 8880.789658354004), (26495.395333126435, 8678.558726253652), (26570.010773117152, 8475.425704038795), (26647.277680318955, 8271.176748179898), (26727.05409202646, 8065.59445561384), (26809.194520515477, 7858.456954397525), (26893.550486414802, 7649.537318553419), (26979.969177276005, 7438.603172830748), (27068.295057220224, 7225.416639871479), (27158.36990430522, 7009.734618900038), (27250.03174193493, 6791.309344246127), (27343.115464992305, 6569.889158957252), (27437.45399897975, 6345.219507991653), (27532.876145496826, 6117.0440923504575), (27629.208317897515, 5885.106152906996), (27726.274573055198, 5649.149878445107), (27823.89502886805, 5408.921799758542), (27921.88809901201, 5164.172263855673), (28020.068955295894, 4914.65682202119), (28118.250239461595, 4660.137531285471), (28216.242764284892, 4400.384178989298), (28313.854618951293, 4135.175365659399), (28410.891908982216, 3864.2995456750796), (28507.15898229047, 3587.55584086036), (28602.4592339389, 3304.7545792298315), (28696.594247669833, 3015.717565932061), (28789.364648654533, 2720.277974093582), (28880.570375685333, 2418.2798198708906), (28970.011557417834, 2109.5770222883903), (29057.487754498216, 1794.0319849622626), (29142.798836872673, 1471.5140859301475), (29225.745929022763, 1141.8980674367897), (29306.13067656335, 805.0622125412799), (29383.756749300468, 460.88631452688355), (29458.4302784041, 109.24964906936498), (29529.958613279, -249.97094052996977), (29598.10571983007, -616.9025990995488), (29662.66178651539, -991.677518168013), (29723.441415315283, -1374.433509690591), (29780.263077102027, -1765.313869058427), (29832.949509186707, -2164.4661378684527), (29881.328759504955, -2572.0397549578643), (29925.234132452897, -2988.1825796272983), (29964.50471479408, -3413.0367163463557), (29998.986513224714, -3846.730788934569), (30028.533010018822, -4289.371720268323), (30053.005754620266, -4741.037189549762), (30072.27436420349, -5201.773108145208), (30086.217644072764, -5671.599323280522), (30094.724144961423, -6150.517006266214), (30097.69269492133, -6638.514572624113), (30095.032399352596, -7135.55811721991), (30086.15902923791, -7640.889615520712), (30071.007048994026, -8154.381864908478), (30049.52449905905, -8675.965789715989), (30021.67301804039, -9205.46171601268), (29987.4275100835, -9742.710873763008), (29946.776491641962, -10287.549335347925), (29899.722872883416, -10839.812957034566), (29846.283526220424, -11399.339786990946), (29786.490377995993, -11965.97164464603), (29720.389733412034, -12539.556815786385), (29648.041480675187, -13119.947211261853), (29569.52033693556, -13706.990608934855), (29484.91478354155, -14300.522729388307), (29394.32524976434, -14900.35745157269), (29297.865980112845, -15506.270620211315), (29195.661222870694, -16117.97182941571), (29087.846094176355, -16735.072528060755), (28974.563937729785, -17357.063857246987), (28855.965608821723, -17983.30854923565), (28732.206969702565, -18613.060058080806), (28603.446815735217, -19245.499261695903), (28469.84590398307, -19879.702510249295), (28331.564306630564, -20514.569596736936), (28188.759089265346, -21148.80293504166), (28041.58303124115, -21780.942252640492), (27890.18196489403, -22409.4490007718), (27734.693707052975, -23032.874046175373), (27575.247588306396, -23650.06746146231), (27411.96312604398, -24260.252041894808), (27244.95233055739, -24863.16291627679), (27074.319365462732, -25459.026390113), (26900.765643269282, -26047.335171465802), (26724.78128598089, -26628.223383929948), (26546.445286066053, -27202.71067028995), (26365.833756425134, -27771.76839137331), (26183.02007196543, -28336.23198529783), (25998.0734221034, -28896.891054298536), (25811.06016813778, -29454.57309734465), (25622.04504212251, -30010.155013994972), (25431.09271613632, -30564.479755058932), (25238.273528957856, -31118.169390760417), (25043.65993771098, -31671.51738379592), (24847.326733772206, -32224.44698363975), (24649.3499650257, -32776.51076236879), (24449.805078428966, -33326.927557443014), (24248.766106887953, -33874.668534707314), (24046.30447572977, -34418.5672408279), (23842.48912626935, -34957.41228042734), (23637.38847697842, -35490.01785866259), (23431.07087496669, -36015.26700533271), (23223.605226332395, -36532.122995047765), (23015.062155231986, -37039.69304829735), (22805.51286841074, -37537.23149000969), (22595.029493795635, -38024.206394517845), (22383.685717516062, -38500.31341214573), (22171.555145275088, -38965.47793885212), (21958.711625086056, -39419.77591444924), (21745.227981568474, -39863.33197886639), (21531.17530241231, -40296.272379811555), (21316.62034565267, -40718.72425256202), (21101.625067376473, -41130.85424475313), (20886.244950616972, -41532.90053967645), (20670.528296927027, -41925.181907298705), (20454.514491222595, -42308.098916628536), (20238.233228618374, -42682.129754521295), (20021.703838616748, -43047.82156082654), (19804.935110313876, -43405.777208429), (19587.923489520617, -43756.63981348562), (19370.653076912487, -44101.07743811963), (19153.095702746075, -44439.770913290085), (18935.209805066443, -44773.40524367561), (18716.9398487121, -45102.66301024712), (18498.216685456704, -45428.21839499933), (18278.95804503424, -45750.7317837815), (18059.066937723182, -46070.84467233421), (17838.53025339703, -46389.572681782905), (17617.3531452327, -46707.89300833122), (17395.55299905272, -47026.875068124056), (17173.12996278583, -47347.52132814383), (16950.062498726096, -47670.7720268842), (16726.359451074793, -47997.71020512831), (16502.10461212026, -48329.75262629323), (16277.309800544775, -48668.125517211614), (16051.927587492719, -49013.8382879567), (15825.870846547587, -49367.66217040165), (15599.01080204143, -49730.03038666877), (15371.21210589716, -50101.13625585932), (15142.344514598582, -50480.955369837866), (14912.28412564363, -50869.244443087046), (14680.916009197474, -51265.5121711029), (14448.134361542285, -51669.06383313645), (14213.842800315691, -52079.03664392228), (13977.955398016205, -52494.44827620843), (13740.396942115614, -52914.257271060975), (13501.103265509817, -53337.434454598006), (13260.021197568345, -53763.03883661713), (13017.109515042222, -54190.288158388204), (12772.339097298744, -54618.642037452206), (12525.694090030105, -55047.857125018876), (12277.171267444795, -55478.02980580947), (12026.781115375648, -55909.60874066856), (11774.547999049071, -56343.34797332402), (11520.510091863685, -56780.180779705166), (11264.718888369109, -57221.05502692839), (11007.237719764373, -57666.79942415264), (10748.139302180833, -58118.06289468658), (10487.505090380004, -58575.28192058147), (10225.420339640012, -59038.66814442613), (9961.972867197743, -59508.21996594944), (9697.248508678287, -59983.75542151179), (9431.326994261148, -60464.96510139334), (9164.279434172971, -60951.46354079042), (8896.165264881623, -61442.82121686939), (8627.027512404813, -61938.593331131044), (8356.888464851127, -62438.341096103926), (8085.7440090816235, -62941.64617206276), (7813.555547973994, -63448.120835388116), (7540.241965033159, -63957.41723919908), (7265.668645559137, -64469.237708194436), (6989.63223159285, -64983.34471049795), (6711.239766984807, -65498.504365695575), (6430.08514773776, -66014.618)]



  mpc4_xs, mpc4_ys = zip(*mpc4)
  bc_xs, bc_ys = zip(*bc)
  lstm_xs, lstm_ys = zip(*lstm)
  plt.plot(mpc4_xs, mpc4_ys, 'b-', label="MPC")
  plt.plot(bc_xs, bc_ys, 'g-',label="BC")
  plt.plot(lstm_xs, lstm_ys, 'm-', label="LSTM")
  plt.legend()
  plt.savefig('./eval/graphs/trajectories.jpg')
  plt.close()

    
