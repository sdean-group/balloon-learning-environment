import scipy.interpolate
from balloon_learning_environment.agents import agent, opd
from balloon_learning_environment.env.balloon.jax_balloon import JaxBalloon, JaxBalloonState
from balloon_learning_environment.env.wind_field import JaxWindField
from balloon_learning_environment.env.balloon.standard_atmosphere import JaxAtmosphere
from balloon_learning_environment.env.balloon import solar
from balloon_learning_environment.utils import jax_utils, spherical_geometry, units
from balloon_learning_environment.models import jax_perciatelli
import s2sphere as s2
import numpy as np
import jax
import jax.numpy as jnp
import scipy
from functools import partial
import datetime as dt
import time
import json
import optax

def inverse_sigmoid(x):
    return jnp.log((x+1)/(1-x))

def sigmoid(x):
    """Returns a sigmoid function with range [-1, 1] instead of [0,1]"""
    return 2 / (1 + jnp.exp(-x)) - 1

def jax_balloon_cost(balloon: JaxBalloon):
    # d = jnp.sqrt((balloon.state.x/1000)**2 + (balloon.state.y/1000)**2)
    # return jnp.exp((d-100)/20)

    r_2 = (balloon.state.x/1000)**2 + (balloon.state.y/1000)**2
    
    soc = balloon.state.battery_charge / balloon.state.battery_capacity
    
    battery_cost = 50**2 * (1 -  (1 / (1 + jnp.exp(-100*(soc - 0.1)))))

    return r_2 + battery_cost

#cost function from paper doesnt seem to work well.
"""sgd stats:
seed=0, cumulative_reward=116.58868254285353, time_within_radius=0.071875, out_of_power=False, final_timestep=960)
I0623 17:43:51.871635 110232 eval_lib.py:300] Power safety layer violations: 23
I0623 17:43:51.871635 110232 eval_lib.py:301] Altitude safety layer violations: 0
I0623 17:43:51.871635 110232 eval_lib.py:302] Envelope safety layer violations: 0

"""
def jax_balloon_cost2(balloon: JaxBalloon):
    r_2 = (balloon.state.x/1000)**2 + (balloon.state.y/1000)**2
    
    soc = balloon.state.battery_charge / balloon.state.battery_capacity
    
    denom = 1 + jnp.exp((jnp.sqrt(r_2)/50**2) - 1)

    return (1+ jnp.exp(-1)) * (soc**0.9)/denom

#adapted cost from bellemare paper
def jax_balloon_cost3(balloon: JaxBalloon):
    #test cost 1
    r_2 = (balloon.state.x/1000)**2 + (balloon.state.y/1000)**2
    #minimal if within radius

    # Smooth ramp outside radius
    # penalty_outside = jnp.where(
    #     r_2 > 50**2,
    #     (r_2 - 50**2) ** 2,
    #     0.0
    # )

    # # SOC factor (optional, or tweak as you like)
    # soc = balloon.state.battery_charge / balloon.state.battery_capacity
    # battery_penalty = 1.0 - 1.0 / (1 + jnp.exp(-100.0 * (soc - 0.1)))
    # return penalty_outside + battery_penalty

    #test cost 2
    r_dist = jax.lax.cond(
        r_2 <= 50**2,
        lambda _: 1.0,
        lambda _: 0.4 * jnp.power(2.0, -((r_2 - 2500)/100.0)),
        operand=None
    )
    
    
    soc = balloon.state.battery_charge / balloon.state.battery_capacity

    f_w = 0.95 - 0.3*soc
    return f_w * r_dist
    



class TerminalCost:
    def __call__(self, balloon: JaxBalloon, wind_forecast: JaxWindField):
        pass

class QTerminalCost(TerminalCost):
    def __init__(self, num_wind_layers, distilled_params):
        self.num_wind_layers = num_wind_layers
        self.distilled_params = distilled_params

    def __call__(self, balloon: JaxBalloon, wind_forecast: JaxWindField):
        model = jax_perciatelli.DistilledNetwork()
        feature_vector = jax_perciatelli.jax_construct_feature_vector(balloon, wind_forecast, self.get_input_size(), self.num_wind_layers)
        q_vals = model.apply(self.distilled_params, feature_vector)
        terminal_cost = -(jnp.mean(q_vals)**2) # NOTE: can also test with max(Q_values)
        return terminal_cost
    
    def get_input_size(self):
        return jax_perciatelli.get_distilled_model_input_size(self.num_wind_layers)
    
    def tree_flatten(self): 
        return (self.distilled_params, ), {'num_wind_layers': self.num_wind_layers}

    @classmethod
    def tree_unflatten(cls, aux_data, children): 
        q_func = QTerminalCost(aux_data['num_wind_layers'], children[0])
        return q_func

jax.tree_util.register_pytree_node_class(QTerminalCost)

class NoTerminalCost(TerminalCost):
    def __call__(self, balloon: JaxBalloon, wind_forecast: JaxWindField):
        return 0.0
    
    def tree_flatten(self):
        return (), {}
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return NoTerminalCost()

jax.tree_util.register_pytree_node_class(NoTerminalCost)

@partial(jax.jit, static_argnums=(5,6))
def jax_plan_cost(plan, balloon, wind_field, atmosphere, terminal_cost_fn, time_delta, stride):
    return jax_plan_cost_no_jit(plan, balloon, wind_field, atmosphere, terminal_cost_fn,  time_delta, stride)

def jax_plan_cost_no_jit(plan, balloon: JaxBalloon, wind_field: JaxWindField, atmosphere: JaxAtmosphere, terminal_cost_fn: TerminalCost, time_delta: 'int, seconds', stride: 'int, seconds'):
    cost = 0.0
    discount_factor = 0.99

    plan = sigmoid(plan)
    def update_step(i, balloon_and_cost: tuple[JaxBalloon, float]):
        balloon, cost = balloon_and_cost
        #jax.debug.print("update_step: i = {i}", i=i)

        wind_vector = wind_field.get_forecast(balloon.state.x/1000, balloon.state.y/1000, balloon.state.pressure, balloon.state.time_elapsed)
        
        
        # action = plan[i]
        action = jax.lax.cond(((balloon.state.battery_charge/balloon.state.battery_capacity) < 0.025),
                     lambda op: jnp.array(0.0, dtype=jnp.float64),
                     lambda op: op[0],
                     operand=(plan[i],))
        
        #jax.debug.print("action[{i}] = {a}", i=i, a=action)

        next_balloon = balloon.simulate_step_continuous_no_jit(wind_vector, atmosphere, action, time_delta, stride)

        #jax.debug.print("state[{i}] x = {x}, y = {y}, pressure = {p}, mols = {m}", i=i, x=next_balloon.state.x, y=next_balloon.state.y, p=next_balloon.state.pressure, m=next_balloon.state.mols_air)

        cost += (discount_factor**i) * jax_balloon_cost(next_balloon)

        return next_balloon, cost
    #jax.debug.print("entering loop")
    final_balloon, cost = jax.lax.fori_loop(0, len(plan), update_step, init_val=(balloon, cost))
    #jax.debug.print("final state x = {x}, y = {y}, pressure = {p}", x=final_balloon.state.x, y=final_balloon.state.y, p=final_balloon.state.pressure)
    terminal_cost = (discount_factor**len(plan)) * (jax_balloon_cost(final_balloon) + terminal_cost_fn(final_balloon, wind_field))
    return cost + terminal_cost

def backtrack(x0, grad, balloon, forecast, atmosphere, terminal_cost_fn, expected_remaining_normalized_charge, time_delta, stride, t, alpha, beta, count):
    """ t is the step size"""
    t=0.01
    iters = 0
    while (iters < count and jax_plan_cost(x0 - t*grad, balloon, forecast, atmosphere, terminal_cost_fn, expected_remaining_normalized_charge, time_delta, stride) > jax_plan_cost(x0, balloon, forecast, atmosphere, terminal_cost_fn, expected_remaining_normalized_charge, time_delta, stride)-alpha*t*jnp.dot(grad, grad)):
        t = beta * t
        iters+=1
    return t

def grad_descent_optimizer(initial_plan, dcost_dplan, balloon, forecast, atmosphere, terminal_cost_fn, expected_remaining_normalized_charge, time_delta, stride, alpha=0.01,max_iters=80, isNormalized=0):
    """Basic gradient descent with just a learning rate and also normalizing the gradient. Doing both the learning rate and normalization seemed to perform worse and take longer. High learning rates had power violations. Slower learning rates performed better than baseline but took more time (almost 4s)."""
    print(f"USING LEARNING RATE {alpha}, MAXITERS {max_iters}, ISNORMALIZED {isNormalized}")
    plan = initial_plan

    for gradient_steps in range(max_iters):
        gradient_plan = dcost_dplan(plan, balloon, forecast, atmosphere, terminal_cost_fn, time_delta, stride)

        if  np.isnan(gradient_plan).any() or abs(jnp.linalg.norm(gradient_plan)) < 1e-7:
            # print('Exiting early, |∂plan| =',abs(jnp.linalg.norm(dplan)))
            break
        # print("A", gradient_steps, abs(jnp.linalg.norm(dplan)))
        #plan -= dplan / jnp.linalg.norm(dplan)
        alpha = backtrack(plan, gradient_plan, balloon, forecast, atmosphere, terminal_cost_fn, expected_remaining_normalized_charge, time_delta, stride, 1, 0.5, 0.8, 10)
        if isNormalized==1:
            plan -= alpha * gradient_plan / jnp.linalg.norm(gradient_plan)
        else:
            plan -= alpha * gradient_plan

    return plan, gradient_steps


def grad_descent_optimizer_with_momentum(
    initial_plan,
    dcost_dplan,
    balloon,
    forecast,
    atmosphere,
    terminal_cost_fn,
    expected_remaining_normalized_charge,
    time_delta,
    stride,
    learning_rate=1,
    momentum=0.99,
    max_steps=100,
    isNormalized=1
):
    """ No power violation problem compared to gradient descent unless the learning rate was too slow and the momentum was high. High momentum seems important. Performance improvement with learning_rate=1, and normalization. Similar performance with lower learning rate and no normalization but slightly longer time. Could try more combos with less iterations"""
    print(f"USING LEARNING RATE {learning_rate}, MOMENTUM {momentum}, MAXITERS {max_steps}, ISNORMALIZED {isNormalized}")
    plan = initial_plan
    velocity = jnp.zeros_like(plan)  # Initialize momentum vector


    for step in range(max_steps):
        gradient_plan = dcost_dplan(plan, balloon, forecast, atmosphere, terminal_cost_fn, expected_remaining_normalized_charge, time_delta, stride)

        if jnp.isnan(gradient_plan).any() or abs(jnp.linalg.norm(gradient_plan)) < 1e-7:
            break

        # Update momentum
        if isNormalized:
            velocity = momentum * velocity + (gradient_plan / jnp.linalg.norm(gradient_plan))
        else:
            velocity = momentum * velocity + (gradient_plan)

        # Update plan
        plan -= learning_rate * velocity

    return plan, step

def sgd(initial_plan, dcost_dplan, balloon: JaxBalloon, forecast, atmosphere: JaxAtmosphere, terminal_cost_fn, time_delta, stride, learning_rate=0.1, momentum=0.99, max_steps=40, isNormalized=1):
    """ SGD with momentum. Has similar performance with lower times (less iterations) compared to adam. High momentum, learning rate, and nesterov being true seems to be important. All the eval files were results with normalization. Without normalizing there's Nan's in the plan"""
    print(f"USING LEARNING RATE {learning_rate}, MOMENTUM {momentum}, MAXITERS {max_steps}, ISNORMALIZED {isNormalized}")
    solver = optax.sgd(learning_rate=learning_rate, momentum=momentum, nesterov=True)
    opt_state = solver.init(initial_plan)
    plan = initial_plan

    def update_step(plan, opt_state):
        gradient_plan = dcost_dplan(plan, balloon, forecast, atmosphere, terminal_cost_fn, time_delta, stride)
        #print(f"gradient plan is {gradient_plan}")
        if isNormalized==1:
            updates, opt_state = solver.update(gradient_plan/jnp.linalg.norm(gradient_plan), opt_state, plan)
        else:
            updates, opt_state = solver.update(gradient_plan, opt_state, plan)

        plan = optax.apply_updates(plan, updates)
        return plan, opt_state

    for steps in range(max_steps):
        prev_plan = plan
        plan, opt_state = update_step(plan, opt_state)
        if jnp.isnan(plan).any() or jnp.isinf(plan).any():
            print("Plan has NaN or Inf, stopping to prevent instability.")
            return prev_plan, steps
    return plan, steps

def sgd_constrained(initial_plan, dcost_dplan, balloon: JaxBalloon, forecast, atmosphere: JaxAtmosphere, terminal_cost_fn, time_delta, stride, learning_rate=0.1, momentum=0.99, max_steps=40, isNormalized=1):
    """ SGD with momentum with constrained parameters. Has similar performance with lower times (less iterations) compared to adam. High momentum, learning rate, and nesterov being true seems to be important. All the eval files were results with normalization. Without normalizing there's Nan's in the plan"""
    print(f"USING LEARNING RATE {learning_rate}, MOMENTUM {momentum}, MAXITERS {max_steps}, ISNORMALIZED {isNormalized}")
    #print(f"initial plan is {initial_plan[:15]}")
    if jnp.isnan(initial_plan).any() or jnp.isinf(initial_plan).any():
        print('----------------INIT PLAN HAS NANS OR INFS---------------------')
    # The initial plan has to be a feasible starting point
    solver = optax.sgd(learning_rate=learning_rate, momentum=momentum, nesterov=True)
    params = {'plan': initial_plan}
    opt_state = solver.init(params)

    def update_step(params, opt_state):
        gradient_plan = dcost_dplan(params['plan'], balloon, forecast, atmosphere, terminal_cost_fn, time_delta, stride)
        grad_params = {'plan': gradient_plan}
        if isNormalized==1:
            grad_params = {'plan': gradient_plan/jnp.linalg.norm(gradient_plan)}
            updates, opt_state = solver.update(grad_params, opt_state, params)
        else:
            updates, opt_state = solver.update(grad_params, opt_state, params)

        params = optax.apply_updates(params, updates)
        params = optax.projections.projection_box(params,lower={'plan':0.0}, upper={'plan':0.9999})
        return params, opt_state

    for steps in range(max_steps):
        prev = params
        params, opt_state = update_step(params, opt_state)
        if jnp.isnan(params['plan']).any() or jnp.isinf(params['plan']).any():
            print("Plan has NaN or Inf, stopping to prevent instability.")
            return prev['plan'],steps
    return params['plan'], steps

def adam_optimizer(initial_plan, dcost_dplan, balloon: JaxBalloon, forecast, atmosphere: JaxAtmosphere, terminal_cost_fn, time_delta, stride, learning_rate=1, max_steps=50, isNormalized=1):
    """ Higher alpha seems better (went up to 0.9 unsure if its okay to go over 1). Less iterations does worse but takes less time (2s compared to 4s). Normalizing helped a little bit for alpha=0.1 only. In general this seems to do better with more aggressive steps?"""
    print(f"USING LEARNING RATE {learning_rate}, MAXITERS {max_steps}, ISNORMALIZED {isNormalized}")
    solver = optax.adam(learning_rate=learning_rate)
    opt_state = solver.init(initial_plan)
    plan = initial_plan

    def update_step(plan, opt_state):
        gradient_plan = dcost_dplan(plan, balloon, forecast, atmosphere, terminal_cost_fn, time_delta, stride)
        if isNormalized == 1:
            updates, opt_state = solver.update(gradient_plan/jnp.linalg.norm(gradient_plan), opt_state, plan)
        else:
            updates, opt_state = solver.update(gradient_plan, opt_state, plan)
        plan = optax.apply_updates(plan, updates)
        return plan, opt_state

    for adam_step in range(max_steps):
        plan, opt_state = update_step(plan, opt_state)
        if jnp.isnan(plan).any() or jnp.isinf(plan).any():
            print("Plan has NaN or Inf, stopping to prevent instability.")
            break
    return plan, adam_step

def adabelief_optimizer(initial_plan, balloon: JaxBalloon, forecast, atmosphere: JaxAtmosphere, terminal_cost_fn, time_delta, stride, learning_rate=1, max_steps=10, isNormalized=1):
    """ Needed to make separate function since optax had more specific PyTree structure requirements for adabelief. Normalizing doesn't seem to add much benefit. Like adam, higher learning rates seem to do better"""
    #print(f"USING LEARNING RATE {learning_rate}, MAXITERS {max_steps}, ISNORMALIZED {isNormalized}")
    initial_plan = {"params": jnp.array(initial_plan)}
    def cost_fn(plan,balloon, forecast, atmosphere, terminal_cost_fn, time_delta, stride):
        return jax_plan_cost(plan["params"], balloon, forecast, atmosphere, terminal_cost_fn, time_delta, stride)
    
    dcost_dplan = jax.grad(cost_fn)
    solver = optax.adabelief(learning_rate=learning_rate)
    opt_state = solver.init(initial_plan)
    plan = initial_plan

    def update_step(plan, opt_state):
        gradient_plan = dcost_dplan(plan, balloon, forecast, atmosphere, terminal_cost_fn, time_delta, stride)
        grad = {"params": gradient_plan["params"]}
        if isNormalized == 1:
            gradient_norm = jnp.linalg.norm(gradient_plan["params"])
            grad = {"params": gradient_plan["params"] / (gradient_norm + 1e-8)}
            
        updates, opt_state = solver.update(grad, opt_state, plan)
        plan = optax.apply_updates(plan, updates)
        return plan, opt_state


#this was an attempt to make the loops faster with jit compatibility. 
    # def update_step(state):
    #     plan, opt_state, step, max_steps = state

    #     gradient_plan = dcost_dplan(plan, balloon, forecast, atmosphere, terminal_cost_fn, time_delta, stride)
    #     gradient_norm = jnp.linalg.norm(gradient_plan["params"])
    #     normalized_grad = {"params": gradient_plan["params"] / (gradient_norm + 1e-8)}
    #     updates, opt_state = solver.update(gradient_plan, opt_state, plan)
    #     plan = optax.apply_updates(plan, updates)
    #     step += 1
    #     return (plan, opt_state, step, max_steps)
    
    # def is_stable(state):
    #     plan, _, step, max_steps = state
    #     finite = jnp.all(jnp.isfinite(plan["params"]))
    #     return jnp.logical_and(step < max_steps, finite)
    # state = (initial_plan, opt_state, 0, max_steps)
    #plan, opt_state, final_step, _ = jax.lax.while_loop(is_stable, update_step, state)

    for ada_step in range(max_steps):
        plan, opt_state = update_step(plan, opt_state)
        if jnp.isnan(plan["params"]).any() or jnp.isinf(plan["params"]).any():
            print("Plan has NaN or Inf, stopping to prevent instability.")
            break
   
    return plan["params"], ada_step



#------------------------- 2ND Order Optimizers ----------------------------------

def lbfgs_optimizer(initial_plan, balloon: JaxBalloon, forecast, atmosphere: JaxAtmosphere, terminal_cost_fn, expected_remaining_normalized_charge,time_delta, stride, max_steps=1):
    """ literally takes too long so i dont even know if it works"""
    #@partial(jax.jit, static_argnums=())
    def cost_fn(plan):
        return jax_plan_cost_no_jit(plan, balloon, forecast, atmosphere, terminal_cost_fn, expected_remaining_normalized_charge, time_delta, stride)
    solver = optax.lbfgs()
    opt_state = solver.init(initial_plan)
    plan = initial_plan
    value_and_grad_fn = optax.value_and_grad_from_state(cost_fn)

    def update_step(plan, opt_state):
        value, grad = value_and_grad_fn(plan, state=opt_state)
        updates, opt_state = solver.update(grad, opt_state, plan, value=value, grad=grad, value_fn=cost_fn)
        plan = optax.apply_updates(plan, updates)
        return plan, opt_state

    for steps in range(max_steps):
        plan, opt_state = update_step(plan, opt_state)
        if jnp.isnan(plan).any() or jnp.isinf(plan).any():
            print("Plan has NaN or Inf, stopping to prevent instability.")
            break
    return plan, steps

def newton_method(initial_plan, dcost_dplan, d2cost_d2plan, balloon: JaxBalloon, forecast, atmosphere: JaxAtmosphere, terminal_cost_fn, expected_remaining_normalized_charge, time_delta, stride, max_steps=30, frequency=10):
    print(f"USING MAXITERS {max_steps}, FREQUENCY {frequency}")
    plan = initial_plan
    #hessian_plan = hess_fn(plan, balloon, forecast, atmosphere, terminal_cost_fn, time_delta, stride)

    for step in range(max_steps):
        gradient_plan = dcost_dplan(plan, balloon, forecast, atmosphere, terminal_cost_fn, expected_remaining_normalized_charge, time_delta, stride)
        if step % frequency == 0:
            hessian_plan = d2cost_d2plan(plan, balloon, forecast, atmosphere, terminal_cost_fn, expected_remaining_normalized_charge, time_delta, stride)
        
            
        if jnp.isnan(gradient_plan).any() or jnp.isinf(gradient_plan).any() or jnp.isinf(hessian_plan).any() or abs(jnp.linalg.norm(gradient_plan)) < 1e-7:
            print("Converged after", step+1, "iterations.")
            break

        # Update plan
        #regularizing
        epsilon = 1e-3
        hessian_reg = hessian_plan + epsilon * jnp.eye(hessian_plan.shape[0])

        delta = jnp.linalg.solve(hessian_reg, gradient_plan)

        if jnp.linalg.norm(delta) > 1e5:
            print("Skipping update: delta too large")
            break
        if jnp.isnan(delta).any() or jnp.isinf(delta).any():
            print("Delta contains NaNs or infs. Gradient norm:", jnp.linalg.norm(gradient_plan))
            break
        plan -= delta

    
    return plan, step


#@partial(jax.jit, static_argnames=['optimizer'])
# jit has some issues with getting the cost
def run_optimizer(optimizer:str, initial_plan, dcost_dplan, d2cost_d2plan, balloon: JaxBalloon, forecast, atmosphere: JaxAtmosphere, terminal_cost_fn, time_delta, stride, hyperparams=[]):
    """ Requires that evaluation passes in the correct number and order of hyperparams for the optimizer it chooses. No checking is enforced here. """
    start_cost = jax_plan_cost(initial_plan, balloon, forecast, atmosphere, terminal_cost_fn, time_delta, stride)
    plan = initial_plan
    if (optimizer == 'adabelief'):
        plan,step= adabelief_optimizer(initial_plan, balloon, forecast, atmosphere, terminal_cost_fn, time_delta, stride)
    elif (optimizer == 'sgd'):
        plan,step= sgd(initial_plan, dcost_dplan, balloon, forecast, atmosphere, terminal_cost_fn, time_delta, stride)
    elif (optimizer=='grad'):
        print(hyperparams)
        plan,step= grad_descent_optimizer(initial_plan,dcost_dplan, balloon, forecast, atmosphere, terminal_cost_fn, time_delta, stride)
    elif (optimizer=='momentum'):
        plan,step= grad_descent_optimizer_with_momentum(initial_plan, dcost_dplan, balloon, forecast, atmosphere, terminal_cost_fn, time_delta, stride)
    elif (optimizer=='lbfgs'):
        plan,step= lbfgs_optimizer(initial_plan, balloon, forecast, atmosphere, terminal_cost_fn, time_delta, stride)
    elif (optimizer=='newton'):
        plan,step= newton_method(initial_plan, dcost_dplan, d2cost_d2plan ,balloon, forecast, atmosphere, terminal_cost_fn, time_delta, stride)
    else:
        plan,step= adam_optimizer(initial_plan, dcost_dplan, balloon, forecast, atmosphere, terminal_cost_fn, time_delta, stride)
    
    after_cost = jax_plan_cost(plan, balloon, forecast, atmosphere, terminal_cost_fn, time_delta, stride)
    #print(f"{optimizer} optimizer, {step}, ∆cost = {after_cost} - {start_cost} = {after_cost - start_cost}")
    #print(plan[:15])
    #print(f'Took {step + 1} iterations')
    return plan, step+1


np.random.seed(seed=42)
def get_initial_plans(balloon: JaxBalloon, num_plans, forecast: JaxWindField, atmosphere: JaxAtmosphere, plan_steps, time_delta, stride):
    # flight_record = [(atmosphere.at_pressure(balloon.state.pressure).height.km, 0)]
    flight_record = {atmosphere.at_pressure(balloon.state.pressure).height.km.item(): 0}

    time_to_top = 0
    max_km_to_explore = 19.1

    up_balloon = balloon
    while time_to_top < plan_steps and atmosphere.at_pressure(up_balloon.state.pressure).height.km < max_km_to_explore:
        wind_vector = forecast.get_forecast(up_balloon.state.x/1000, up_balloon.state.y/1000, up_balloon.state.pressure, up_balloon.state.time_elapsed)
        up_balloon = up_balloon.simulate_step_continuous(wind_vector, atmosphere, 0.99, time_delta, stride)
        time_to_top += 1

        flight_record[atmosphere.at_pressure(up_balloon.state.pressure).height.km.item()] = time_to_top

    time_to_bottom = 0
    min_km_to_explore = 15.4

    down_balloon = balloon
    while time_to_bottom < plan_steps and atmosphere.at_pressure(down_balloon.state.pressure).height.km > min_km_to_explore:
        wind_vector = forecast.get_forecast(down_balloon.state.x/1000, down_balloon.state.y/1000, down_balloon.state.pressure, down_balloon.state.time_elapsed)
        down_balloon = down_balloon.simulate_step_continuous(wind_vector, atmosphere, -0.99, time_delta, stride)
        time_to_bottom += 1

        flight_record[atmosphere.at_pressure(down_balloon.state.pressure).height.km.item()] = time_to_bottom
    
    # sorted (should be)
    # flight_record = flight_record_down[::-1] + flight_record_up

    # Sort the dictionary by keys (altitudes) and split them into two separate lists
    sorted_flight_record = sorted(flight_record.items())

    flight_record_altitudes = [altitude for altitude, _ in sorted_flight_record]
    flight_record_steps = [steps for _, steps in sorted_flight_record]
    
    interpolator = scipy.interpolate.RegularGridInterpolator((flight_record_altitudes, ), flight_record_steps, bounds_error=False, fill_value=None)

    plans = []

    for i in range(num_plans):
        random_height = np.random.uniform(15.4, 19.1)
        going_up = random_height >= atmosphere.at_pressure(balloon.state.pressure).height.km
        steps = int(round(interpolator(np.array([random_height]))[0]))
        # print(steps)

        plan = np.zeros((plan_steps, ))
        plan[:steps] = +0.99 if going_up else -0.99 
        # print(random_height, steps)
        try:
            if steps < plan_steps:
                plan[steps:] += np.random.uniform(-0.3, 0.3, plan_steps - steps)
        except:
            print(atmosphere.at_pressure(balloon.state.pressure).height.km.item(), random_height, steps, plan_steps)

        plans.append(plan)
    
    return inverse_sigmoid(np.array(plans))


@partial(jax.jit, static_argnums=(5,6))
@partial(jax.grad, argnums=0)
def get_dplan(plan, balloon: JaxBalloon, wind_field: JaxWindField, atmosphere: JaxAtmosphere, terminal_cost_fn: TerminalCost, time_delta, stride):
     # jax.debug.print("{balloon}, {wind_field}, {atmosphere}, {terminal_cost_fn}, {time_delta}, {stride}", balloon=balloon, wind_field=wind_field, atmosphere=atmosphere, terminal_cost_fn=terminal_cost_fn, time_delta=time_delta, stride=stride)
    return jax_plan_cost_no_jit(plan, balloon, wind_field, atmosphere, terminal_cost_fn, time_delta, stride)

@partial(jax.jit, static_argnums=(5,6))
@partial(jax.hessian, argnums=0)
def get_d2plan(plan, balloon: JaxBalloon, wind_field: JaxWindField, atmosphere: JaxAtmosphere, terminal_cost_fn: TerminalCost, time_delta, stride):
    return jax_plan_cost_no_jit(plan, balloon, wind_field, atmosphere, terminal_cost_fn, time_delta, stride)

class MPC4Agent(agent.Agent):
        
    def __init__(self, num_actions: int, observation_shape): # Sequence[int]
        super(MPC4Agent, self).__init__(num_actions, observation_shape)
        self.forecast = None # WindField
        self.ble_atmosphere = None 
        self.atmosphere = None # Atmosphere

        # self._get_dplan = jax.jit(jax.grad(jax_plan_cost, argnums=0), static_argnames=("time_delta", "stride"))

        self.plan_time = 2*24*60*60 # 2 days in seconds
        self.time_delta = 3*60 # 3min
        self.stride = 10

        # self.plan_steps = 960 + 23 
        self.plan_steps = 160 # (self.plan_time // self.time_delta) // 3
        self.horizon = 47 #og was 23
        # self.N = self.plan_steps

        self.plan = None # jnp.full((self.plan_steps, ), fill_value=1.0/3.0)
        self.i = 0

        self.key = jax.random.key(seed=0)

        self.avg_opt_time=0
        self.avg_iters =0
        self.optimizer="sgd" #default optimizer for now, should be passed in
        self.hyperparams = [0,0,0,0] #hyperparams that should be passed in
        self.expected_remaining_normalized_charge = jnp.astype(1.0, jnp.float64)
        self.isNight = False

        self.balloon = None
        self.time = None
        self.steps_within_radius = 0


        using_Q_function = False

        if using_Q_function:
            self.num_wind_levels = 181
            params = jax_perciatelli.get_distilled_perciatelli(self.num_wind_levels)[0]
            self.terminal_cost_fn = QTerminalCost(self.num_wind_levels, params)
        else:
            self.terminal_cost_fn = NoTerminalCost()

    def check_action(self, action):
        """ Does the night power checks from power_safety.py. """
        night_power:units.Power = units.Power(watts=self.balloon.state.nighttime_power_load)
        center_latlng = s2.LatLng(
            float(self.balloon.state.center_latlng.lat),
            float(self.balloon.state.center_latlng.lng)) 
        x = units.Distance(m=self.balloon.state.x)
        y = units.Distance(m=self.balloon.state.y)
        latlng = spherical_geometry.calculate_latlng_from_offset(
        center_latlng, x,y)

        if (latlng.lat().degrees < 60.0):
            timestamp = float(self.balloon.state.date_time) 
            date_time = dt.datetime.fromtimestamp(timestamp,dt.timezone.utc)
            sunrise, sunset = solar.get_next_sunrise_sunset(
            latlng, date_time)
            time_hysteresis = dt.timedelta(minutes=30)
            sunrise_with_hysteresis = sunrise + time_hysteresis
            time_to_sunrise = sunrise_with_hysteresis - date_time
            floating_charge = night_power * time_to_sunrise
            jax_floating_charge = jnp.astype(floating_charge.watt_hours, jnp.float64)

            self.expected_remaining_normalized_charge= (self.balloon.state.battery_charge - jax_floating_charge) / self.balloon.state.battery_capacity

            #print(f"soc is: {self.balloon.state.battery_charge/self.balloon.state.battery_capacity}")
            #print(f"CALCULATED expected charge: { self.expected_remaining_normalized_charge}, isNight {sunset >= sunrise_with_hysteresis}")
            #theres some discrepancy btwn calculated and what balloon has i think that would fix the occuring violations. This is why we have a conservative limit of 0.09 instead of 0.025
            # Check that it is night
            self.isNight = sunset >= sunrise_with_hysteresis
            if (sunset >= sunrise_with_hysteresis and 
                (self.balloon.state.battery_charge/self.balloon.state.battery_capacity < 0.025 
            or self.expected_remaining_normalized_charge < 0.09)):
                print('forcing SIGMOID')
                #print(f"curr plan {self.plan[:10]}")
                clipped_plan = np.clip(sigmoid(self.plan), 0, 1)
                # need to make a feasible plan that satisfies constraints
                start_cost = jax_plan_cost(clipped_plan, self.balloon, 
                    self.forecast, 
                    self.atmosphere,
                    self.terminal_cost_fn,
                    self.time_delta, 
                    self.stride)
                before = time.time()
    
                self.plan, iters = sgd_constrained(
                    clipped_plan, 
                    get_dplan,
                    self.balloon, 
                    self.forecast, 
                    self.atmosphere,
                    self.terminal_cost_fn,
                    self.time_delta, 
                    self.stride)
                after = time.time()
                after_cost = jax_plan_cost(self.plan, self.balloon, 
                    self.forecast, 
                    self.atmosphere,
                    self.terminal_cost_fn,
                    self.time_delta, 
                    self.stride)
                
                
                #print(f"CONSTRAINED: {step}, ∆cost = {after_cost} - {start_cost} = {after_cost - start_cost}")
                self.avg_opt_time += after-before
                self.avg_iters += iters
                action = self.plan[self.i]
                #print(f"constrained plan is {self.plan[:10]}, action is {action}")
                return action.item()
                #return 0.01 #cant do 0 cause discrete commands are 0 (down) 1 (stay) 2 (up) so this forces continous step which is [-1, 1]
        return action.item()

    """
    calculates the current position of a moving object by using a previously determined position, or fix, and incorporating estimates of speed, heading, and elapsed time.
    an observation is a state. it has balloon_obersvation and wind information. we use ob_t and MPC predicts a_t which then gives us ob_t+1
    """
    def _deadreckon(self):
        # wind_vector = self.ble_forecast.get_forecast(
        #     units.Distance(meters=self.balloon.state.x),
        #     units.Distance(meters=self.balloon.state.y), 
        #     self.balloon.state.pressure,
        #     dt.datetime())
        
        # wind_vector = wind_vector.u.meters_per_second, wind_vector.v.meters_per_second
        
        wind_vector = self.forecast.get_forecast(
            self.balloon.state.x/1000, 
            self.balloon.state.y/1000, 
            self.balloon.state.pressure, 
            self.balloon.state.time_elapsed)
    
        # print(self.balloon.state.time_elapsed/3600.0)

        # print(self.balloon.state.time_elapsed)
        self.balloon = self.balloon.simulate_step_continuous(
            wind_vector, 
            self.atmosphere, 
            self.plan[self.i], 
            self.time_delta, 
            self.stride)
        
        if (self.balloon.state.x/1000)**2 + (self.balloon.state.y/1000)**2 <= (50.0)**2:
            self.steps_within_radius += 1

    def begin_episode(self, observation: np.ndarray) -> int:
        # TODO: actually convert observation into an ndarray (it is a JaxBalloonState, see features.py)
        # balloon = JaxBalloon(jax_balloon_state_from_observation(observation))

        observation: JaxBalloonState = observation
        # if self.balloon is not None:
        #     observation.x = self.balloon.state.x
        #     observation.y = self.balloon.state.y
        self.balloon = JaxBalloon(observation)


        # TODO: is it necessary to pass in forecast when just trying to get to a height?
        
        initialization_type = 'best_altitude'
        #print('USING ' + initialization_type + ' INITIALIZATION')

        if initialization_type == 'opd':
            start = opd.ExplorerState(
                self.balloon.state.x,
                self.balloon.state.y,
                self.balloon.state.pressure,
                self.balloon.state.time_elapsed)

            search_delta_time = 60*60
            best_node, best_node_early = opd.run_opd_search(start, self.forecast, [0, 1, 2], opd.ExplorerOptions(budget=25_000, planning_horizon=240, delta_time=search_delta_time))
            initial_plan =  opd.get_plan_from_opd_node(best_node, search_delta_time=search_delta_time, plan_delta_time=self.time_delta)

        elif initialization_type == 'best_altitude':
            if self.plan==None:
                print(f"First")
                self.avg_opt_time=0
                self.avg_iters =0

            initial_plans = get_initial_plans(self.balloon, 100, self.forecast, self.atmosphere, self.plan_steps, self.time_delta, self.stride)
           
            batched_cost = []
            for i in range(len(initial_plans)):
                # tmp = jax.make_jaxpr(jax_plan_cost, static_argnums=(5, 6))(initial_plans[i], self.balloon, self.forecast, self.atmosphere, self.terminal_cost_fn, self.time_delta, self.stride)
                # print(tmp)

                batched_cost.append(jax_plan_cost(initial_plans[i], self.balloon, self.forecast, self.atmosphere, self.terminal_cost_fn, self.time_delta, self.stride))
            min_index_so_far = np.argmin(batched_cost)
            min_value_so_far = batched_cost[min_index_so_far]

            initial_plan = initial_plans[min_index_so_far]
            isUsingPrevPlan = False
            if self.plan is not None and jax_plan_cost(self.plan, self.balloon, self.forecast, self.atmosphere, self.terminal_cost_fn, self.time_delta, self.stride) < min_value_so_far:
                #print('Using the previous optimized plan as initial plan')
                initial_plan = self.plan
                isUsingPrevPlan = True

            coast = inverse_sigmoid(np.random.uniform(-0.2, 0.2, size=(self.plan_steps, )))
            if jax_plan_cost(coast, self.balloon, self.forecast, self.atmosphere, self.terminal_cost_fn, self.time_delta, self.stride) < min_value_so_far:
                #print('Using the nothing plan as initial plan')
                initial_plan = coast
                

        elif initialization_type == 'random':
            initial_plan = np.random.uniform(-1.0, 1.0, size=(self.plan_steps, ))
        elif initialization_type == 'previous':
            if self.plan==None:
                print(f"First")

                initial_plans = get_initial_plans(self.balloon, 100, self.forecast, self.atmosphere, self.plan_steps, self.time_delta, self.stride)
                
                batched_cost = []
                for i in range(len(initial_plans)):
                    # tmp = jax.make_jaxpr(jax_plan_cost, static_argnums=(5, 6))(initial_plans[i], self.balloon, self.forecast, self.atmosphere, self.terminal_cost_fn, self.time_delta, self.stride)
                    # print(tmp)

                    batched_cost.append(jax_plan_cost(initial_plans[i], self.balloon, self.forecast, self.atmosphere, self.terminal_cost_fn, self.time_delta, self.stride))
                
                min_index_so_far = np.argmin(batched_cost)
                min_value_so_far = batched_cost[min_index_so_far]

                initial_plan = initial_plans[min_index_so_far]
                coast = inverse_sigmoid(np.random.uniform(-0.2, 0.2, size=(self.plan_steps, )))
                if jax_plan_cost(coast, self.balloon, self.forecast, self.atmosphere, self.terminal_cost_fn, self.time_delta, self.stride) < min_value_so_far:
                    #print('Using the nothing plan as initial plan')
                    initial_plan = coast
            else:
                initial_plan = self.plan
                isUsingPrevPlan = True               
        else:
            initial_plan = np.zeros((self.plan_steps, ))

        optimizing_on = True
        if optimizing_on:
            b4 = time.time()

            #Newton specifcally has to be run in combination with some fast first-order method
            if self.optimizer == "newton":
                if isUsingPrevPlan:
                    self.plan, iters= run_optimizer(
                    "newton",
                    initial_plan, 
                    get_dplan,
                    get_d2plan,
                    self.balloon, 
                    self.forecast, 
                    self.atmosphere,
                    self.terminal_cost_fn,
                    self.time_delta, 
                    self.stride,
                    self.hyperparams)
                else: 
                    self.plan, iters= run_optimizer(
                    "sgd",
                    initial_plan, 
                    get_dplan,
                    get_d2plan,
                    self.balloon, 
                    self.forecast, 
                    self.atmosphere,
                    self.terminal_cost_fn,
                    self.time_delta, 
                    self.stride,
                    self.hyperparams)
            else:
                self.plan, iters= run_optimizer(
                    self.optimizer,
                    initial_plan, 
                    get_dplan,
                    get_d2plan,
                    self.balloon, 
                    self.forecast, 
                    self.atmosphere,
                    self.terminal_cost_fn,
                    self.time_delta, 
                    self.stride,
                    self.hyperparams)
            
            
            #print(time.time() - b4, 's to get optimized plan')
            self.plan = sigmoid(self.plan)
            after = time.time()
            #print(after - b4, 's to get optimized plan')
            self.avg_opt_time += after-b4
            self.avg_iters += iters
        else:
            self.plan = sigmoid(initial_plan)

        self.i = 0

        b4 = time.time()
        self._deadreckon()
        # print(time.time() - b4, 's to deadreckon ballooon')

        action = self.plan[self.i]
        #return self.check_action(action)
        # print('action', action)
        return action.item()

    def step(self, reward: float, observation: np.ndarray) -> int:
        REPLANNING = True
        observation: JaxBalloonState = observation
        self.i+=1
        # self._deadreckon()
        # print(observation.battery_charge/observation.battery_capacity)
        if not REPLANNING:
            self._deadreckon()
            action = self.plan[self.i]
            return self.check_action(action)
            #return action.item()
        else:
            
            N = min(len(self.plan), self.horizon)
            if self.i>0 and self.i%N==0:
                # self.plan_steps -= N
                self.key, rng = jax.random.split(self.key, 2)
                # self.plan = self.plan[N:]
                self.plan = inverse_sigmoid(jnp.hstack((self.plan[N:], jax.random.uniform(rng, (N, ), minval=-0.3, maxval=0.3))))
                #print(self.plan.shape)
                return self.begin_episode(observation)
            else:
                # print('not replanning')
                self._deadreckon()
                action = self.plan[self.i]
                return self.check_action(action)
                # print('action', action)
                #return action.item()
            
    def write_diagnostics_start(self, observation, diagnostics):
        if 'mpc4_agent' not in diagnostics:
            diagnostics['mpc4_agent'] = {'x': [], 'y': [], 'z':[], 'wind':[], 'plan':[]}

        observation: JaxBalloonState = observation
        balloon = JaxBalloon(observation)
        
        height = self.atmosphere.at_pressure(balloon.state.pressure).height.km.item()

        diagnostics['mpc4_agent']['x'].append(balloon.state.x.item()/1000)
        diagnostics['mpc4_agent']['y'].append(balloon.state.y.item()/1000)
        diagnostics['mpc4_agent']['z'].append(height)
        # diagnostics['mpc4_agent']['plan'].append(0.0)

        wind_vector = self.forecast.get_forecast(
            balloon.state.x.item()/1000, 
            balloon.state.y.item()/1000, 
            balloon.state.pressure.item(), 
            balloon.state.time_elapsed)
        diagnostics['mpc4_agent']['wind'].append([wind_vector[0].item(), wind_vector[1].item()])

    
    def write_diagnostics(self, diagnostics):
        if 'mpc4_agent' not in diagnostics:
            diagnostics['mpc4_agent'] = {'x': [], 'y': [], 'z':[], 'wind':[], 'plan':[]}
        
        height = self.atmosphere.at_pressure(self.balloon.state.pressure).height.km.item()

        diagnostics['mpc4_agent']['x'].append(self.balloon.state.x.item()/1000)
        diagnostics['mpc4_agent']['y'].append(self.balloon.state.y.item()/1000)
        diagnostics['mpc4_agent']['z'].append(height)
        diagnostics['mpc4_agent']['plan'].append(self.plan[self.i].item())

        wind_vector = self.forecast.get_forecast(
            self.balloon.state.x/1000, 
            self.balloon.state.y/1000, 
            self.balloon.state.pressure, 
            self.balloon.state.time_elapsed)
        diagnostics['mpc4_agent']['wind'].append([wind_vector[0].item(), wind_vector[1].item()])


    def write_diagnostics_end(self, diagnostics):
        if 'mpc4_agent' not in diagnostics:
            diagnostics['mpc4_agent'] = {'x': [], 'y': [], 'z':[], 'wind':[], 'plan':[]}
        
        X = diagnostics['mpc4_agent']['x']
        if len(X) != 0:
            diagnostics['mpc4_agent']['twr'] = self.steps_within_radius/len(X)
        else:
            diagnostics['mpc4_agent']['twr'] = 0 
 
    def end_episode(self, reward: float, terminal: bool = True) -> None:
        self.i = 0
        self.steps_within_radius = 0
        self.balloon = None
        self.plan = None
        self.avg_opt_time = 0
        self.avg_iters = 0
        # self.plan_steps = 960 + 23

    def update_forecast(self, forecast: agent.WindField): 
        self.ble_forecast = forecast
        self.forecast = forecast.to_jax_wind_field()

    def update_atmosphere(self, atmosphere: agent.standard_atmosphere.Atmosphere): 
        self.ble_atmosphere = atmosphere
        self.atmosphere = atmosphere.to_jax_atmosphere() 


class MPC4FollowerAgent(agent.Agent):
        
    def __init__(self, num_actions: int, observation_shape): # Sequence[int]
        super(MPC4FollowerAgent, self).__init__(num_actions, observation_shape)
        self.i = 0

        datapath = "diagnostics/MPC4Agent-1740952765564.json"
        agent_name = 'mpc4_agent'
        diagnostics = json.load(open(datapath, 'r'))

        self.plan = diagnostics["0"]['rollout'][agent_name]['plan']

    def begin_episode(self, observation: np.ndarray) -> int:
        action = self.plan[self.i]
        self.i += 1
        return action

    def step(self, reward: float, observation: np.ndarray) -> int:
        return self.begin_episode(observation)

    def end_episode(self, reward: float, terminal: bool = True) -> None:
        self.i = 0
