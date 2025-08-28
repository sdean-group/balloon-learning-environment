import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from balloon_learning_environment.env.balloon.jax_balloon import JaxBalloon
from balloon_learning_environment.env.wind_field import JaxWindField

# import TerminalCost base class
# from balloon_learning_environment.agents.mpc4_agent import TerminalCost

class ValueNetworkFeature:
    @property
    def num_input_dimensions(self):
        raise NotImplementedError('Implement num_input_dimensions')

    def compute(self, balloon: JaxBalloon, wind_forecast: JaxWindField):
        raise NotImplementedError('Implement computing')

class ValueNetwork(eqx.Module):
    residual: eqx.nn.MLP
    prior: eqx.nn.MLP

    @staticmethod
    def create(key, input_dim=5, hidden_dim=128):
        key_residual, key_prior = jax.random.split(key, 2)
        residual_network = eqx.nn.MLP(
            in_size=input_dim,
            out_size=1,
            depth=2,
            width_size=hidden_dim,
            key=key_residual
        )
        prior_network = eqx.nn.MLP(
            in_size=input_dim,
            out_size=1,
            depth=2,
            width_size=hidden_dim,
            key=key_prior
        )
        return ValueNetwork(residual=residual_network, prior=prior_network)

    # @eqx.debug.assert_max_traces(max_traces=1)
    def __call__(self, x):
        # print("recompiling")
        return (self.residual(x) + self.prior(x)).squeeze()

    def loss_and_grad(self, x, y):
        def _loss_fn(residual_network: eqx.nn.MLP, prior_network: eqx.nn.MLP, x, y):
            model = ValueNetwork(residual_network, prior_network)
            preds = model(x)
            return jnp.mean((preds - y) ** 2)

        loss_value, grads = eqx.filter_value_and_grad(_loss_fn)(self.residual, self.prior, x, y)
        return loss_value, grads

class ValueNetworkTrainer:
    def __init__(self, *, initial_model, learning_rate=1e-3, jit_model_loss_and_grad=True):
        self.model: ValueNetwork = initial_model
        self.optimizer: optax.GradientTransformation = optax.sgd(learning_rate=learning_rate)
        self.opt_state: optax.OptState = self.optimizer.init(self.model.residual)

        self.jit_model_loss_and_grad = jit_model_loss_and_grad

    def update(self, x, y):
        if self.jit_model_loss_and_grad:
            _, grads = eqx.filter_jit(self.model.loss_and_grad)(x, y)
        else:
            _, grads = self.model.loss_and_grad(x, y)

        updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
        model_residual = eqx.apply_updates(self.model.residual, updates)
        self.model = ValueNetwork(model_residual, self.model.prior)
        return self.model
    
class ValueNetworkTerminalCost(eqx.Module):
    value_network: ValueNetwork 
    value_network_feature: ValueNetworkFeature

    def __init__(self, value_network_feature: ValueNetworkFeature, value_network: ValueNetwork):
        self.value_network = value_network
        self.value_network_feature = value_network_feature

    def __call__(self, jax_balloon: JaxBalloon, wind_forecast: JaxWindField):
        feature = self.value_network_feature.compute(jax_balloon, wind_forecast)
        return self.value_network(feature)
    
class EnsembleValueNetworkTerminalCost(eqx.Module):
    value_networks: tuple[ValueNetwork]
    value_network_feature: ValueNetworkFeature
    kappa: float

    """ Uses an ensemble of value networks to calculate terminal cost """
    def __init__(self, vn_feature: ValueNetworkFeature, ensemble: tuple[ValueNetwork], kappa: float):
        
        self.value_networks = ensemble
        self.value_network_feature = vn_feature
        self.kappa = kappa

    def __call__(self, jax_balloon: JaxBalloon, wind_forecast: JaxWindField):
        print("recompiling")
        feature = self.value_network_feature.compute(jax_balloon, wind_forecast)
        # values = jax.lax.map(lambda vn: vn(feature), self.value_networks)
        # total = jnp.sum(jnp.exp(self.kappa * values))
        v0 = self.value_networks[0](feature)
        v1 = self.value_networks[1](feature)
        v2 = self.value_networks[2](feature)

        values = jnp.array([v0, v1, v2])
        total = jnp.sum(jnp.exp(self.kappa * values))
        return jnp.log(total)
