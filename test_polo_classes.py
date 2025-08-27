import equinox as eqx
import optax
import jax

class ValueNetwork(eqx.Module):
    """ A value network to be trained in the POLO framework. """
    residual: eqx.nn.MLP
    prior: eqx.nn.MLP
    optimizer: optax.GradientTransformation
    opt_state: optax.OptState

    def __init__(self, key, input_dim=5, hidden_dim=128, lr=1e-3):
        key_res, key_prior = jax.random.split(key)
        self.residual = eqx.nn.MLP(input_dim, 1, hidden_dim, depth=2, key=key_res)
        self.prior = eqx.nn.MLP(input_dim, 1, hidden_dim, depth=2, key=key_prior)

        self.optimizer = optax.adam(lr)
        self.opt_state = self.optimizer.init(eqx.filter(self.residual, eqx.is_array))

    def __call__(self, x):
        # Predict total value = prior + residual
        prior_val = jnp.squeeze(self.prior(x), axis=-1)
        resid_val = jnp.squeeze(self.residual(x), axis=-1)
        return prior_val + resid_val

    def update(self, x, y):
        def loss_fn(residual_params):
            model = eqx.tree_at(lambda m: m.residual, self, residual_params)
            return jnp.mean((model(x) - y) ** 2)

        grads = jax.grad(loss_fn)(eqx.filter(self.residual, eqx.is_array))
        updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
        self.residual = eqx.apply_updates(self.residual, updates)


