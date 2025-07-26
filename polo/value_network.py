import jax
import jax.numpy as jnp
import equinox as eqx
import optax

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

    def __call__(self, x):
        return self.residual(x) + self.prior(x)

def loss_and_grad(model: ValueNetwork, x, y):
    def _loss_fn(residual_network: eqx.nn.MLP, prior_network: eqx.nn.MLP, x, y):
        model = ValueNetwork(residual_network, prior_network)
        preds = model(x)
        return jnp.mean((preds - y) ** 2)

    loss_value, grads = eqx.filter_value_and_grad(_loss_fn)(model.residual, model.prior, x, y)
    return loss_value, grads

model = ValueNetwork.create(jax.random.key(seed=0))

# Test that gradients work
x = jnp.array([[1.0, 2.0, 3.0, 4.0, 5.0]]).T.squeeze()
y = jnp.array([[10.0]])
print(model(x), y)

loss_value, grad = loss_and_grad(model, x, y)
print("Loss:", loss_value)
model_residual = eqx.apply_updates(model.residual, jax.tree.map(lambda g: -1e-3 * g, grad))
model = ValueNetwork(model_residual, model.prior)

print("Loss:", loss_and_grad(model, x, y)[0])


optimizer = optax.sgd(learning_rate=1e-3)
opt_state = optimizer.init(model.residual)

for i in range(100):
    loss_value, grad = loss_and_grad(model, x, y)
    updates, opt_state = optimizer.update(grad, opt_state)
    model_residual = eqx.apply_updates(model.residual, updates)
    model = ValueNetwork(model_residual, model.prior)

    print("Loss:", loss_and_grad(model, x, y)[0])

# Check if prior stayed the same and residual was updated
