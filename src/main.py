import numpyro
import numpyro.distributions as dist
import jax.numpy as np
import jax
from jax.flatten_util import ravel_pytree
import argparse
import boundingmachine as bm
import iwboundingmachine as iwbm
import opt
from model_handler import load_model

		


args_parser = argparse.ArgumentParser(description='Process arguments')
args_parser.add_argument('-boundmode', type=str, default='UHA', help='UHA or IW.')
args_parser.add_argument('-model', type=str, default='log_sonar', help='Model to use.')
args_parser.add_argument('-N', type=int, default=5, help='Number of samples to estimate gradient at each step.')
args_parser.add_argument('-nbridges', type=int, default=10, help='Number of bridging densities.')
args_parser.add_argument('-lfsteps', type=int, default=1, help='Number of leapfrog steps.')
args_parser.add_argument('-iters', type=int, default=5000, help='Number of iterations.')
args_parser.add_argument('-lr', type=float, default=0.01, help='Learning rate.')
args_parser.add_argument('-seed', type=int, default=1, help='Random seed to use.')
args_parser.add_argument('-vdmode', type=int, default=1, help='Variational distribution mode.')
info = args_parser.parse_args()

log_prob_model, dim = load_model(info.model)

rng_key_gen = jax.random.PRNGKey(info.seed)

# Train initial variational distribution to maximize the ELBO
trainable=('vd',)
params_flat, unflatten, params_fixed = bm.initialize(dim=dim, vdmode=info.vdmode, nbridges=0, trainable=trainable)
grad_and_loss = jax.jit(jax.grad(bm.compute_bound, 1, has_aux = True), static_argnums = (2, 3, 4))
losses, diverged, params_flat, tracker = opt.run(info, params_flat, unflatten, params_fixed, log_prob_model, grad_and_loss, trainable, rng_key_gen)
vdparams_init = unflatten(params_flat)[0]['vd']

elbo = -np.mean(np.array(losses[-500:]))
print('Done training initial parameters, got ELBO %.2f.' % elbo)

if info.boundmode == 'UHA':
	trainable = ('vd', 'eps', 'eps', 'mgridref_y')
	params_flat, unflatten, params_fixed = bm.initialize(dim=dim, vdmode=info.vdmode, nbridges=info.nbridges, 
		lfsteps=info.lfsteps, vdparams=vdparams_init, trainable=trainable)
	grad_and_loss = jax.jit(jax.grad(bm.compute_bound, 1, has_aux = True), static_argnums = (2, 3, 4))
	losses, diverged, params_flat, tracker = opt.run(info, params_flat, unflatten, params_fixed, log_prob_model, grad_and_loss, trainable, rng_key_gen)

elif info.boundmode == 'IW':
	params_flat, unflatten, params_fixed = iwbm.initialize(dim=dim, vdmode=info.vdmode, vdparams=vdparams_init, k=info.nbridges)
	grad_and_loss = jax.jit(jax.grad(iwbm.compute_bound, 1, has_aux = True), static_argnums = (2, 3, 4))
	losses, diverged, params_flat, tracker = opt.run(info, params_flat, unflatten, params_fixed, log_prob_model, grad_and_loss, trainable, rng_key_gen)

else:
	raise NotImplementedError('Mode %s not implemented.' % info.boundmode)

final_elbo = -np.mean(np.array(losses[-500:]))
print('Done training, got ELBO %.2f.' % final_elbo)

# import matplotlib.pyplot as plt
# plt.figure()
# plt.plot(losses)
# plt.show()
