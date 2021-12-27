import jax.numpy as np
import jax
import variationaldist as vd
import momdist as md
from jax.flatten_util import ravel_pytree
import functools


# For now, betas equally spaced between 0 and 1.

def initialize(dim, vdparams=None, vdmode=1, k=10, trainable=['vd']):
	if vdparams is None:
		params = {'vd': vd.initialize(dim, vdmode)} # Has all trainable parameters
	else:
		params = {'vd': vdparams} # Has all trainable parameters
	
	# Other fixed parameters - these are always fixed
	params_fixed = (dim, vdmode, k)
	params_flat, unflatten = ravel_pytree((params, {}))
	return params_flat, unflatten, params_fixed

def compute_log_weight(seed, params_flat, unflatten, params_fixed, log_prob):
	params, _ = unflatten(params_flat)
	dim, vdmode, k = params_fixed
	# Sample
	rng_key = jax.random.PRNGKey(seed)
	z = vd.sample_rep(rng_key, vdmode, params['vd'])
	w = -vd.log_prob(vdmode, jax.lax.stop_gradient(params['vd']), z) # For drep
	# w = -vd.log_prob(vdmode, params['vd'], z) # For rep
	w = w + log_prob(z)
	return w

def compute_bound(seed, params_flat, unflatten, params_fixed, log_prob):
	# This function returns two things:
	# 1- The loss to compute the drep estimator
	# 2- The loss to plot
	rng_key = jax.random.PRNGKey(seed)
	dim, vdmode, k = params_fixed
	seeds = jax.random.randint(rng_key, (k,), 1, 1e6)
	log_ws = jax.vmap(compute_log_weight, in_axes = (0, None, None, None, None))(seeds, params_flat, unflatten, params_fixed, log_prob)
	max_log_w = np.max(log_ws)
	log_ws_shifted = log_ws - max_log_w
	# Drep gradient loss
	ws_normalized = np.exp(log_ws_shifted) / np.sum(np.exp(log_ws_shifted))
	loss_grad = np.square(jax.lax.stop_gradient(ws_normalized)) * log_ws # For drep
	# loss_grad = jax.lax.stop_gradient(ws_normalized) * log_ws # For rep
	# Loss loss (tracking IWELBO)
	loss_loss = np.log(1. / k)
	loss_loss = loss_loss + max_log_w
	loss_loss = loss_loss + np.log(np.sum(np.exp(log_ws_shifted)))
	return -1. * loss_grad, -1. * loss_loss

compute_bound_vec = jax.vmap(compute_bound, in_axes = (0, None, None, None, None))

@functools.partial(jax.jit, static_argnums = (2, 3, 4))
def compute_bound(seeds, params_flat, unflatten, params_fixed, log_prob):
	bounds_grad, bounds_loss = compute_bound_vec(seeds, params_flat, unflatten, params_fixed, log_prob)
	return bounds_grad.mean(), (bounds_loss.mean(), None)






