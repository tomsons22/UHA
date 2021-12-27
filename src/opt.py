import jax.numpy as np
import jax
from jax.flatten_util import ravel_pytree
from tqdm import tqdm
import sys
import functools

def adam(step_size, b1 = 0.9, b2 = 0.999, eps = 1e-8):
	# Basically JAX's thing with added projection for some parameters.
	# Assumes ravel_pytree will always work the same way, so no need to update the
	# unflatten function (which may be problematic for jitting stuff)
	def init(x0):
		m0 = np.zeros_like(x0)
		v0 = np.zeros_like(x0)
		return x0, m0, v0
	def update(i, g, state, unflatten, trainable):
		def project(x, unflatten, trainable):
			x_train, x_notrain = unflatten(x)
			# if 'eps' in trainable:
				# x_train['eps'] = jax.nn.relu(x_train['eps'])
				# x_train['eps'] = np.clip(x_train['eps'], 0, 0.1)
			if 'eta' in trainable:
				x_train['eta'] = np.clip(x_train['eta'], 0, 0.99)
			if 'mgridref_y' in trainable:
				x_train['mgridref_y'] = jax.nn.relu(x_train['mgridref_y'] - 0.001) + 0.001
			return ravel_pytree((x_train, x_notrain))[0]
		x, m, v = state
		m = (1 - b1) * g + b1 * m # First moment estimate
		v = (1 - b2) * np.square(g) + b2 * v # Second moment estimate
		mhat = m / (1 - np.asarray(b1, m.dtype) ** (i + 1)) # Bias correction
		vhat = v / (1 - np.asarray(b2, m.dtype) ** (i + 1))
		x = x - step_size * mhat / (np.sqrt(vhat) + eps)
		x = project(x, unflatten, trainable)
		return x, m, v
	def get_params(state):
		x, _, _ = state
		return x
	return init, update, get_params

@functools.partial(jax.jit, static_argnums = (1, 2))
def collect_eps(params_flat, unflatten, trainable):
	if 'eps' in trainable:
		return unflatten(params_flat)[0]['eps']
	return 0.

def run(info, params_flat, unflatten, params_fixed, log_prob_model, grad_and_loss, trainable, rng_key_gen, extra=True):
	# try:
	opt_init, update, get_params = adam(info.lr)
	update = jax.jit(update, static_argnums = (3, 4))
	opt_state = opt_init(params_flat)
	losses = []
	tracker = {'eps': []}
	for i in tqdm(range(info.iters)):
		rng_key, rng_key_gen = jax.random.split(rng_key_gen)
		seeds = jax.random.randint(rng_key, (info.N,), 1, 1e6)
		params_flat = get_params(opt_state)
		tracker['eps'].append(collect_eps(params_flat, unflatten, trainable))
		grad, (loss, _) = grad_and_loss(seeds, params_flat, unflatten, params_fixed, log_prob_model)
		losses.append(loss.item())
		if np.isnan(loss):
			print('Diverged')
			return [], True, None, tracker
		opt_state = update(i, grad, opt_state, unflatten, trainable)
	# Get more estimates at end if extra
	if extra:
		loss_end = []
		for i in range(500):
			rng_key, _ = jax.random.split(rng_key)
			seeds = jax.random.randint(rng_key, (int(info.N),), 1, 1e6)
			_, (loss, _) = grad_and_loss(seeds, params_flat, unflatten, params_fixed, log_prob_model)
			loss_end.append(loss)
		losses = losses + loss_end
	return losses, False, params_flat, tracker
	# except Exception as e:
	# 	print('Sth failed!', e)
	# 	print('Sth failed!', file = sys.stderr)
	# 	print(e, file = sys.stderr)
	# 	return [], True, None


