from jax import jit, grad, vmap
import jax.numpy as np
import jax
import numpyro
from jax.flatten_util import ravel_pytree
import numpyro.distributions as npdists
import models.logistic_regression as model_lr
import models.seeds as model_seeds
import inference_gym.using_jax as gym


models_gym = ['lorenz', 'brownian', 'banana']

def load_model(model = 'log_sonar'):
	if model in models_gym:
		return load_model_gym(model)
	return load_model_other(model)


def load_model_gym(model = 'banana'):
	def log_prob_model(z):
		x = target.default_event_space_bijector(z)
		return (target.unnormalized_log_prob(x) + target.default_event_space_bijector.forward_log_det_jacobian(z, event_ndims = 1))
	if model == 'lorenz':
		target = gym.targets.ConvectionLorenzBridge()
	if model == 'brownian':
		target = gym.targets.BrownianMotionUnknownScalesMissingMiddleObservations()
	if model == 'banana':
		target = gym.targets.Banana()
	target = gym.targets.VectorModel(target, flatten_sample_transformations = True)
	dim = target.event_shape[0]
	return log_prob_model, dim


def load_model_other(model = 'log_sonar'):
	if model == 'log_sonar':
		model, model_args = model_lr.load_model('sonar')
	if model == 'log_ionosphere':
		model, model_args = model_lr.load_model('ionosphere')
	if model == 'log_australian':
		model, model_args = model_lr.load_model('australian')
	if model == 'log_a1a':
		model, model_args = model_lr.load_model('a1a')
	if model == 'log_madelon':
		model, model_args = model_lr.load_model('madelon')
	if model == 'nn1_wine':
		model, model_args = model_nn1.load_model('redwine', 20)
	if model == 'seeds':
		model, model_args = model_seeds.load_model()
	if model == 'eight_schools':
		model, model_args = model_es.load_model()
	if model == 'orange_tree':
		model, model_args = model_ot.load_model()
	if model == 'funnel_mine':
		model, model_args = model_f.load_model()
	if model == 'bat':
		model, model_args = model_bat.load_model()
	
	rng_key = jax.random.PRNGKey(1)
	model_param_info, potential_fn, constrain_fn, _ = numpyro.infer.util.initialize_model(rng_key, model, model_args = model_args)
	params_flat, unflattener = ravel_pytree(model_param_info[0])
	log_prob_model = lambda z: -1. * potential_fn(unflattener(z))
	dim = params_flat.shape[0]
	unflatten_and_constrain = lambda z: constrain_fn(unflattener(z))
	return log_prob_model, dim





