import jax.numpy as np
import numpyro.distributions as npdist
import jax


def encode_params(mean, M):
	return {'mean': mean, 'M': M}

def decode_params(params):
	mean, M = params['mean'], params['M']
	return mean, np.tril(M)

def initialize(dim):
	mean = np.zeros(dim)
	M = np.eye(dim) * 0.5
	return encode_params(mean, M)

def build(params):
	mean, M = decode_params(params)
	return npdist.MultivariateNormal(loc = mean, scale_tril = M)

def log_prob(z, params):
	dist = build(params)
	return dist.log_prob(z)

def log_prob_frozen(z, params):
	dist = build(jax.lax.stop_gradient(params))
	return dist.log_prob(z)

def entropy(params):
	mean, M = decode_params(params)
	dim = mean.shape[0]
	return dim * (1. + np.log(2. * np.pi)) / 2. + np.sum(np.log(np.abs(np.diag(M))))

def reparameterize(params, eps):
	mean, M = decode_params(params)
	return np.dot(M, eps) + mean

# def sample_eps(rng_key, nsamples, dim):
# 	return jax.random.normal(rng_key, shape = (nsamples, dim))

# def sample_rep(rng_key, params, nsamples):
# 	mean, _ = decode_params(params)
# 	dim = mean.shape[0]
# 	eps = sample_eps(rng_key, nsamples, dim)
# 	return reparameterize(params, eps)

def sample_eps(rng_key, dim):
	return jax.random.normal(rng_key, shape = (dim,))

def sample_rep(rng_key, params):
	mean, _ = decode_params(params)
	dim = mean.shape[0]
	eps = sample_eps(rng_key, dim)
	return reparameterize(params, eps)



# class DLRGaussian():
# 	def __init__(self, dim, rank):
# 		self.mode = "dlr"
# 		self.dim = dim
# 		self.rank = rank

# 	def encode_params(self, mean, logdiag, F):
# 		return {'mean': mean, 'logdiag': logdiag, 'F': F}

# 	def decode_params(self, params):
# 		mean, logdiag, F = params['mean'], params['logdiag'], params['F']
# 		return mean, logdiag, F

# 	def init_params(self):
# 		mean = np.zeros(self.dim)
# 		logdiag = np.ones(self.dim) * -1.
# 		F = np.zeros((self.dim, self.rank))
# 		return self.encode_params(mean, logdiag, F)

# 	def build(self, params):
# 		mean, logdiag, F = self.decode_params(params)
# 		return npdist.LowRankMultivariateNormal(loc = mean, cov_diag = np.exp(2. * logdiag), cov_factor = F)

# 	# This is for "sticking the landing"
# 	def log_prob(self, z, params):
# 		dist = self.build(jax.lax.stop_gradient(params))
# 		return dist.log_prob(z)

# 	def entropy(self, params):
# 		dist = self.build(params)
# 		return dist.entropy()

# 	def reparameterize(self, params, eps):
# 		mean, logdiag, F = self.decode_params(params)
# 		z = np.exp(logdiag) * eps['d'] + np.dot(F, eps['f']) + mean
# 		return z

# 	def sample_eps(self, rng_key, nsamples):
# 		eps = {}
# 		eps['d'] = jax.random.normal(rng_key, shape = (nsamples, self.dim))
# 		eps['f'] = jax.random.normal(rng_key, shape = (nsamples, self.rank))
# 		return eps



