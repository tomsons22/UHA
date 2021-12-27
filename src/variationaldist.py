import vardist.diag_gauss as diagvd
import vardist.full_gauss as fullvd

# How to deal with vdmode nicely?!

def initialize(dim, vdmode):
	"""
	Returns tuple (parameters, apply_fun)
	"""
	if vdmode is None or vdmode == 1:
		return diagvd.initialize(dim)
	elif vdmode == 2:
		return fullvd.initialize(dim)
	raise NotImplementedError('Variational distribution %s not available.' % str(vdmode))

def sample_rep(rng_key, vdmode, vdparams):
	if vdmode is None or vdmode == 1:
		return diagvd.sample_rep(rng_key, vdparams)
	elif vdmode == 2:
		return fullvd.sample_rep(rng_key, vdparams)
	raise NotImplementedError('Variational distribution %s not available.' % str(vdmode))

def log_prob(vdmode, vdparams, z):
	if vdmode is None or vdmode == 1:
		return diagvd.log_prob(z, vdparams)
	elif vdmode == 2:
		return fullvd.log_prob(z, vdparams)
	raise NotImplementedError('Variational distribution %s not available.' % str(vdmode))




