from numpy import *
from matplotlib.colors import Colormap

class meg_colormap(Colormap):
	"""A colormap for MEG data."""

	__super_init = Colormap.__init__

	def __init__(self, name, N = 256):
		self.__super_init(name, N)
		self.monochrome = False

	def __call__(self, x, alpha = 1.):
		alpha = max(alpha, 0.0)
		alpha = min(alpha, 1.0)

		x = asarray(x)
		if x.shape == ():
			x.shape = 1,
		rgba = zeros(x.shape + (4,), 'f')

		rgba[..., 0] = self.red(x)
		rgba[..., 1] = self.green(x)
		rgba[..., 2] = self.blue(x)
		rgba[..., 3] = alpha

		return rgba

	def red(self, x):
		r0, r1, r2, gam = .3, .5, .9, .7
		x = 1. - x
		y = zeros(x.shape, x.dtype)

		y[x < r0] = 1.

		w = (r0 <= x) & (x < r1)
		if any(w):
			m = 1. / (r0 - r1)
			b = -m * r1
			v = m * x[w] + b
			y[w] = v ** gam

		w = (r1 <= x) & (x < r2)
		y[w] = 0.

		x[1. < x] = 1.
		w = r2 <= x
		if any(w):
			m = .5 / (1. - r2)
			b = -m * r2
			v = m * x[w] + b
			y[w] = v;

		return y

	def green(self, x):
		r0, r1, r2, r3 = .1, .3, .7, .9
		x = 1. - x
		y = zeros(x.shape, x.dtype)

		y[x < r0] = 1.

		w = (r0 <= x) & (x < r1)
		if any(w):
			m = 1. / (r0 - r1)
			b = -m * r1
			v = m * x[w] + b
			y[w] = v

		w = (r1 <= x) & (x < r2)
		y[w] = 0.

		w = (r2 <= x) & (x < r3)
		if any(w):
			m = 1. / (r3 - r2)
			b = -m * r2
			v = m * x[w] + b
			y[w] = v;

		y[r3 <= x] = 1.

		return y

	def blue(self, x):
		return self.red(1. - x)

meg_cmap = meg_colormap('meg_cmap')
