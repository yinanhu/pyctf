from numpy import arange, sin, sqrt, pi

# Riedel & Sidorenko sine tapers.

def sine_taper(k, N):
	"Compute the kth sine taper of length N"

	s = sqrt(2. / (N + 1))
	d = arange(N, dtype = 'd')
	return s * sin(pi * (k + 1) * (d + 1.) / (N + 1))
