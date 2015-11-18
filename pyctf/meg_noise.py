from numpy import zeros, array, arange
from numpy.fft import fft, ifft
from numpy.random import normal

def meg_noise(l, n = .5):
    """Return l samples of 1/f noise."""

    l2 = l / 2

    d = zeros((l,), 'f')
    y = 0.
    for i in range(l):
        x = normal()    # white
        y += x          # brown
        d[i] = y

    # detrend
    d = d - arange(l) * (d[-1] - d[0]) / l

    # Fractional derivative of d. Regular derivative (n=1) adds 2 to the
    # exponent of the spectrum. Fractional derivative does a multiple of that,
    # so n = .5 adds 1 to the exponent. Thus for brown (-2) you get pink (-1).

    w = array(range(l2) + range(-l2, 0))
    jwn = pow((1j) * w, n)
    D = fft(d)
    D = D * jwn
    dd = ifft(D).real / l

    dd -= dd.mean()
    dd /= dd.std()

    return dd
