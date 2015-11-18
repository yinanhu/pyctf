import re
from numpy import array, hypot

def get_coord(f):
    l = f.next()
    return float(l.split()[2])

def coord(f):
    x = get_coord(f)
    y = get_coord(f)
    z = get_coord(f)
    return array((x, y, z))

def getHC(filename, frame):
    """n, l, r = getHC(filename, frame)
    Return the nasion, left, and right fiducial points as three arrays.
    frame may be either 'dewar' or 'head'.
    """

    if frame != 'head' and frame != 'dewar':
        raise ValueError("bad frame value")

    nasion = re.compile('measured nasion .* %s' % frame)
    left = re.compile('measured left .* %s' % frame)
    right = re.compile('measured right .* %s' % frame)

    n, l, r = None, None, None

    x = open(filename).xreadlines()
    for s in x:
        if nasion.match(s):
            n = coord(x)
        elif left.match(s):
            l = coord(x)
        elif right.match(s):
            r = coord(x)

    return n, l, r

if __name__ == '__main__':
    import sys

    def length(d):
        return hypot.reduce(d)

    n, l, r = getHC(sys.argv[1], 'dewar')

    print('nasion: %.3f %.3f %.3f' % tuple(n))
    print('left ear: %.3f %.3f %.3f' % tuple(l))
    print('right ear: %.3f %.3f %.3f' % tuple(r))
    print('left - right: %.3f cm' % length(l - r))
    print('nasion - left: %.3f cm' % length(n - l))
    print('nasion - right: %.3f cm' % length(n - r))
