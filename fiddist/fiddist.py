#! /usr/bin/env python

import sys, os.path, getopt, re
from numpy import array, hypot

from pyctf.thd_atr import afni_header_read
from pyctf.fid import *
from pyctf.util import *

usage("""-m|c|b name
Return the interfiducial distances for name.
If -b is used, name is an AFNI brik.
If -m or -c is used, name is an MEG dataset.
For -m, the .hdm file is examined (the output of localSpheres).
For -c, the .hc file is examined (the head coil locations).""")

HDM = 1
HC = 2
BRIK = 3
type = None

optlist, args = parseargs("mcb")

for opt, arg in optlist:
    if opt == '-m':
        type = HDM
    elif opt == '-c':
        type = HC
    elif opt == '-b':
        type = BRIK

if type == None or len(args) != 1:
    printusage()
    sys.exit(1)

filename = args[0]

# If the argument is a .ds directory, get the corresponding .hc or .hdm file.
s = filename.split('.')
ext = s[-1]
if ext == 'ds' or ext == 'ds/':
    base = s[0].split('/')[-1]
    if type == HDM:
        msg("using default.hdm\n")
        filename += '/' + 'default.hdm'
    elif type == HC:
        msg("using %s.hc\n" % base)
        filename += '/' + base + '.hc'

# If it's a BRIK, read the header, otherwise just open the file.

if type == BRIK:
    h = afni_header_read(filename)
    if not h.has_key('TAGSET_NUM'):
        printerror("%s has no tags!" % filename)
        sys.exit(1)
else:
    x = open(filename)

# HC
def get_coord(l):
    return float(l.split()[2])

def coord(l, i):
    x = get_coord(l[i])
    y = get_coord(l[i+1])
    z = get_coord(l[i+2])
    return array((x, y, z))

# HDM
def coord2(s):
    return array(list(map(int, s.split()[-3:]))) * .1

# BRIK
def coord3(s):
    x = array(list(map(fuzz, tl)[0:3])) * .1
    # convert from RAI to PRI
    return array((-x[1], x[0], x[2]))

def fuzz(t):
    if abs(t) < 1.e-8:
        return 0.
    return t

if type == HC:
    nasion = re.compile('measured nasion .* head')
    left = re.compile('measured left .* head')
    right = re.compile('measured right .* head')

    ll = x.readlines()
    i = 0
    while i < len(ll):
        s = ll[i]
        i += 1
        if nasion.match(s):
            n = coord(ll, i)
            i += 3
        elif left.match(s):
            l = coord(ll, i)
            i += 3
        elif right.match(s):
            r = coord(ll, i)
            i += 3

elif type == HDM:
    nasion = re.compile('.*NASION:')
    left = re.compile('.*LEFT_EAR:')
    right = re.compile('.*RIGHT_EAR:')
    sres = re.compile('.*SAGITTAL:')
    cres = re.compile('.*CORONAL:')
    ares = re.compile('.*AXIAL:')
    sr = None

    for s in x:
        if nasion.match(s):
            n = coord2(s)
        if left.match(s):
            l = coord2(s)
        if right.match(s):
            r = coord2(s)
        if sres.match(s):
            sr = float(s.split()[-1])
        if cres.match(s):
            cr = float(s.split()[-1])
        if ares.match(s):
            ar = float(s.split()[-1])
    if sr == None:
        msg("no resolution found, assuming 1 mm/voxel\n")
        sr = cr = ar = 1.
    res = array((sr, cr, ar))
    n *= res
    l *= res
    r *= res
    msg("using slice coordinates: SCA\n")

elif type == BRIK:
    ntags, pertag = h['TAGSET_NUM']
    f = h['TAGSET_FLOATS']
    lab = h['TAGSET_LABELS']
    d = {}
    for i in range(ntags):
        tl = f[i * pertag : (i+1) * pertag]
        d[lab[i]] = coord3(tl)
    n = d[NASION]
    l = d[LEAR]
    r = d[REAR]

def length(d):
    return hypot.reduce(d)

print('nasion: %.3f %.3f %.3f' % tuple(n))
print('left ear: %.3f %.3f %.3f' % tuple(l))
print('right ear: %.3f %.3f %.3f' % tuple(r))
print('left - right: %.3f cm' % length(l - r))
print('nasion - left: %.3f cm' % length(n - l))
print('nasion - right: %.3f cm' % length(n - r))

if l[1] < 0:
    msg("Warning: left / right flip detected.\n")
