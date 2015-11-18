def paramDict(filename):
    """A simple parameter file with lines of the form
        key [value] ..., return a dict."""

    d = {}
    ll = open(filename).readlines()
    for l in ll:
        # Ignore all past a '#'
        l = l.partition('#')[0].split()
        if len(l) == 0:
            continue

        # Get the name and any values.
        name = l.pop(0)
        ll = []
        for x in l:
            try:
                x = float(x)
            except ValueError:
                pass
            ll.append(x)
        d[name] = ll

    # Now convert any singleton lists into just the first element.

    for k in d:
        if len(d[k]) == 1:
            d[k] = d[k][0]

    return d
