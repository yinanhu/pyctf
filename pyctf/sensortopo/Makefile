include ../../config/Makefile.config

all: _fieldspline.so

_fieldspline.so: _fieldspline.c
	env PYMODNAME=_fieldspline $(MAKE) -f $(CONFDIR)/Makefile.pymod

DEST = $(LIBDIR)/sensortopo

install: all
	@mkdir -p $(DEST)
	cp _fieldspline.so *.py CTF275.txt $(DEST)

clean: clean-x
