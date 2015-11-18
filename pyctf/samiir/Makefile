include ../../config/Makefile.config

all: samiir.so

samiir.so: samiir.c mkiir.o bdiir.o response.o FFTfilter.o Butterworth.o
	env PYMODNAME=samiir PYMODCFLAGS="" PYMODLIBS="mkiir.o bdiir.o response.o FFTfilter.o Butterworth.o -lfftw3 -lm" $(MAKE) -f $(CONFDIR)/Makefile.pymod

DEST = $(LIBDIR)/samiir

install: all
	@mkdir -p $(DEST)
	cp samiir.so *.py $(DEST)

clean: clean-x
