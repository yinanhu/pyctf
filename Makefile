include config/Makefile.config

targets = pyctf parsemarks fiddist chlDs avghc headPosDs StockwellDs

all:
	for x in $(targets) ; do \
		$(MAKE) -C $$x $@ || exit ;\
	done

install: all
	for x in $(targets) ; do \
		$(MAKE) -C $$x $@ || exit ;\
	done

clean: clean-x

# Moved to Makefile.config clean-x
#clean:
#	for x in $(targets) ; do \
#		$(MAKE) -C $$x $@ || exit ;\
#	done
#	rm -f *~ *.pyc *.o *.a *.so
#	rm -r __pycache__
