include ../config/Makefile.config

targets = samiir st sensortopo

all:
	for x in $(targets) ; do \
		$(MAKE) -C $$x $@ || exit ;\
	done

install: all
	for x in $(targets) ; do \
		$(MAKE) -C $$x $@ || exit ;\
	done
	cp *.py $(LIBDIR)

clean: clean-x
