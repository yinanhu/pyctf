include config/Makefile.config

targets = pyctf parsemarks fiddist avghc StockwellDs

all:
        for x in $(targets) ; do \
                $(MAKE) -C $$x $@ || exit ;\
        done

install: all
        for x in $(targets) ; do \
                $(MAKE) -C $$x $@ || exit ;\
        done

clean: clean-x
