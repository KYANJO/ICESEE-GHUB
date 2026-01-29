all:

install:
	. /etc/environ.sh; use -e -r anaconda-7; cd ..; export PYTHONPATH=$$PWD; jupyter-book build --all icesee_jupyter_book

clean:

distclean: clean
	cd ..; jupyter-book clean icesee_jupyter_book

.PHONY: all install clean distclean
