
# For python linking (require f2py part of numpy)
ifeq ($(origin MENUM),undefined)
  MENUM=2
endif
P1_mup_epvevmx/matrix$(MENUM)py.so:
	cd P1_mup_epvevmx;make matrix$(MENUM)py.so
all:  P1_mup_epvevmx/matrix$(MENUM)py.so 
	echo 'done'