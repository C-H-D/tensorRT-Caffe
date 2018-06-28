OUTNAME_RELEASE = giexec
OUTNAME_DEBUG   = giexec_debug
MAKEFILE ?= Makefile.$(OUTNAME_RELEASE)
include $(MAKEFILE)
