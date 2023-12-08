# Get the root directory of the project (the directory where this Makefile lives)
ROOT_DIR:=$(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

.PHONY: all mypy

all: mypy

mypy:
	mypy $(ROOT_DIR)/src/ --check-untyped-defs