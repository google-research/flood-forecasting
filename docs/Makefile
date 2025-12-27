# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SPHINXAPIDOC  ?= sphinx-apidoc
SOURCEDIR     = source
BUILDDIR      = build
PACKAGEDIR    = ../googlehydrology

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile apidoc clean-api

# Target to auto-generate API docs
# -f: Force overwriting of any existing generated files.
# -o: Directory to place the output files.
# -e: Put documentation for each module on its own page.
apidoc:
	@echo "Generating API documentation..."
	@$(SPHINXAPIDOC) -f -e -o "$(SOURCEDIR)/api" "$(PACKAGEDIR)"

# Target to clean API docs
clean-api:
	@echo "Cleaning API documentation..."
	@rm -rf "$(SOURCEDIR)/api"

# Explicitly define html target to run apidoc first
html: apidoc
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Clean build directory and API docs
clean: clean-api
	@$(SPHINXBUILD) -M clean "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)