.SHELLFLAGS := -euc

CONDA_BASE ?= $(shell conda info --base)
CONDA_BUILD ?= conda-build
CONDA_ENV ?= dev
$(VERBOSE).SILENT:

CONDA_BASE := ${CONDA_BASE}
CONDA_ENV := ${CONDA_ENV}
CONDA_ACTIVATE := . $(CONDA_BASE)/etc/profile.d/conda.sh ; conda activate ; conda activate

# recursive wildcard function
# from https://stackoverflow.com/questions/2483182/recursive-wildcards-in-gnu-make/18258352#18258352
rwildcard=$(foreach d,$(wildcard $(1:=/*)),$(call rwildcard,$d,$2) $(filter $(subst *,%,$2),$d))

.PHONY: help
help:
	@echo "This project assumes that a conda installation is present."
	@echo "The following make targets are available:"
	@echo "  build    build sdist and wheel"
	@echo "  cbuild   build conda package"
	@echo "  clean    clean build files, doc files, and test files"
	@echo "  codecov  upload test coverage to codecov.io"
	@echo "  conda    deploy to conda channel"
	@echo "  deploy   deploy built package to PyPI repo, pull and build conda package, and deploy to conda channel"
	@echo "  docs     create docs for all relevant modules"
	@echo "  install  install package into conda environment"
	@echo "  lint     lint all project python code"
	@echo "  major    update project major version number"
	@echo "  minor    update project minor version number"
	@echo "  patch    update project patch version number"
	@echo "  pypi     upload build package to PyPI"
	@echo "  setup    setup conda environment for development"
	@echo "  tag      create git tag with current project version and push"
	@echo "  test     run all tests"

CONDA_PYTHON ?= ${CONDA_BASE}/bin/python
.PHONY: clean
clean:
	$(CONDA_PYTHON) setup.py clean --env=$(CONDA_ENV)

$(CONDA_BASE)/envs/$(CONDA_ENV): setup.cfg
	@$(CONDA_PYTHON) setup.py env --env-name=$(CONDA_ENV)

.PHONY: setup
setup: | $(CONDA_BASE)/envs/$(CONDA_ENV)

dist: | setup.py
ifeq ("$(wildcard $(CONDA_BASE)/envs/$(CONDA_ENV))","")
	@$(error run "make setup" first)
endif
	@$(MAKE) lint
	@$(CONDA_ACTIVATE) $(CONDA_ENV); python setup.py sdist bdist_wheel

.PHONY: build
build: | dist

.PHONY: pypi
pypi: dist | setup.py
ifeq ("$(wildcard $(CONDA_BASE)/envs/$(CONDA_ENV))","")
	@$(error run "make setup" first)
endif
	@$(CONDA_ACTIVATE) $(CONDA_ENV); python setup.py deploy_pypi --pypi-token=$(PYPI_TOKEN)

CONDA_RECIPE ?= conda
$(CONDA_BUILD): setup.cfg $(wildcard $(CONDA_RECIPE)/*)
ifeq ("$(wildcard $(CONDA_BASE)/envs/$(CONDA_ENV))","")
	@$(error run "make setup" first)
endif
	@$(MAKE) pypi
	@$(CONDA_ACTIVATE) $(CONDA_ENV); python setup.py conda_build --build-dir=$(CONDA_BUILD) --recipe=$(CONDA_RECIPE)

.PHONY: cbuild
cbuild: | $(CONDA_BUILD)

.PHONY: conda
conda: | $(CONDA_BUILD)
ifeq ("$(wildcard $(CONDA_BASE)/envs/$(CONDA_ENV))","")
	@$(error run "make setup" first)
endif
	@$(CONDA_ACTIVATE) $(CONDA_ENV); python setup.py deploy_conda --build-dir=$(CONDA_BUILD) --username=$(ANACONDA_USERNAME) --password=$(ANACONDA_PASSWORD)

.PHONY: deploy
deploy:
	@$(MAKE) codecov
	@$(MAKE) pypi
	@$(MAKE) conda

SPHINX_TEMPLATE_DIR ?= sphinx/templates
docs: $(call rwildcard,.,*.py)
ifeq ("$(wildcard $(CONDA_BASE)/envs/$(CONDA_ENV))","")
	@$(error run "make setup" first)
endif
	@$(CONDA_ACTIVATE) $(CONDA_ENV); python setup.py docs --template-dir=$(SPHINX_TEMPLATE_DIR)

EDIT ?= 0
.PHONY: install
install:
ifneq ($(EDIT),0)
	@$(MAKE) install_editable
else
	@$(MAKE) install_conda
endif

.PHONY: install_conda
install_conda: $(CONDA_BUILD)
ifeq ("$(wildcard $(CONDA_BASE)/envs/$(CONDA_ENV))","")
	@$(error run "make setup" first)
endif
	@$(CONDA_ACTIVATE) $(CONDA_ENV); python setup.py install_local --build-dir=$(CONDA_BUILD) 

.PHONY: install_editable
install_editable:
ifeq ("$(wildcard $(CONDA_BASE)/envs/$(CONDA_ENV))","")
	@$(error run "make setup" first)
endif
	@$(CONDA_ACTIVATE) $(CONDA_ENV); python setup.py install_local --editable

.PHONY: lint
lint: $(call rwildcard,.,*.py) | setup.py src tests
ifeq ("$(wildcard $(CONDA_BASE)/envs/$(CONDA_ENV))","")
	@$(error run "make setup" first)
endif
	@$(CONDA_ACTIVATE) $(CONDA_ENV); python setup.py lint

.PHONY: major
major: | setup.py
ifeq ("$(wildcard $(CONDA_BASE)/envs/$(CONDA_ENV))","")
	@$(error run "make setup" first)
endif
	@$(CONDA_ACTIVATE) $(CONDA_ENV); python setup.py version --type=major

.PHONY: minor
minor: | setup.py
ifeq ("$(wildcard $(CONDA_BASE)/envs/$(CONDA_ENV))","")
	@$(error run "make setup" first)
endif
	@$(CONDA_ACTIVATE) $(CONDA_ENV); python setup.py version --type=minor

.PHONY: patch
patch: | setup.py
ifeq ("$(wildcard $(CONDA_BASE)/envs/$(CONDA_ENV))","")
	@$(error run "make setup" first)
endif
	@$(CONDA_ACTIVATE) $(CONDA_ENV); python setup.py version --type=patch

GIT_REMOTE ?= origin
.PHONY: tag
tag: setup.py
ifeq ("$(wildcard $(CONDA_BASE)/envs/$(CONDA_ENV))","")
	@$(error run "make setup" first)
endif
	@$(CONDA_ACTIVATE) $(CONDA_ENV); python setup.py tag --remote=$(GIT_REMOTE)

coverage.xml: | setup.py
ifeq ("$(wildcard $(CONDA_BASE)/envs/$(CONDA_ENV))","")
	@$(error run "make setup" first)
endif
	@$(MAKE) install
	@$(CONDA_ACTIVATE) $(CONDA_ENV); python setup.py test

.PHONY: codecov
codecov: coverage.xml | setup.py
ifeq ("$(wildcard $(CONDA_BASE)/envs/$(CONDA_ENV))","")
	@$(error run "make setup" first)
endif
	@$(CONDA_ACTIVATE) $(CONDA_ENV); python setup.py codecov --codecov-token=$(CODECOV_TOKEN)

.PHONY: test
test: | coverage.xml
