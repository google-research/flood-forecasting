# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import datetime
import os
import sys

sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('../../'))
# -- Project information -----------------------------------------------------
about = {}
with open('../../googlehydrology/__about__.py', 'r') as fp:
    exec(fp.read(), about)

project = 'GoogleHydrology'
copyright = f'{datetime.datetime.now().year}, Frederik Kratzert'
author = 'Frederik Kratzert'

# The full version, including alpha/beta/rc tags
release = about['__version__']

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',  # autodocument
    'sphinx.ext.napoleon',  # google and numpy doc string support
    'sphinx.ext.mathjax',  # latex rendering of equations using MathJax
    'nbsphinx',  # for direct embedding of jupyter notebooks into sphinx docs
    'nbsphinx_link',  # to be able to include notebooks from outside of the docs folder
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['**.ipynb_checkpoints', '_build']

suppress_warnings = [
    # Ignore new warning in Sphinx 7.3.0 while pickling environment:
    #   WARNING: cannot cache unpickable configuration value: 'sphinx_gallery_conf'
    # See also: https://github.com/sphinx-doc/sphinx/issues/12300
    'config.cache',
]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# -- Napoleon autodoc options -------------------------------------------------
napoleon_numpy_docstring = True

# -- Other settings -----------------------------------------------------------

# Path to logo image file
html_logo = '_static/img/neural-hyd-logo.png'

html_theme_options = {'style_nav_header_background': '#175762'}

# Allows to build the docs with a minimal environment without warnings about missing packages
autodoc_mock_imports = [
    'matplotlib',
    'pandas',
    'ruamel',
    'scipy',
    'tqdm',
    'xarray',
]
