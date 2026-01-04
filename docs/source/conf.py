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
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('../../'))
# -- Project information -----------------------------------------------------
about = {}
with open('../../googlehydrology/__about__.py', 'r') as fp:
    exec(fp.read(), about)

project = 'GoogleHydrology'
copyright = f'{datetime.datetime.now().year}, Google Research'
author = 'Grey Nearing, adapted from work by Frederik Kratzert'

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
    'nbsphinx'
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


def copy_notebooks(app):
    """Copies notebooks from the tutorial directory to the source directory."""
    root = Path(__file__).parent.parent.parent
    examples_dir = root / 'tutorial'
    tutorial_dir = root / 'docs' / 'source' / 'tutorial'
    
    # Mapping: Source relative to tutorial/ -> Destination relative to tutorial/
    notebooks = {
        'googlehydrology-tutorial.ipynb': 'googlehydrology-tutorial.ipynb',
    }

    if not tutorial_dir.exists():
        tutorial_dir.mkdir(parents=True)

    for src, dst in notebooks.items():
        src_path = examples_dir / src
        dst_path = tutorial_dir / dst
        if src_path.exists():
            print(f"Copying {src_path} to {dst_path}")
            shutil.copyfile(src_path, dst_path)
        else:
            print(f"Warning: Notebook {src_path} not found.")

def setup(app):
    app.connect('builder-inited', copy_notebooks)