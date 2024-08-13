# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sys,os
sys.path.insert(0,os.path.abspath('../../src/'))

project = 'Prime VI'
copyright = '2024, Wyatt Bridgman'
author = 'Wyatt Bridgman'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.inheritance_diagram',
    'myst_nb',
    'sphinxcontrib.bibtex'
    ]
bibtex_bibfiles = ['references.bib']
bibtex_default_style = 'unsrt'


templates_path = ['_templates']
exclude_patterns = []

# latex_elements = {'preamble': r'\input{latex_macros.tex.txt}'}
# latex_elements = {'preamble': r'\\input{blah.txt}'}
# latex_elements = {'preamble': '\\input{{blah.txt}}'}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
