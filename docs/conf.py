# Configuration file for the Sphinx documentation builder.
#

import os
import sys
sys.path.insert(0, os.path.abspath('../'))


project = 'TSADmetrics'
copyright = '2025, Pedro Rafael Velasco Priego'
author = 'Pedro Rafael Velasco Priego'
release = 'MIT'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


extensions = ['sphinx.ext.duration', 'sphinx.ext.doctest', 'sphinx.ext.autodoc','sphinx.ext.mathjax']



templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store',"_add_docs"]

html_theme = 'furo'
html_static_path = ['_static']
html_theme_options = {
    #"sidebar_hide_name": True,
    "light_css_variables": {
        "color-brand-primary": "#2e5c7d",
        "color-brand-content": "#2e5c7d",
        "codebgcolor": "red",
        "codetextcolor": "red",
    },
    "dark_css_variables": {
        "color-brand-primary": "#6998b4",
        "color-brand-content": "#6998b4",
        "codebgcolor": "green",
        "codetextcolor": "green",
    },
    "navigation_with_keys": True

}
html_baseurl = ''

html_css_files = [
    'css/custom.css',
]

epub_show_urls = 'footnote'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output




### -- LaTeX options -------------------------------------------------

# comando para compilar: make latexpdf LATEXMKOPTS="-xelatex"

latex_elements = {
    'maxlistdepth': '10',  # Aumenta el límite de anidamiento
    'papersize': 'a4paper',
    'pointsize': '10pt',
    
}