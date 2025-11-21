# Configuration file for the Sphinx documentation builder.
#
# This file was migrated from Jupyter Book to Sphinx.
# For the full list of built-in configuration values, see:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import os

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here.
# sys.path.insert(0, os.path.abspath('.'))

# -- Project information -----------------------------------------------------

project = 'OpenMDAO'
copyright = '2025, The OpenMDAO Development Team'
author = 'The OpenMDAO Development Team'

# The version info for the project
# version = ''  # The short X.Y version
# release = ''  # The full version, including alpha/beta/rc tags

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings.
extensions = [
    'myst_nb',                      # MyST Markdown and Jupyter Notebook support
    'sphinx.ext.autodoc',           # Auto-documentation from docstrings
    'sphinx.ext.autosummary',       # Auto-generate summary tables
    'sphinx.ext.viewcode',          # Add links to highlighted source code
    'sphinx_sitemap',               # Generate XML sitemap
    'numpydoc',                     # NumPy-style docstring support
    'sphinxcontrib.bibtex',         # Bibliography support
    'sphinx_design',                # Design elements (tabs, cards, grids, etc.)
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = [
    '_build',
    'Thumbs.db',
    '.DS_Store',
    '**.ipynb_checkpoints',
    'template.ipynb',
    'README.md',              # Developer documentation, not for users
    'MIGRATION_NOTES.md',     # Internal migration notes, not for users
]

# The suffix(es) of source filenames.
source_suffix = {
    '.rst': 'restructuredtext',
    '.ipynb': 'myst-nb',
    '.md': 'myst-nb',
}

# The master toctree document.
master_doc = 'index'

# -- MyST-NB Configuration ---------------------------------------------------

# Execution mode for notebooks
# 'force': Execute all notebooks on each build
# 'auto': Only execute notebooks that don't have outputs
# 'cache': Cache execution results
# 'off': Don't execute notebooks
nb_execution_mode = os.environ.get("SPHINX_NB_EXECUTION_MODE", "auto")

nb_execution_engine = 'ipyparallel'
nb_ipyparallel_url_file = None

# Configure ipyparallel to use the profile that run_mpi_cluster.py creates
# This allows MyST-NB to find and use the cluster started by build_sphinx_docs.py
# and also allows notebooks that use Client(profile="mpi") to connect
nb_ipyparallel_client_args = {
    'profile': 'mpi'
}

# Show stderr output during notebook execution
nb_execution_show_tb = True

# Raise an error on execution failure
nb_execution_allow_errors = False

# Timeout for notebook execution (in seconds)
nb_execution_timeout = 300

# MyST Markdown extensions
myst_enable_extensions = [
    'amsmath',          # LaTeX math
    'colon_fence',      # Colon fence code blocks
    'dollarmath',       # Dollar sign math delimiters
    'linkify',          # Automatically detect URLs
    'substitution',     # Variable substitution
]

# Enable parallel execution
nb_execution_in_parallel = True

# Explicitly set the number of processes to the available CPU count
nb_execution_session_config = {
    "n_processes": os.cpu_count()
}

# -- BibTeX Configuration ----------------------------------------------------

bibtex_bibfiles = ['references.bib']

# -- HTML output options -----------------------------------------------------

# The theme to use for HTML and HTML Help pages.
html_theme = 'pydata_sphinx_theme'

# Theme options for pydata-sphinx-theme
html_theme_options = {
    "logo": {
        "text": "OpenMDAO",
        "image_light": "OpenMDAO_Logo.png",
        "image_dark": "OpenMDAO_Logo.png",
    },
    "github_url": "https://github.com/OpenMDAO/OpenMDAO",
    "use_edit_page_button": True,
    "show_toc_level": 2,
    "navbar_align": "left",
    "navbar_center": ["navbar-nav"],  # Show main navigation in navbar
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "navigation_with_keys": False,
    "collapse_navigation": False,  # Keep sidebar expanded
    "navigation_depth": 4,  # Show 4 levels in sidebar
    "footer_start": ["copyright"],
    "footer_center": ["sphinx-version"],
    # Configure top navbar links to main sections
    "navbar_start": ["navbar-logo"],
    "header_links_before_dropdown": 6,  # Show all main sections before dropdown
    # External links in navbar
    "external_links": [],
    # Navigation structure
    "show_nav_level": 1,
}

# Context for "Edit on GitHub" button
html_context = {
    "github_user": "OpenMDAO",
    "github_repo": "OpenMDAO",
    "github_version": "master",
    "doc_path": "openmdao/docs/sphinx_docs",
}

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
html_logo = 'OpenMDAO_Logo.png'

# The name of an image file (within the static path) to use as favicon of the
# docs. This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
html_favicon = 'favicon.svg'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Base URL for the documentation (used for sitemap generation)
html_baseurl = 'https://openmdao.org/newdocs/versions/latest/'

# -- LaTeX output options ----------------------------------------------------

latex_documents = [
    (master_doc, 'openmdao.tex', 'OpenMDAO Documentation',
     'The OpenMDAO Development Team', 'manual'),
]

# -- Sitemap Configuration ---------------------------------------------------

sitemap_filename = 'sitemap.xml'

# -- AutoDoc Configuration ---------------------------------------------------

# This value controls how to represent typehints
autodoc_typehints = 'description'

# -- NumPyDoc Configuration --------------------------------------------------

# Whether to show all members of a class in the Methods and Attributes sections
numpydoc_show_class_members = False
