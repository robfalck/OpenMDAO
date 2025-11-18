# OpenMDAO Sphinx Documentation

This directory contains the Sphinx-based documentation for OpenMDAO, migrated from Jupyter Book.

## Overview

This documentation setup uses Sphinx with MyST-NB to build documentation from Jupyter notebooks and Markdown files. The configuration preserves the Jupyter notebooks in their original form while using Sphinx as the build system.

## Key Features

- **MyST-NB**: Enables Sphinx to parse and execute Jupyter notebooks
- **MyST Markdown**: Supports advanced Markdown features including directives and roles
- **Custom Theme**: Uses the `om-theme` custom theme (based on sphinx-book-theme)
- **Notebook Execution**: Notebooks are executed during the build process
- **Bibliography Support**: BibTeX citations via sphinxcontrib-bibtex
- **Auto-documentation**: API docs generated from Python docstrings

## Installation

1. Install the required packages:

```bash
pip install -r requirements.txt
```

2. Ensure OpenMDAO is installed in your environment:

```bash
pip install openmdao
```

## Building the Documentation

### HTML Documentation

To build the HTML documentation:

```bash
make html
```

The built documentation will be in `_build/html/`. Open `_build/html/index.html` in your browser to view it.

### PDF Documentation

To build the LaTeX/PDF documentation:

```bash
make latexpdf
```

The PDF will be in `_build/latex/openmdao.pdf`.

### Clean Build

To clean all build artifacts:

```bash
make clean
```

### Other Build Targets

To see all available build targets:

```bash
make help
```

## Directory Structure

```
sphinx_docs/
├── conf.py                    # Sphinx configuration
├── index.rst                  # Main documentation index
├── requirements.txt           # Python dependencies
├── Makefile                   # Unix build script
├── make.bat                   # Windows build script
├── README.md                  # This file
├── references.bib             # Bibliography file
├── OpenMDAO_Logo.png          # Logo image
├── favicon.svg                # Favicon
├── om-theme/                  # Custom Sphinx theme
├── _static/                   # Static assets (CSS, JS, etc.)
├── _srcdocs/                  # Auto-generated source documentation
├── getting_started/           # Getting started guide
├── basic_user_guide/          # Basic tutorials
├── advanced_user_guide/       # Advanced tutorials
├── features/                  # Feature reference (largest section)
├── examples/                  # Example problems
├── theory_manual/             # Theory and algorithms
└── other_useful_docs/         # Developer docs, tools, etc.
```

## Configuration

### Notebook Execution

The documentation is configured to force-execute all notebooks on each build:

- Execution mode: `force`
- Timeout: 300 seconds per notebook
- Error handling: Strict (no errors allowed)
- Output: stderr is displayed

These settings are in [conf.py](conf.py):

```python
nb_execution_mode = 'force'
nb_execution_timeout = 300
nb_execution_allow_errors = False
nb_execution_show_tb = True
```

### MyST Extensions

The following MyST markdown extensions are enabled:

- `amsmath`: Advanced LaTeX math
- `colon_fence`: Colon-style code blocks
- `dollarmath`: Dollar-sign math delimiters
- `linkify`: Automatic URL detection
- `substitution`: Variable substitution

## Differences from Jupyter Book

While this Sphinx setup aims to be functionally equivalent to the Jupyter Book build, there are some differences:

1. **Build System**: Uses `sphinx-build` instead of `jupyter-book build`
2. **Configuration**: Settings are in `conf.py` instead of `_config.yml` and `_toc.yml`
3. **TOC Format**: Table of contents is in RST format in `index.rst` instead of YAML
4. **Extensions**: Uses MyST-NB instead of Jupyter Book's built-in notebook support

## Troubleshooting

### Notebooks Failing to Execute

If notebooks fail during execution:

1. Check that all required dependencies are installed
2. Verify OpenMDAO is properly installed
3. Check the error output for missing modules or other issues
4. You can temporarily disable execution by setting `nb_execution_mode = 'off'` in conf.py

### Theme Not Found

If the custom theme is not found:

1. Verify `om-theme/` directory exists
2. Check that `html_theme_path = ['.']` is set in conf.py
3. Ensure all theme files are present in `om-theme/`

### Missing Bibliography

If citations fail:

1. Verify `references.bib` exists
2. Check that `sphinxcontrib-bibtex` is installed
3. Ensure `bibtex_bibfiles = ['references.bib']` is set in conf.py

## Development

When adding new content:

1. Add notebook/markdown files to the appropriate directory
2. Update `index.rst` to include the new files in the TOC
3. Run `make html` to test the build
4. Check for any execution errors or warnings

## Migration Notes

This Sphinx documentation was migrated from Jupyter Book with the following approach:

- **Content**: All notebooks and markdown files copied as-is
- **Structure**: TOC structure preserved from `_toc.yml`
- **Theme**: Custom om-theme copied directly
- **Configuration**: Jupyter Book settings translated to Sphinx equivalents
- **Extensions**: MyST-NB chosen for maximum compatibility with existing MyST markdown

The goal is to produce identical documentation output while using Sphinx as the build system.
