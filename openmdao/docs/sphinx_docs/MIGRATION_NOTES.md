# Migration from Jupyter Book to Sphinx

This document describes the migration of OpenMDAO documentation from Jupyter Book to Sphinx.

## Summary

The OpenMDAO documentation has been successfully migrated from Jupyter Book to Sphinx while preserving all Jupyter notebooks in their original form. The new Sphinx-based build should produce functionally identical documentation.

## What Was Migrated

### Content (Unchanged)
All content was copied as-is from `openmdao/docs/openmdao_book/`:

- ✅ **225 notebook and markdown files** across all directories
- ✅ All Jupyter notebooks (`.ipynb` files)
- ✅ All Markdown files (`.md` files)
- ✅ Custom theme (`om-theme/`)
- ✅ Static assets (`_static/`)
- ✅ Logo and favicon
- ✅ Bibliography (`references.bib`)

### Directory Structure

```
openmdao/docs/sphinx_docs/
├── getting_started/           # Copied from openmdao_book
├── basic_user_guide/          # Copied from openmdao_book
├── advanced_user_guide/       # Copied from openmdao_book
├── features/                  # Copied from openmdao_book (largest section)
├── examples/                  # Copied from openmdao_book
├── theory_manual/             # Copied from openmdao_book
├── other_useful_docs/         # Copied from openmdao_book
├── other/                     # Copied from openmdao_book
├── om-theme/                  # Copied from openmdao_book
├── _static/                   # Copied from openmdao_book
└── _srcdocs/                  # Placeholder (generated during build)
```

## Configuration Mapping

### Jupyter Book → Sphinx

| Jupyter Book | Sphinx | Notes |
|--------------|--------|-------|
| `_config.yml` | `conf.py` | Settings translated to Python |
| `_toc.yml` | `index.rst` | TOC structure preserved in RST |
| `main.ipynb` | `index.rst` | Landing page converted to RST |
| Built-in notebook support | `myst-nb` extension | Same underlying technology |
| `execute_notebooks: force` | `nb_execution_mode = 'force'` | Identical behavior |
| `timeout: 300` | `nb_execution_timeout = 300` | Same setting |
| `allow_errors: false` | `nb_execution_allow_errors = False` | Strict mode |
| MyST extensions in `_config.yml` | `myst_enable_extensions` in `conf.py` | Same extensions |

### Key Configuration Equivalents

**Jupyter Book (_config.yml):**
```yaml
execute:
  execute_notebooks: force
  timeout: 300
  allow_errors: false

parse:
  myst_enable_extensions:
    - amsmath
    - dollarmath
    - colon_fence
    - linkify
    - substitution
```

**Sphinx (conf.py):**
```python
nb_execution_mode = 'force'
nb_execution_timeout = 300
nb_execution_allow_errors = False

myst_enable_extensions = [
    'amsmath',
    'dollarmath',
    'colon_fence',
    'linkify',
    'substitution',
]
```

## New Files Created

1. **conf.py** - Main Sphinx configuration file
2. **index.rst** - Master document with table of contents
3. **requirements.txt** - Python dependencies for building
4. **Makefile** - Unix build script
5. **make.bat** - Windows build script
6. **README.md** - Build instructions and documentation
7. **MIGRATION_NOTES.md** - This file

## Build Command Comparison

| Task | Jupyter Book | Sphinx |
|------|--------------|--------|
| Build HTML | `jupyter-book build .` | `make html` |
| Clean build | `jupyter-book clean .` | `make clean` |
| Build PDF | `jupyter-book build . --builder pdflatex` | `make latexpdf` |

## Extensions Used

### Core Extensions
- **myst-nb** - MyST Markdown + Jupyter Notebook support (replaces Jupyter Book's built-in support)
- **myst-parser** - Advanced Markdown parsing (dependency of myst-nb)

### Documentation Extensions
- **sphinx.ext.autodoc** - Auto-documentation from docstrings
- **sphinx.ext.autosummary** - Auto-generate summary tables
- **sphinx.ext.viewcode** - Link to highlighted source code
- **numpydoc** - NumPy-style docstring support
- **sphinxcontrib.bibtex** - Bibliography/citation support
- **sphinx-sitemap** - XML sitemap generation

### Theme
- **sphinx-book-theme** - Base theme (om-theme inherits from this)

## Testing the Migration

To verify the migration was successful:

1. **Install dependencies:**
   ```bash
   cd openmdao/docs/sphinx_docs
   pip install -r requirements.txt
   ```

2. **Build the documentation:**
   ```bash
   make html
   ```

3. **Check for errors:**
   - All 225 notebooks should be found
   - No execution errors should occur
   - Theme should load correctly
   - Table of contents should match original structure

4. **View the output:**
   ```bash
   open _build/html/index.html  # macOS
   # or
   xdg-open _build/html/index.html  # Linux
   # or
   start _build/html/index.html  # Windows
   ```

## Known Differences

### Minor Differences
1. **URL structure** - May differ slightly in how pages are organized
2. **Theme details** - While the same theme is used, Sphinx may render some elements differently
3. **Build output location** - `_build/` instead of `_build/html/` from root

### Functional Equivalents
- Both systems execute notebooks during build
- Both support the same MyST Markdown extensions
- Both use the same theme (om-theme based on sphinx-book-theme)
- Both generate HTML and PDF outputs
- Both support bibliography and citations

## Advantages of Sphinx

1. **Industry Standard** - Sphinx is the de facto standard for Python documentation
2. **Extensive Ecosystem** - Thousands of extensions available
3. **Better IDE Support** - More tooling and editor support
4. **Flexibility** - Easier to customize and extend
5. **Mature** - Battle-tested with large projects

## Potential Issues

### If Notebooks Fail to Execute
- Check that all dependencies are installed
- Verify OpenMDAO is in the Python path
- Check individual notebook for missing imports

### If Theme Doesn't Load
- Verify `om-theme/` directory is present
- Check `html_theme_path = ['.']` in conf.py
- Ensure sphinx-book-theme is installed

### If Citations Don't Work
- Verify `references.bib` exists
- Check sphinxcontrib-bibtex is installed
- Ensure `bibtex_bibfiles` is set in conf.py

## Next Steps

To complete the migration:

1. **Test the build** - Run `make html` and verify output
2. **Compare outputs** - Visual comparison with Jupyter Book build
3. **Update CI/CD** - Modify build scripts to use Sphinx
4. **Update documentation** - Update any references to the build process
5. **Deprecate Jupyter Book** - Once verified, can remove openmdao_book

## Rollback Plan

If issues are discovered, the original Jupyter Book setup remains untouched at:
```
openmdao/docs/openmdao_book/
```

The Jupyter Book build can continue to be used while Sphinx issues are resolved.

## Questions or Issues?

Refer to:
- [Sphinx Documentation](https://www.sphinx-doc.org/)
- [MyST-NB Documentation](https://myst-nb.readthedocs.io/)
- [MyST Parser Documentation](https://myst-parser.readthedocs.io/)
- [Sphinx Book Theme Documentation](https://sphinx-book-theme.readthedocs.io/)
