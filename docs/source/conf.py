# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

project = "QuantumSpectra-2024"
copyright = "2024, Ben Koppe"
author = "Ben Koppe"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    # "sphinx.ext.napoleon",
    "numpydoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.duration",
    "sphinx.ext.intersphinx",
    "sphinx_rtd_theme",
    "sphinx_copybutton",
    "sphinx.ext.autosummary",
    # "autoapi.extension",
    "sphinx.ext.autosectionlabel",
    # "nbsphinx",
    # "sphinx_autodoc_typehints",
]

templates_path = ["_templates"]
exclude_patterns = [
    # Sometimes sphinx reads its own outputs as inputs!
    "build/html",
    "README.md",
    "_build",
]

pygments_style = "sphinx"

autodoc_member_order = "bysource"
# autodoc_mock_imports = ["jaxtyping"]
autodoc_typehints_format = "short"


# def process_jaxtyping_docstrings(app, what, name, obj, options, lines):
#     for i, line in enumerate(lines):
#         if "Float[Array," in line:
#             print(line)
#             lines[i] = line.replace('Float[Array, "num_modes"]', ":ref:`jaxtyping`")
#             print(lines[i])


# def setup(app):
#     app.connect("autodoc-process-docstring", process_jaxtyping_docstrings)


autosummary_generate = True

# autoclass_content = "class"

numpydoc_show_class_members = True
numpydoc_class_members_toctree = False
# numpydoc_xref_param_type = True
# numpydoc_xref_aliases = {

# }

autosectionlabel_prefix_document = True

autoapi_dirs = ["../../quantumspectra_2024"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
}


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_logo = "qs-logo.png"

html_theme_options = {
    "collapse_navigation": False,
    "logo_only": True,
}

sys.path.insert(0, os.path.abspath("../.."))
