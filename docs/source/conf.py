# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "doppler_dimming_lib"
copyright = "2024, Hervé Haudemand"
author = "Hervé Haudemand"
release = "1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
]

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

templates_path = ["_templates"]
exclude_patterns = []

# -- Options for latex output -------------------------------------------------
# latex_engine = "xelatex"
# latex_elements = {
#    "passoptionstopackages": r"""
# \PassOptionsToPackage{svgnames}{xcolor}
# """,
#    "fontpkg": r"""
# \setmainfont{DejaVu Serif}
# \setsansfont{DejaVu Sans}
# \setmonofont{DejaVu Sans Mono}
# """,
#    "preamble": r"""
# \usepackage[titles]{tocloft}
# \cftsetpnumwidth {1.25cm}\cftsetrmarg{1.5cm}
# \setlength{\cftchapnumwidth}{0.75cm}
# \setlength{\cftsecindent}{\cftchapnumwidth}
# \setlength{\cftsecnumwidth}{1.25cm}
# """,
#    "sphinxsetup": "TitleColor=DarkGoldenrod",
#    "fncychap": r"\usepackage[Bjornstrup]{fncychap}",
#    "printindex": r"\footnotesize\raggedright\printindex",
# }
# latex_show_urls = "footnote"


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = "alabaster"
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# needed for autodoc
import os
import sys

sys.path.insert(0, os.path.abspath("../../src/"))
