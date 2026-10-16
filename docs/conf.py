"""Build the documentation without importing the operator library."""
project = "LunaEquivariance"
copyright = "LunaEquivariance Developers"
extensions = ["myst_parser", "sphinx.ext.mathjax"]
source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
root_doc = "index"
language = "en"
exclude_patterns = [
    "_build", "templates",
]
myst_heading_anchors = 3
html_theme = "pydata_sphinx_theme"
html_title = "LunaEquivariance"
html_static_path = ["_static"]
html_css_files = ["layout.css"]
html_js_files = ["navigation.js"]
html_show_sourcelink = False
html_sidebars = {"**": ["site-navigation.html"]}
templates_path = ["templates"]
html_logo = "_static/logo-horizontal.svg"
html_theme_options = {
    "navigation_depth": 3,
    "navbar_center": [],
    "navbar_end": ["theme-switcher", "github-link"],
    "secondary_sidebar_items": ["page-toc"],
    "show_toc_level": 2,
    "logo": {"alt_text": "LunaEquivariance"},
    "footer_start": ["footer-credit"],
    "footer_end": [],
}
nitpicky = True

myst_enable_extensions = ["fieldlist"]
