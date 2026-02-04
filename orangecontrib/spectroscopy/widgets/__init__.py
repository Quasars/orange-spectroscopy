import sysconfig
# Category metadata.

# Category icon show in the menu
ICON = "icons/category.svg"

# Background color for category background in menu
# and widget icon background in workflow.
BACKGROUND = "light-blue"

# Location of widget help files.
WIDGET_HELP_PATH = (
    # Development documentation
    # You need to build help pages manually using
    # make htmlhelp
    # inside doc folder
    ("{DEVELOP_ROOT}/doc/build/htmlhelp/index.html", None),
    # Documentation included in wheel
    # Correct DATA_FILES entry is needed in setup.py and documentation has to be built
    # before the wheel is created.
    ("{}/help/orange-spectroscopy/index.html".format(sysconfig.get_path("data")), None),
    # Online documentation url, used when the local documentation is not available.
    # Url should point to a page with a section Widgets. This section should
    # includes links to documentation pages of each widget. Matching is
    # performed by comparing link caption to widget name.
    ("https://orange-spectroscopy.readthedocs.io/en/latest/", ""),
)
