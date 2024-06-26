#!/usr/bin/env python

import os
from os import walk, path
import subprocess
import sys

from setuptools import setup, find_packages, Command

PACKAGES = find_packages()

PACKAGE_DATA = {}

README_FILE = os.path.join(os.path.dirname(__file__), 'README.pypi')
LONG_DESCRIPTION = open(README_FILE, "rt", encoding="utf8").read()

DATA_FILES = [
    # Data files that will be installed outside site-packages folder
]

ENTRY_POINTS = {
    # Entry points that marks this package as an orange add-on. If set, addon will
    # be shown in the add-ons manager even if not published on PyPi.
    'orange3.addon': (
        'spectroscopy = orangecontrib.spectroscopy',
    ),

    # Entry point used to specify packages containing tutorials accessible
    # from welcome screen. Tutorials are saved Orange Workflows (.ows files).
    'orange.widgets.tutorials': (
        # Syntax: any_text = path.to.package.containing.tutorials
        'infraredtutorials = orangecontrib.spectroscopy.tutorials',
    ),

    # Entry point used to specify packages containing widgets.
    'orange.widgets': (
        # Syntax: category name = path.to.package.containing.widgets
        # Widget category specification can be seen in
        #    orangecontrib/example/widgets/__init__.py
        'Spectroscopy = orangecontrib.spectroscopy.widgets',
    ),

    # Register widget help
    "orange.canvas.help": (
        'html-index = orangecontrib.spectroscopy.widgets:WIDGET_HELP_PATH',)

}

KEYWORDS = [
    # [PyPi](https://pypi.python.org) packages with keyword "orange3 add-on"
    # can be installed using the Orange Add-on Manager
    'orange3 add-on',
    'spectroscopy',
    'infrared'
]


class CoverageCommand(Command):
    """A setup.py coverage subcommand developers can run locally."""
    description = "run code coverage"
    user_options = []
    initialize_options = finalize_options = lambda self: None

    def run(self):
        """Check coverage on current workdir"""
        sys.exit(subprocess.call(r'''
        coverage run -m unittest
        echo; echo
        coverage report
        coverage html &&
            { echo; echo "See also: file://$(pwd)/htmlcov/index.html"; echo; }
        ''', shell=True, cwd=os.path.dirname(os.path.abspath(__file__))))


class LintCommand(Command):
    """A setup.py lint subcommand developers can run locally."""
    description = "run code linter(s)"
    user_options = []
    initialize_options = finalize_options = lambda self: None

    def run(self):
        """Lint current branch compared to a reasonable master branch"""
        sys.exit(subprocess.call(r'''
        set -eu
        upstream="$(git remote -v |
                    awk '/[@\/]github.com[:\/]Quasars\/orange-spectroscopy[\. ]/{ print $1; exit }')"
        git fetch -q $upstream master
        best_ancestor=$(git merge-base HEAD refs/remotes/$upstream/master)
        .travis/check_pylint_diff $best_ancestor
        ''', shell=True, cwd=os.path.dirname(os.path.abspath(__file__))))


TEST_SUITE = "orangecontrib.spectroscopy.tests.suite"


def include_documentation(local_dir, install_dir):
    global DATA_FILES

    doc_files = []
    for dirpath, _, files in walk(local_dir):
        doc_files.append((dirpath.replace(local_dir, install_dir),
                          [path.join(dirpath, f) for f in files]))
    DATA_FILES.extend(doc_files)


if __name__ == '__main__':

    cmdclass = {
        'coverage': CoverageCommand,
        'lint': LintCommand,
    }

    include_documentation('doc/build/htmlhelp', 'help/orange-spectroscopy')

    setup(
        name="Orange-Spectroscopy",
        python_requires='>3.8.0',
        description='Extends Orange to handle spectral and hyperspectral analysis.',
        long_description=LONG_DESCRIPTION,
        long_description_content_type='text/markdown',
        author='Canadian Light Source, Biolab UL, Soleil, Elettra',
        author_email='marko.toplak@gmail.com',
        version="0.6.14",
        packages=PACKAGES,
        package_data=PACKAGE_DATA,
        data_files=DATA_FILES,
        install_requires=[
            'setuptools>=36.3',  # same as for Orange 3.28
            'pip>=9.0',  # same as for Orange 3.28
            'numpy>=1.20.0,<2.0.0',
            'Orange3>=3.34.0',
            'orange-canvas-core>=0.1.28',
            'orange-widget-base>=4.19.0',
            'scipy>=1.9.0',
            'scikit-learn>=1.0.1',
            'spectral>=0.22.3,!=0.23',
            'serverfiles>=0.2',
            'AnyQt>=0.1.0',
            'pyqtgraph>=0.12.2,!=0.12.4',  # https://github.com/pyqtgraph/pyqtgraph/issues/2237
            'colorcet',
            'h5py',
            'extranormal3 >=0.0.3',
            'renishawWiRE>=0.1.8',
            'pillow',
            'lmfit>=1.0.2,<1.3.0',
            'bottleneck',
            'pebble',
            'agilent-format>=0.4.5'
        ],
        extras_require={
            'test': ['coverage'],
            'doc': ['sphinx', 'recommonmark'],
        },
        entry_points=ENTRY_POINTS,
        keywords=KEYWORDS,
        namespace_packages=['orangecontrib'],
        test_suite=TEST_SUITE,
        include_package_data=True,
        zip_safe=False,
        url="https://github.com/Quasars/orange-spectroscopy",
        cmdclass=cmdclass,
        license='GPLv3+',
    )
