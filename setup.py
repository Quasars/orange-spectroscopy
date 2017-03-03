#!/usr/bin/env python

import os
from setuptools import setup, find_packages, Command
import subprocess
import sys

ENTRY_POINTS = {
    # Entry point used to specify packages containing tutorials accessible
    # from welcome screen. Tutorials are saved Orange Workflows (.ows files).
    'orange.widgets.tutorials': (
        # Syntax: any_text = path.to.package.containing.tutorials
        'exampletutorials = orangecontrib.infrared.tutorials',
    ),

    # Entry point used to specify packages containing widgets.
    'orange.widgets': (
        # Syntax: category name = path.to.package.containing.widgets
        # Widget category specification can be seen in
        #    orangecontrib/example/widgets/__init__.py
        'Infrared = orangecontrib.infrared.widgets',
    ),
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
        coverage run --source=orangecontrib.infrared -m unittest
        echo; echo
        coverage report --omit="*/tests/*"
        coverage html --omit="*/tests/*" &&
            { echo; echo "See also: file://$(pwd)/htmlcov/index.html"; echo; }
        ''', shell=True, cwd=os.path.dirname(os.path.abspath(__file__))))


if 'test' in sys.argv:
    extra_setuptools_args = dict(
        test_suite='orangecontrib.infrared.tests',
    )
else:
    extra_setuptools_args = dict()

if __name__ == '__main__':

    cmdclass = {
       'coverage': CoverageCommand,
    }


    setup(
        name="Orange-Infrared",
        description='',
        author='Canadian Light Source, Biolab UL, Soleil, Elettra',
        author_email='marko.toplak@gmail.com',
        version="0.1.4",
        packages=find_packages(),
        install_requires=[
            'Orange3>=3.3.12',
            'scipy>=0.14.0',
            'spectral>=0.18',
            'opusFC>=1.1.0',
            'serverfiles>=0.2',
            'AnyQt>=0.0.6',
            'pyqtgraph>=0.10.0',
        ],
        entry_points=ENTRY_POINTS,
        keywords=KEYWORDS,
        namespace_packages=['orangecontrib'],
        include_package_data=True,
        zip_safe=False,
        url="https://github.com/markotoplak/orange-infrared",
        cmdclass=cmdclass,
        **extra_setuptools_args
    )
