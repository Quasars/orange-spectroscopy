[![DOI](https://zenodo.org/badge/53335377.svg)](https://zenodo.org/badge/latestdoi/53335377)

[![Test](https://github.com/Quasars/orange-spectroscopy/actions/workflows/test.yml/badge.svg)](https://github.com/Quasars/orange-spectroscopy/actions/workflows/test.yml)
[![Lint workflow](https://github.com/Quasars/orange-spectroscopy/actions/workflows/lint_workflow.yml/badge.svg)](https://github.com/Quasars/orange-spectroscopy/actions/workflows/lint_workflow.yml)
[![Release](https://github.com/Quasars/orange-spectroscopy/actions/workflows/release.yml/badge.svg)](https://github.com/Quasars/orange-spectroscopy/actions/workflows/release.yml)

Orange toolbox for spectral data analysis
=========================================

This is an add-on for [Orange3](https://orangedatamining.com/) for the analysis
of spectral data.

Installation
------------

To use this add-on, download and install the
[Quasar distribution of Orange](https://quasar.codes/), which comes with 
the Orange Spectroscopy add-on pre-installed.

Alternatively, you can install it into a pre-installed [Orange](https://orange.biolab.si/)
data mining suite with the "Add-ons" dialog, which can be opened from the Options menu.

Usage
-----

After the installation, the widgets from this add-on will appear in the toolbox 
under the section Spectroscopy.

For an introduction to this add-on, see the following YouTube channels:

* [Getting started with Orange](https://www.youtube.com/playlist?list=PLmNPvQr9Tf-ZSDLwOzxpvY-HrE0yv-8Fy) -
  introduces data analysis with Orange 
* [Spectral Orange](https://www.youtube.com/playlist?list=PLmNPvQr9Tf-bPWjDJvJBPZJ6us_KTAD5T) -
  tutorials that use the Spectroscopy add-on on spectral data 

For more, see the widget documentation:

* [Orange widgets](https://orange.biolab.si/toolbox/) - general data analysis widgets 
* [Spectroscopy widgets](https://orange-spectroscopy.readthedocs.io/) - 
  widgets specific to spectroscopy

For developers
--------------

If you would like to install from cloned git repository, run

    pip install .

To register this add-on with Orange, but keep the code in the development
directory (do not copy it to Python's site-packages directory), run

    pip install -e .[dev,test]

Further details can be found in [CONTRIBUTING.md](CONTRIBUTING.md)
