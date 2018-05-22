[![DOI](https://zenodo.org/badge/53335377.svg)](https://zenodo.org/badge/latestdoi/53335377)

Travis test

Orange for spectral data
========================

This is an add-on for [Orange3](http://orange.biolab.si) for the analysis
of spectral data.

Installation
------------

To use this add-on, first download and install the current version of
[Orange3](http://orange.biolab.si). Next, install the add-on: 

1. Open Orange Canvas, choose "Options" from the menu and then "Add-ons". 
2. A new window will open. There, tick the checkbox in front of "Orange-Infrared" and confirm. 
3. Restart Orange.

Usage
-----

After the installation, the widgets from this add-on are registered with
Orange. The new widgets will appear in Orange Canvas, in the toolbox bar 
under the section Infrared.

For developers
--------------

If you would like to install from cloned git repository, run

    pip install .

To register this add-on with Orange, but keep the code in the development
directory (do not copy it to Python's site-packages directory), run

    pip install -e .
