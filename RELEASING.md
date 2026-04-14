Making a new release of Orange-Spectroscopy
===========================================


PRECHECKS
---------

1. Check state of tests on the master branch on github
2. Run tests locally.
3. Confirm that any changed requirements from setup.py were also 
   incorporated int conda/meta.yaml


PREPARE RELEASE
---------------

Prepare changelog:

    git log 0.9.1..master --first-parent --format='%b' > changelog.txt

Review and edit the changelog.

Bump version in pyproject.toml

Commit, start the commit message with "Release x.x.x", add the 
changelog to the commit message.

    git commit -a

Tag the release:

    git tag x.x.x

Now, build a conda package. This will test the release quite thoroughly.

    conda-build conda/

If conda-build succeeded, good, you may continue, otherwise throw away the
release commit and tag, fix problems, and restart.


BUILD PACKAGES
--------------

Continue only if conda-build ended without errors!

Push it to GitHub, push also tags:

    git push upstream master
    git push upstream --tags

Create a release on GitHub. Packages should be built and uploaded to PyPi automatically.


UPLOAD PACKAGES
---------------

Package is uploaded to PyPi automatically.

Copy the built conda package to the correct static/conda/noarch folder in 
the quasar.codes web page. Add the new files and then either push or 
make a pull request.
  
    git add . && git commit -a -m "Orange-Spectroscopy x.x.x"

Finally, add a new release on the orange-spectroscopy github page:
https://github.com/quasars/orange-spectroscopy
