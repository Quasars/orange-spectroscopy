# How to contribute

Thanks for your interest in contributing to the Orange-Spectroscopy add-on!
The following documents how to get started with a development installation, and
some preferred procedures for getting your contribution included in the project.

## Orange vs Spectroscopy Add-on

The Orange-Spectroscopy add-on extends and enhances the Orange data mining suite
to enable the analysis of spectroscopic data. Your intended enhancement might be
more generally useful or improve on something already present in the main Orange3
code, in which case you might want to work there. Get in touch by opening an issue
with your idea if you are not sure where it belongs.

## Getting Started

You will want a "fork" repository to base your changes. We follow the
fork -> branch -> pull request -> merge workflow common on GitHub. See
[https://help.github.com/articles/fork-a-repo/](https://help.github.com/articles/fork-a-repo/)
for more details.

Once you have a git checkout of the add-on code, you will want some kind of virtual
environment to keep your development separate from your regular Orange install (and
other Python work/programs you may have). The follow describes how to do this using
the Anaconda conda environment system, but can also apply to virtualenvs.

Run "Anaconda Prompt" or similar and:

    conda config --add channels conda-forge
    conda create --name="orange-spectroscopy" orange3
    conda activate orange-spectroscopy

Navigate to your orange-spectroscopy src directory, then install in development mode:

    pip install -e .[dev,test]

It is highly recommended to set up `pre-commit` hooks to automatically check your code quality before each commit:

    pre-commit install

If all went well, you should be able to run the tests:

    python -m unittest orangecontrib.spectroscopy.tests

And run Orange (with spectroscopy widgets present):

    orange-canvas

Each time you want to use the development version, you must activate the environment first:

    conda activate orange-spectroscopy

## Communication

It's a good idea to let the team know what you're working on before embarking,
especially for a large project. We prefer to discuss this in GitHub issues, so please
file one describing your contribution before getting too far in.

For large projects in particular, consider working in a public Pull Request marked
`[WIP]` (work-in-progress) and ask for feedback along the way.

## Making Changes

* Create a topic branch from where you want to base your work.
  * This is usually the master branch.
  * To quickly create a topic branch based on master, run `git checkout -b
    my-contribution master`. Please avoid working directly on the
    `master` branch.
* Make commits of logical and atomic units.
* Make sure your commit messages are clear and reference the code/module you are changing.
* **Make sure you have added the necessary tests for your changes.**
* Run _all_ the tests to assure nothing else was accidentally broken.
  * `python -m unittest orangecontrib.spectroscopy.tests`
* Check your code quality with ruff:
  * `ruff check .`
* Please add appropriate **documentation** for your new or changed feature.

## Submitting Changes

* Push your changes to a topic branch in your fork of the repository.
* Submit a pull request to the repository.
* Automated tests (the same as you ran yourself above) will be run on your branch.
* A core team member(s) with the appropriate expertise will review your proposal
  and merge if appropriate.

## Thanks!
