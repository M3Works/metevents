=========
metevents
=========


.. image:: https://img.shields.io/pypi/v/metevents.svg
        :target: https://pypi.python.org/pypi/metevents
.. image:: https://github.com/M3Works/metevents/actions/workflows/testing.yml/badge.svg
        :target: https://github.com/M3Works/metevents/actions/workflows/testing.yml
        :alt: Testing Status
.. image:: https://readthedocs.org/projects/metevents/badge/?version=latest
        :target: https://metevents.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status


Meteorlogical Events

metevents is a python library created with the goal of consistent, simple identification of
meteorology timeseries data. metevents is developed by `M3 Works <https://m3works.io>`_ as a tool
for performing analysis for hydrology models and station data. Contributions welcome!

Warning - This software is provided as is (see the license), so use at your own risk.
This is an opensource package with the goal of making data wrangling easier. We make
no guarantees about the quality or accuracy of the data and any interpretation of the meaning
of the data is up to you.


* Free software: BSD license


Features
--------


Requirements
------------
python >= 3.7

Install
-------
.. code-block:: bash

    python3 -m pip install metevents


Local install for dev
---------------------
The recommendation is to use virtualenv, but other local python
environment isolation tools will work (pipenv, conda)

.. code-block:: bash

    python3 -m pip install --upgrade pip
    python3 -m pip install -r requirements_dev
    python3 -m pip install .

Testing
-------

.. code-block:: bash

    pytest

If contributing to the codebase, code coverage should not decrease
from the contributions. Make sure to check code coverage before
opening a pull request.

.. code-block:: bash

    pytest --cov=metevents

Documentation
-------------
readthedocs coming soon

https://metevents.readthedocs.io.

Usage
-----
See usage documentation https://metevents.readthedocs.io/en/latest/usage.html


Usage Examples
==============



Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
