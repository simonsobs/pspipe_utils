================
PSpipe utilities
================

Useful functions for ``PSpipe``, the Simons Observatory and ACT CMB power spectrum pipeline.

.. image:: https://img.shields.io/pypi/v/pspipe_utils.svg?style=flat
   :target: https://pypi.python.org/pypi/pspipe_utils

.. image:: https://img.shields.io/badge/license-BSD-yellow
   :target: https://github.com/simonsobs/pspipe_utils/blob/main/LICENSE

.. image:: https://img.shields.io/github/actions/workflow/status/simonsobs/pspipe_utils/testing.yml?branch=main
   :target: https://github.com/simonsobs/pspipe_utils/actions

.. image:: https://codecov.io/gh/simonsobs/pspipe_utils/branch/main/graph/badge.svg?token=IVHHH73BI7
   :target: https://codecov.io/gh/simonsobs/pspipe_utils


Installing the code
-------------------

The easiest way to install and to use ``pspipe_utils`` likelihood is *via* ``pip``

.. code:: shell

    pip install pspipe_utils

If you want to dig into the code, you'd better clone this repository to some location

.. code:: shell

    git clone https://github.com/simonsobs/pspipe_utils.git /where/to/clone

Then you can install the ``pspipe_utils`` suite and its dependencies *via*

.. code:: shell

    pip install -e /where/to/clone

The ``-e`` option allow the developer to make changes within the ``pspipe_utils`` directory without
having to reinstall at every changes. If you plan to just use the utilities suite and do not develop
it, you can remove the ``-e`` option.
