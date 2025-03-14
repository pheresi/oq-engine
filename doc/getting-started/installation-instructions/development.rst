.. _development:

Installing for development
==========================

To develop with the OpenQuake Engine and Hazardlib an installation from
sources must be performed. The easiest way it to use the `universal
installer <universal.md>`__. The guide here is for people wanting to do
everything manually.

Prerequisites
-------------

Knowledge of `Python <https://www.python.org/>`__ (and its `virtual
environments <https://docs.python.org/3.11/tutorial/venv.html>`__),
`git <https://git-scm.com/>`__ and `software
development <https://xkcd.com/844/>`__ are required.

Some software prerequisites are needed to build the development
environment. First of your you need a Python version supported by the
engine. At the moment we recommend Python 3.11, which is the only
version supported on Windows and macOS.

**NB: Python 3.12 are not supported yet, so please do NOT install such
versions**

**NB: Python version below 3.9 are not supported , so please do NOT
install such versions**

Linux
-----

Please check the documentation of your distribution to install one of
the supported version of Python. At the moment we recommend Python 3.11
but it also can used Python 3.9 and 3.10

macOS
-----

Please see instructions for the :doc:`Installing the OpenQuake Engine on MacOS <macos>`

Encoding
--------

Make sure that the encoding set in the terminal is ``en_US.UTF-8``. To
force it, you should put the following lines in your ``~/.profile``:

.. code:: bash

   export LC_ALL=en_US.UTF-8
   export LANG=en_US.UTF-8

Download the OpenQuake source code
----------------------------------

Considering that the complete repository is quite large given its long
history, we recommend shallow cloning the repository to download only
the latest revision.

.. code:: bash

   mkdir src && cd src
   git clone https://github.com/gem/oq-engine.git --depth=1

In case you needed the source code with the full history of the
repository, you can convert the shallow clone into a full repository
with the command ``git fetch --unshallow``.

Install OpenQuake in development mode
-------------------------------------

.. code:: bash

   $ cd oq-engine && python3 install.py devel

Loading and unloading the development environment
-------------------------------------------------

To exit from the OpenQuake development environment type ``deactivate``.
Before using again the OpenQuake software the environment must be
reloaded running ``source openquake/bin/activate``\ (assuming that it
has been installed under ‘openquake’). For more information about
*virtualenv*, see
http://docs.python-guide.org/en/latest/dev/virtualenvs/.

To load the virtual environment automatically at every login, add the
following line at the bottom of your ``~/.bashrc`` (Linux) or
``~/.profile`` (macOS):

.. code:: bash

   source $HOME/openquake/bin/activate

You can also add a short-hand command to enable it:

.. code:: bash

   alias oqenv="source $HOME/openquake/bin/activate"

Put it again at the bottom of ``~/.bashrc`` or ``~/.profile``; close and
re-open the terminal. You can now load your environment just typing
``oqenv``.

It is also possible to run the ``oq`` command without the corresponding
virtual environment loaded. Just run ``$HOME/openquake/bin/oq``; for
convenience you can also add it as an ``alias`` in your ``~/.bashrc``
(Linux) or ``~/.profile`` (macOS):

.. code:: bash

   alias oq="$HOME/openquake/bin/oq"

Multiple installations
----------------------

If any other installation of the Engine exists on the same machine - like
a system-wide installation made with packages - you should change the
DbServer port from the default one (1908) to any other unused port.

If the two installations have the same structure of the database and
the codebase is not that different (which is the normal case) everything will
work even without changing the DbServer port.

However, in general using a DbServer started from a different installation
could lead to unexpected behaviours and errors. To change
the DbServer port the file ``openquake.cfg`` must be
updated:

::

   [dbserver]          |  [dbserver]
   ## cut ##           |  ## cut ##
   port = 1908         >  port = 1985
   authkey = changeme  |  authkey = changeme
   ## cut ##           |  ## cut ##

The file should be put in your virtual environment.

Running the tests
-----------------

To run the OpenQuake Engine tests see the
`testing <https://github.com/gem/oq-engine/blob/master/doc/contributing/testing.md>`__
page.

Sync the source code with remote
--------------------------------

You can pull all the latest changes to the source code running

.. code:: bash

   cd oq-engine
   oq dbserver stop
   git pull

Uninstall the OpenQuake Engine
------------------------------

To uninstall the OpenQuake development make sure that its environment is
not loaded, typing ``deactivate``, and then remove the folder where it
has been installed: ``rm -Rf $HOME/openquake``.

Install third party software
----------------------------

It is possible to install, as an example, the `Silx HDF5
viewer <http://www.silx.org/>`__ in the same environment as the
OpenQuake Engine. To make that happen run the following commands via the
``oq-console.bat`` prompt:

.. code:: bash

   pip install PyQt5 silx

Silx viewer can be then run as

.. code:: bash

   silx view calc_NNN.hdf5

--------------

Notes
-----

If your system does not support the provided binary dependencies (python
wheels)

.. code:: bash

   pip install -e oq-engine

will try to download the required dependencies from
`pypi <http://pypi.python.org/>`__. This may require some extra work to
get all the external C dependencies resolved. Also, there is not
guarantee that the engine wil work, since newer versions of the
libraries could be incompatible. If you are using a non-standard python
distribution (like *macports* or *anaconda*) you should use tools
provided by such distribution to get the required dependencies.

--------------

Getting help
------------

If you need help or have questions/comments/feedback for us, you can
subscribe to the OpenQuake users mailing list:
https://groups.google.com/g/openquake-users
