Welcome to edi12's documentation!
=================================

The edi12 package provides a simple and lightweight means by which to extract
strain from Energy Dispersive X-ray Diffraction (EDXD) data collected at the
Diamond Light Source's I12:JEEP (EH2) beamline. The package's core functionality is
to take raw NeXus data files and output enhanced files which include detector
specific strain and the subsequently calculated in-plane principal strains. A plotting
module builds upon the analysis framework to allow for the convenient
visulisation of strain data, with notable emphasis being placed on the
visulisation of 2D-data sets. While 3D functionality is planned and partially
implemented, this is not currently a priority and support is limited.


Installation
=======

edi12 is designed to be cross platform and should work happily on Windows (under a variety of different Python frameworks), OS X, and Linux. The library has been (and continues to be) developed in Python 3. Efforts have been made to
ensure backwards compatibility with Python 2.7, with Python 3 specific syntax and packages being avoided. Despite this, testing has been limited and no guarantees can be made for the stability of the library under Python 2.
If you find a bug in Python 2 please feel free to submit a pull request.

Dependencies
------------

edi12 is built on Python's scientific stack (numpy, scipy, matplotlib). Additionally, the h5py package is required for the manipulation and management of the NeXus data files.
Testing and development were carried out using the Anaconda (v 2.5.0) package manager, which built with the following versions:

* Python: version 3.5.1
* numpy: version 1.10.4
* scipy: version 0.17
* matplotlib: version 1.5
* h5py: version 2.5.0

There is no reason to believe that older versions of the scientific stack won't be compatible but further testing is needed to guarantee this.

Installing edi12
----------------
Installing edi12 is easily done using pip<sup>1</sup>. Assuming it is installed, just run the following from the command-line:

::

    pip install edi12

This command will download the latest version of edi12 from the Python Package Index and install it to your system.

Alternatively, you can install from the distribution using the setup.py script. The source is stored in the GitHub repo, which can be browsed at:

* https://github.com/casimp/edi12

Simply download and unpack, then navigate to the download directory and run the following from the command-line:

::

    python setup.py install

<sup>1</sup> The package is still in a pre-release state and has not yet been pushed to PyPI. Installation must be completed directly from the GitHub repo.


Contents
========

.. toctree::
   :maxdepth: 2

   background
   tools
   example
   changes
   LICENSE


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
