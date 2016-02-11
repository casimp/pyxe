=================================
Welcome to edi12's documentation!
=================================

The edi12 package provides a simple and lightweight means by which to extract
strain from Energy Dispersive X-ray Diffraction (EDXD) data collected at the
Diamond Light Source's I12 (EH2) beamline. The package's core functionality is
to take raw NeXus data files and output enhanced files which include detector
specific strain and the associated, complete in-plane strain state. A plotting
module builds upon the analysis framework to allow for the convenient
visulisation of strain data, with notable emphasis being placed on the
visulisation of 2D-data sets. While 3D functionality is planned and partially
implemented, this is not currently a priority and support is limited.

Installation
============

edi12 is designed to be cross platform and should work happily on Microsoft
Windows (under a variety of different Python frameworks), Mac OS X, and Linux.

Dependancies
------------

Key:
- numpy
- scipy
- h5py

Additional (plotting):
- matplotlib


Support
=======

This library is currently being developed in Python 3. Efforts are being made to
ensure backwards compatibility with Python 2.7 but testing has been limited. If
you find a bug in Python 2 please feel free to submit a pull request.

This library has been primarily developed in Windows. I have no intention to
write fixes or test for platform specific bugs with every update. Pull requests
that fix those issues are always welcome though.

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
