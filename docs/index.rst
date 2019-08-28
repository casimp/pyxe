Welcome to pyXe's documentation!
=================================

pyXe is a package developed to address bottlenecks in the diffraction-strain analysis workflow. It is vital that X-ray diffraction data acquired during synchrotron beamtimes is analysed in as close to real time as is possible. This allows for a tight feedback loop and ensures that decisions regarding acquisition and experimental parameters are optimized.

The pyXe package therefore aims to allow for the efficient analysis and visualization of diffraction data acquired from these large scale facilities. It achieves this through the extraction of strain from individual peak fitting (i.e. not a Reitveld type refinement). Peak fitting routines can be run over 1D, 2D and 3D data sets. Peaks and strain are extracted as a function of azimuthal position (either detector position or caking angle). The resultant strain data is then further interrogated to facilitate the calculation of principal and shear strains. Analysed data is stored in the hdf5 file format and can be easily reloaded and visualised.

This package was originally designed to work with energy dispersive X-ray diffraction data (EDXRD) stored in the NeXus format acquired on the I12:JEEP beamline at the Diamond Light Source, UK. pyXe is now, however, capable of carrying out both the single and multi peak strain analysis from both energy dispersive and monochromatic X-ray sources.

Requirements
------------

pyXe is built on Python’s scientific stack (numpy, scipy, matplotlib). The h5py package is also required for the manipulation and management of the NeXus/hdf5 data files. Development was carried out using the Anaconda (v 2019.03) package manager, which built with the following versions:

-	Python: version 3.7.3
-	numpy: version 1.16.2
-	scipy: version 1.2.1
-	matplotlib: version 3.0.3
-	h5py: version 2.9.0

Backward compatability to python 3.5 is likely but not guaranteed. Monochromatic XRD caking/azimuthal integration within pyXe relies on pyFAI (and fabIO), which is a software package developed at the ESRF, designed to reduce SAXS, WAXS and XRPD images recorded by area detectors to 1D plots or 2D patterns. This caking functionality is not currently under development within pyXe and recent developments within pyFAI may have broken this functionality. While this may be fixed in the future we currently advise that azimuthal integration be carried out as a pre-processing step at the beamline (using pyFAI at ESRF or DAWN at DLS); the pyXe monochromatic post-processing analysis platform should be flexible enough to deal with most data inputs (although interface modules will likely be required outside of the Diamond Light Source).

Installation
------------

Install from the distribution using the setup.py script. The source is stored in the GitHub repo, which can be found at:

https://github.com/casimp/pyxe

Simply download and unpack, then navigate to the download directory and run the following from the command-line:
::

    pip install .

Documentation
-------------

Documentation is hosted by readthedocs. Although still incomplete they do, however, provide some background information and installation details:

http://pyxe.readthedocs.org/en/latest/

Contents
========

.. toctree::
   :maxdepth: 2

   background
   example_EDXD
   changes
   LICENSE


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
