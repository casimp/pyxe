pyXe: X-ray Diffraction Strain Analysis Package
===============================================

What is pyXe?
-------------

PyXe is a package developed to address bottlenecks in the diffraction-strain analysis workflow. It is vital that X-ray diffraction data acquired during synchrotron beamtimes is analysed in as close to real time as is possible. This allows for a tight feedback loop and ensures that decisions regarding acquisition and experimental parameters are optimized.

The pyXe package therefore aims to allow for the efficient analysis and visualization of diffraction data acquired from these large scale facilities. It achieves this through the extraction of strain from individual peak fitting (i.e. not a Reitveld type refinement). Peak fitting routines can be run over 1D, 2D and 3D data sets. Peaks and strain are extracted as a function of azimuthal position (either detector position or caking angle). The resultant strain data is then further interrogated to facilitate the calculation of principal and shear strains. Analysed data is stored in the hdf5 file format and can be easily reloaded and visualised.

This package was originally designed to work with energy dispersive X-ray diffraction data (EDXD) stored in the NeXus format acquired on the I12:JEEP beamline at the Diamond Light Source, UK. As of version 0.5 pyXe has begun to transition away from looking purely at energy dispersive diffraction on the I12 beamline. The introduction of the mono_analysis module is the starting point of a generalisation of the package to tackle the single peak strain analysis from both energy dispersive and monochromatic X-ray sources.

Requirements
------------

PyXe is built on Python’s scientific stack (numpy, scipy, matplotlib). Additionally, the h5py package is required for the manipulation and management of the NeXus data files. Testing and development were carried out using the Anaconda (v 2.5.0) package manager, which built with the following versions:

-	Python: version 3.5.1
-	numpy: version 1.10.4
-	scipy: version 0.17
-	matplotlib: version 1.5
-	h5py: version 2.5.0

The new mono analysis module relies on pyFAI, which is a software package developed at the ESRF, designed to reduce SAXS, WAXS and XRPD images recorded by area detectors to 1D plots or 2D patterns (known as caking or azimuthal regrouping). PyFAI functionality under python 3 is limited and how this will be best integrated into the pyXe workflow is still under consideration. Testing has so far been completed using:

-	Python: version 2.7.11
-	pyFAI: version 0.11.0
-	fabIO: version 0.3.0

Installation
------------

Installing pyXε is easily done using pip<sup>1</sup>. Assuming it is installed, just run the following from the command-line:

```
pip install pyxe
```

This command will download the latest version of pyXε from the Python Package Index and install it to your system.

Alternatively, you can install from the distribution using the setup.py script. The source is stored in the GitHub repo, which can be browsed at:

-	https://github.com/casimp/pyxe

Simply download and unpack, then navigate to the download directory and run the following from the command-line:

```
python setup.py install
```

<sup>1</sup> The package is still in a pre-release state and has not yet been pushed to PyPI. Installation must be completed directly from the GitHub repo.

Documentation
-------------

Documentation is hosted by readthedocs. Although still incomplete they do, however, provide some background information and installation details:

http://edi12.readthedocs.org/en/latest/
