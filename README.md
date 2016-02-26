EDI12: Energy Dispersive X-ray Diffraction (EDXD) Analysis Package
==================================================================

What is ED12?
-------------

EDI12 is a package developed to address bottlenecks in the diffraction-strain analysis workflow. It is vital that X-ray diffraction data acquired during synchrotron beamtimes is analysed in as close to real time as is possible. This allows for a tight feedback loop and ensures that decisions regarding acquisition and experimental parameters are optimized.

The EDI12 package therefore aims to allow for the efficient analysis and visualization of diffraction data acquired from these large scale facilities. It achieves this through the extraction of strain from individual peak fitting (i.e. not a Reitveld type refinement). Peak fitting routines can be run over 1D, 2D and 3D data sets. Peaks and strain are extracted as a function of azimuthal position (either detector position or caking angle). The resultant strain data is then further interrogated to facilitate the calculation of principal and shear strains. Analysed data is stored in the hdf5 file format and can be easily reloaded and visualised.

This package was originally designed to work with energy dispersive X-ray diffraction data (EDXD) stored in the NeXus format acquired on the I12:JEEP beamline at the Diamond Light Source, UK. As of version 0.5 EDI12 has begun to transition away from looking purely at energy dispersive diffraction on the I12 beamline. The introduction of the mono_analysis module is the starting point of a generalisation of the package to tackle the single peak strain analysis from both energy dispersive and monochromatic X-ray sources.

Requirements
------------

edi12 is built on Python’s scientific stack (numpy, scipy, matplotlib). Additionally, the h5py package is required for the manipulation and management of the NeXus data files. Testing and development were carried out using the Anaconda (v 2.5.0) package manager, which built with the following versions:

-	Python: version 3.5.1
-	numpy: version 1.10.4
-	scipy: version 0.17
-	matplotlib: version 1.5
-	h5py: version 2.5.0

The new mono analysis module relies on pyFAI, which is a software package developed at the ESRF, designed to reduce SAXS, WAXS and XRPD images recorded by area detectors to 1D plots or 2D patterns (known as caking or azimuthal regrouping). PyFAI functionality under python 3 is limited and how this will be best integrated into the EDI12 workflow is still under consideration.

-	pyFAI
-	fabIO

Documentation
-------------

Documentation is hosted by readthedocs. Note that it is still a work in progress(!) and more than a little rough around the edges. It does, however provide some background infromation and installation details:

http://edi12.readthedocs.org/en/latest/
