EDI12 - Energy Dispersive X-ray Diffraction (EDXD) Analysis Package
===================================================================

Version 0.5 - Monochromatic and Energy Dispersive Diffraction!

EDI12 has begun to transition away from looking purely at energy dispersive diffraction on the I12 beamline. The introduction of the mono_analysis module is the starting point of a generalisation of the package to tackle the single peak strain analysis from multiple sources.

Package to allow for the analysis and visulisation of data from the I12:JEEP EDXD detector. Specifically aimed at the extraction of strain through individual peak fitting. This package is designed to work with EDXD data stored in the NeXus format specified on the 23 element detector array utilised on the I12 beamline.

Peaks can be fitted to peaks in 1D, 2D and 3D data sets. The peak fitting is done across all detectors, thereby allowing for the construction of a Mohrs circle and the extraction of shear strain.
