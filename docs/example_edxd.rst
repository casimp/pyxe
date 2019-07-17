EDXRD Analysis (DLS - I12:JEEP)
=========================

I12 and EDXRD
------------

*Beamline I12:JEEP (Joint Engineering, Environmental, and Processing) is a high energy X-ray beamline for imaging, diffraction and scattering, which operates at energies of 53-150 keV.*

The I12-JEEP beamline is located at the Diamond Light Source (DLS) in Oxfordshire, UK. Specifications for the beamline have be detailed in the Journal of Synchrontron Radiation [2]. Much of this information is replicated on the DLS's website:

http://www.diamond.ac.uk/Beamlines/Engineering-and-Environment/I12.html

The I12-JEEP beamline can be accessed through two separate experimental hutches. Experimental Hutch 2 (EH2), which is the larger of the hutches, contains the energy dispersive X-ray detector (EDXRD). The layout of the detector can be seen below:

.. figure:: EDXRD.png
    :figwidth: 400px
    :width: 500px
    :alt: EDXRD setup

*The EDXRD system. (a) Geometry of the detector, detector slits and sample slits showing the semi-annular arrangement of 23 independent Ge crystals [2].*

..

The detector is comprised of 23 elements spaced in steps of 8.2°, covering an azimuthal range from 0 to 180°. An additional, unused, detector is available in the case that one detector should fail. The data array that is output contains reference to this detector but it is ignored during the analysis. Detailed information about the EDXRD setup can be found in the previously noted journal article and on the DLS website:

http://www.diamond.ac.uk/Beamlines/Engineering-and-Environment/I12/detectors/EDXRD.html


Data Analysis
-------------

I12 and specifically EH2, houses a 23-element energy dispersive X-ray detector.
Data acquired using this detector is stored in the NeXus format, which is a
data format used principally to store data from X-ray and neutron sources.

The NeXus file contains raw data and meta-data pertaining to each scan. In the
case of the energy dispersive detector it notably contains the location and
raw energy spectra (in keV) from each data point across each detector. These energy
spectra are calibrated and converted to the equivalent momentum transfer vector, q.

These values of q are stored in a multi-dimensional array, such that a scan of
size 10 x 10 would have an associated data array of size 10 x 10 x 24 x 4026.
4026 is the number of bins to collect the data and 24 is the number of detectors.
The 24th detector is unused and is there as a back-up.

The analysis is carried out across each detector, with peaks being calculated
according to a specified value for q0 (or a list of multiple q0 values - [q0_1, q0_2]).
The peaks are found and stored along with the peak errors. This data is
converted to strain (and strain_err) relative to the q0 (or q0s) that were originally specified.


Code Example
------------

The initial analysis step was designed to be simple and complete. Data from all
detectors is assessed and the subsequent strain fitting/tensor calculation is completed.
This increases computation time (slightly) but the intent is for the data to be
saved and reloaded/interrogated at leisure.

>>> from pyxe.edi12_analysis import EDI12
>>> fname = r'./test/50418.nxs'
>>> data = EDI12(file = fname, q0 = 3.1, window = 0.5, func = 'gaussian')
File: C:\\Users\\casim\\Dropbox\\Python\\pyxe\\pyxe\\test\\50418.nxs - 195 acquisition points
Progress: [####################] 100%
Total points: 4485 (23 detectors x 195 positions)
Peak not found in 0 position/detector combintions
Error limit exceeded (or pcov not estimated) 0 times

The output from the analysis gives some basic metrics to judge whether the analysis
was successful. The peak fitting is applied to each data point/detector combination
and peak fitting failures are recorded. It is not unusual for the fitting to
have failed at some data point/detector combination. This is particularly likely
if your peak intensity is low or material contains texture such that particular
orientations have low intensity.

The data object that is output contains a lot of the raw data, most notably
the raw q values:

>>> data.I.shape
(13, 15, 24, 4096)

Along with the extracted strain:

>>> data.strain.shape
> (13, 15, 24, 1)

The final dimension of the strain array refers to the number of q0 values that
were given and peaks that were analysed.

The analysed data can be saved back to a NeXus file, which can be reloaded
without the need for re-analysis.

>>> data.save_to_nxs(fname = 'test.nxs')
